#!/usr/bin/env python
"""Phase-1 driver for the 8-task ICLR open-loop evaluation.

Runs ``scripts/eval_openloop.py`` + ``scripts/analyze_openloop.py`` for every
(run, checkpoint) pair, spread over the 4 idle RTX 4090s (longest-job-first so
the queues finish at roughly the same time).

Resumable: a job whose ``metrics.json`` already contains a positive
``num_windows`` is skipped, so the driver can simply be restarted after an
interruption (partial output dirs are cleaned first).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA")
PY = "/opt/conda/envs/VLA_JEPA/bin/python"
LOG_DIR = ROOT / "eval_openloop" / "logs"
STATUS_DIR = ROOT / "eval_openloop" / "report_data" / "phase1_status"
STATUS = ROOT / "eval_openloop" / "report_data" / "phase1_status.json"
N_GPUS = 4
# Two concurrent processes per 4090 fit easily (7.6 GB each of 24 GB) and roughly
# halve the wall time; set PROCS_PER_GPU=1 for the conservative single-process mode.
PROCS_PER_GPU = int(os.environ.get("PROCS_PER_GPU", "1"))
N_QUEUES = N_GPUS * PROCS_PER_GPU
RUN_TIMEOUT = 5400  # seconds per eval_openloop.py call

# run_id -> episode count (cost proxy: episodes * 8 windows)
RUNS = {
    "iclr_adjust_cup": 50,
    "iclr_open_cabinet": 50,
    "iclr_pick_banana_newtable": 100,
    "iclr_pick_banana_pot": 48,
    "iclr_pick_block": 100,
    "iclr_pick_eggplant_cluttered": 87,
    "iclr_pick_eggplant_drawer": 50,
    "iclr_sponge_wipe": 49,
}


def checkpoints(run: str):
    ckpt_dir = ROOT / "checkpoints" / run / "checkpoints"
    steps = sorted(
        int(p.name.split("_")[1]) for p in ckpt_dir.glob("steps_*_pytorch_model.pt")
    )
    return steps


def build_jobs():
    jobs = []
    for run, n_eps in RUNS.items():
        for step in checkpoints(run):
            out = ROOT / "eval_openloop" / run / f"steps_{step}"
            jobs.append(
                {
                    "run": run,
                    "step": step,
                    "out": out,
                    "cfg": ROOT / "checkpoints" / run / "config.yaml",
                    "ckpt": ROOT / "checkpoints" / run / "checkpoints"
                    / f"steps_{step}_pytorch_model.pt",
                    "cost": n_eps * 8,
                }
            )
    jobs.sort(key=lambda j: -j["cost"])
    return jobs


def done(job):
    m = job["out"] / "metrics.json"
    if not m.exists():
        return False
    try:
        with open(m) as f:
            data = json.load(f)
        if int(data.get("num_windows", 0)) <= 0:
            return False
        # partial npz (interrupted during save) -> redo
        npz = job["out"] / "predictions.npz"
        return npz.exists() and npz.stat().st_size > 0
    except Exception:
        return False


def reset(job):
    if job["out"].exists():
        shutil.rmtree(job["out"])


def run_queue(gpu, queue):
    for job in queue:
        tag = f"{job['run']}_steps_{job['step']}"
        log = LOG_DIR / f"{tag}.log"
        if done(job):
            write_job_status(tag, {"state": "skipped(done)", "gpu": gpu})
            print(f"[gpu{gpu}] {tag} -> skipped(done)", flush=True)
            continue
        reset(job)
        write_job_status(tag, {"state": "running", "gpu": gpu,
                               "started": time.strftime("%F %T")})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        cmd1 = [
            PY, "scripts/eval_openloop.py",
            "--config_yaml", str(job["cfg"]),
            "--checkpoint", str(job["ckpt"]),
            "--output_dir", str(job["out"]),
            "--windows_per_episode", "8", "--batch_size", "4", "--seed", "0",
        ]
        cmd2 = [PY, "scripts/analyze_openloop.py",
                "--predictions", str(job["out"] / "predictions.npz")]
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write(f"# GPU {gpu} :: {' '.join(cmd1)}\n")
            lf.flush()
            try:
                rc1 = subprocess.run(cmd1, cwd=ROOT, env=env, timeout=RUN_TIMEOUT,
                                     stdout=lf, stderr=subprocess.STDOUT).returncode
            except subprocess.TimeoutExpired:
                rc1 = -9
                lf.write(f"\n!! TIMEOUT after {RUN_TIMEOUT}s\n")
            if rc1 == 0:
                lf.write("\n# " + " ".join(cmd2) + "\n")
                lf.flush()
                rc2 = subprocess.run(cmd2, cwd=ROOT, env=env,
                                     stdout=lf, stderr=subprocess.STDOUT).returncode
            else:
                rc2 = None
        rec = {
            "state": "ok" if (rc1 == 0 and rc2 == 0) else "FAILED",
            "gpu": gpu, "rc_eval": rc1, "rc_analyze": rc2,
            "minutes": round((time.time() - t0) / 60, 1), "log": str(log),
        }
        write_job_status(tag, rec)
        print(f"[gpu{gpu}] {tag} -> {rec['state']} ({rec['minutes']} min)", flush=True)


def write_job_status(tag, rec):
    """One file per job -> the 4 queue processes never fight over a shared file."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_DIR / f"{tag}.tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    tmp.replace(STATUS_DIR / f"{tag}.json")


def collect_status():
    out = {}
    for f in sorted(STATUS_DIR.glob("*.json")):
        try:
            out[f.stem] = json.load(open(f))
        except Exception:
            pass
    return out


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    queues = [[] for _ in range(N_QUEUES)]
    load = [0] * N_QUEUES
    for job in jobs:  # LPT: jobs already sorted longest-first
        i = load.index(min(load))
        queues[i].append(job)
        load[i] += job["cost"]
    print(f"{len(jobs)} jobs over {N_QUEUES} queues "
          f"({PROCS_PER_GPU}/GPU), estimated window load: {load}", flush=True)
    for q_i, q in enumerate(queues):
        print(f"  gpu{q_i % N_GPUS}/q{q_i}: "
              + ", ".join(f"{j['run']}@{j['step']}" for j in q), flush=True)

    # one OS process per GPU queue; children re-enter this script via --queue
    procs = []
    for q_i, q in enumerate(queues):
        spec = json.dumps([{k: str(v) if isinstance(v, Path) else v
                            for k, v in j.items()} for j in q])
        procs.append(subprocess.Popen(
            [PY, __file__, "--queue", str(q_i % N_GPUS), spec],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    rc = 0
    for p in procs:
        rc |= p.wait()
    status = collect_status()
    with open(STATUS, "w") as f:
        json.dump(status, f, indent=2)
    n_ok = sum(1 for v in status.values() if v.get("state") in ("ok", "skipped(done)"))
    n_fail = sum(1 for v in status.values() if v.get("state") == "FAILED")
    print(f"phase1 done: ok/skipped={n_ok} failed={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--queue":
        gpu = int(sys.argv[2])
        queue = json.loads(sys.argv[3])
        for j in queue:
            j["out"] = Path(j["out"])
            j["cfg"] = Path(j["cfg"])
            j["ckpt"] = Path(j["ckpt"])
        run_queue(gpu, queue)
    else:
        sys.exit(main())
