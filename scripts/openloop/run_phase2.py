#!/usr/bin/env python
"""Phase-2 driver: dense (per-frame) sweep + trajectory figures for the 8 final
checkpoints, so each task gets a continuous predicted-vs-GT trajectory plot.

Runs ``eval_openloop.py --dense_stride 1 --num_episodes 4 --dense_oversample 4``
followed by ``plot_openloop_trajectory.py``. Same 4-GPU queue scheme as
``run_phase1.py``, and equally resumable.
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
STATUS_DIR = ROOT / "eval_openloop" / "report_data" / "phase2_status"
N_GPUS = 4
RUN_TIMEOUT = 7200

RUNS = {
    "iclr_adjust_cup": 50, "iclr_open_cabinet": 50,
    "iclr_pick_banana_newtable": 100, "iclr_pick_banana_pot": 48,
    "iclr_pick_block": 100, "iclr_pick_eggplant_cluttered": 87,
    "iclr_pick_eggplant_drawer": 50, "iclr_sponge_wipe": 49,
}


def final_step(run):
    return max(int(p.name.split("_")[1])
               for p in (ROOT / "checkpoints" / run / "checkpoints").glob(
                   "steps_*_pytorch_model.pt"))


def build_jobs():
    jobs = []
    for run in RUNS:
        step = final_step(run)
        jobs.append({
            "run": run, "step": step,
            "npz": ROOT / "eval_openloop" / run / ("steps_%d_dense" % step) / "predictions.npz",
            "out": ROOT / "eval_openloop" / run / ("steps_%d_dense" % step),
            "cfg": ROOT / "checkpoints" / run / "config.yaml",
            "ckpt": ROOT / "checkpoints" / run / "checkpoints"
                    / f"steps_{step}_pytorch_model.pt",
            "cost": 1 if run in ("iclr_pick_block", "iclr_pick_banana_newtable") else 0,
        })
    return jobs


def done(job):
    npz = job["npz"]
    if not npz.exists() or npz.stat().st_size == 0:
        return False
    return (job["out"] / "trajectory_fit.png").exists()


def write_job_status(tag, rec):
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_DIR / f"{tag}.tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    tmp.replace(STATUS_DIR / f"{tag}.json")


def run_queue(gpu, queue):
    for job in queue:
        tag = f"{job['run']}_steps_{job['step']}_dense"
        log = LOG_DIR / f"{tag}.log"
        if done(job):
            write_job_status(tag, {"state": "skipped(done)", "gpu": gpu})
            print(f"[gpu{gpu}] {tag} -> skipped(done)", flush=True)
            continue
        if job["out"].exists():
            shutil.rmtree(job["out"])
        write_job_status(tag, {"state": "running", "gpu": gpu,
                               "started": time.strftime("%F %T")})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        cmd1 = [PY, "scripts/eval_openloop.py",
                "--config_yaml", str(job["cfg"]), "--checkpoint", str(job["ckpt"]),
                "--output_dir", str(job["out"]),
                "--dense_stride", "1", "--num_episodes", "4",
                "--dense_oversample", "4", "--batch_size", "4", "--seed", "0"]
        cmd2 = [PY, "scripts/plot_openloop_trajectory.py",
                "--predictions", str(job["npz"]), "--out_dir", str(job["out"]),
                "--episodes", "0", "1", "2", "--zoom_episode", "0"]
        t0 = time.time()
        with open(log, "w") as lf:
            lf.write(f"# GPU {gpu} :: {' '.join(cmd1)}\n")
            lf.flush()
            try:
                rc1 = subprocess.run(cmd1, cwd=ROOT, env=env, timeout=RUN_TIMEOUT,
                                     stdout=lf, stderr=subprocess.STDOUT).returncode
            except subprocess.TimeoutExpired:
                rc1, = (-9,)
                lf.write(f"\n!! TIMEOUT after {RUN_TIMEOUT}s\n")
            rc2 = None
            if rc1 == 0:
                lf.write("\n# " + " ".join(cmd2) + "\n")
                lf.flush()
                rc2 = subprocess.run(cmd2, cwd=ROOT, env=env,
                                     stdout=lf, stderr=subprocess.STDOUT).returncode
        rec = {"state": "ok" if (rc1 == 0 and rc2 == 0) else "FAILED",
               "gpu": gpu, "rc_eval": rc1, "rc_plot": rc2,
               "minutes": round((time.time() - t0) / 60, 1), "log": str(log)}
        write_job_status(tag, rec)
        print(f"[gpu{gpu}] {tag} -> {rec['state']} ({rec['minutes']} min)", flush=True)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    queues = [[] for _ in range(N_GPUS)]
    load = [0] * N_GPUS
    for job in sorted(jobs, key=lambda j: -j["cost"]):
        i = load.index(min(load))
        queues[i].append(job)
        load[i] += job["cost"] + 1
    for g, q in enumerate(queues):
        print(f"  gpu{g}: " + ", ".join(j["run"] for j in q), flush=True)
    procs = []
    for g, q in enumerate(queues):
        spec = json.dumps([{k: str(v) if isinstance(v, Path) else v
                            for k, v in j.items()} for j in q])
        procs.append(subprocess.Popen([PY, __file__, "--queue", str(g), spec],
                                      cwd=ROOT, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.STDOUT))
    rc = 0
    for p in procs:
        rc |= p.wait()
    st = {}
    for f in sorted(STATUS_DIR.glob("*.json")):
        st[f.stem] = json.load(open(f))
    json.dump(st, open(ROOT / "eval_openloop/report_data/phase2_status.json", "w"), indent=2)
    n_fail = sum(1 for v in st.values() if v.get("state") == "FAILED")
    print(f"phase2 done: {len(st)} jobs, failed={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--queue":
        gpu = int(sys.argv[2])
        queue = json.loads(sys.argv[3])
        for j in queue:
            for k in ("npz", "out", "cfg", "ckpt"):
                j[k] = Path(j[k])
        run_queue(gpu, queue)
    else:
        sys.exit(main())
