#!/usr/bin/env python
"""How much of the open-loop error is action-head sampling noise, and can it be
reduced?

Follow-up to ``measure_sampling_noise.py``. For one task it measures, on the same
windows:

  * the prediction std across seeds (sampling noise) at the configured
    ``num_inference_timesteps=4``;
  * the MAE of a single sample vs the MAE of the K-seed average (temporal
    ensembling);
  * the same at ``num_inference_timesteps`` = 1/2/4/8/16, to see whether the
    4-step flow-matching sampler is the limiting factor;
  * the state-hold baseline on the very same windows, so the step-0 comparison
    is apples-to-apples.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/openloop/measure_sampler_options.py \
        --config_yaml checkpoints/iclr_pick_banana_pot/config.yaml \
        --checkpoint  checkpoints/iclr_pick_banana_pot/checkpoints/steps_20000_pytorch_model.pt \
        --num_episodes 4 --windows_per_episode 8 --num_seeds 4 \
        --output eval_openloop/report_data/sampler_options_banana_pot.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path("/share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA")
sys.path.insert(0, str(ROOT))
from starVLA.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402
from starVLA.model.framework import build_framework  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_episodes", type=int, default=4)
    ap.add_argument("--windows_per_episode", type=int, default=8)
    ap.add_argument("--num_seeds", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--timestep_list", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    print("building dataset ...", flush=True)
    dataset = get_vla_dataset(data_cfg=cfg.datasets.vla_data, mode="val",
                              action_horizon=cfg.framework.action_model.action_horizon,
                              video_horizon=cfg.framework.vj2_model.num_frames)
    data_root = Path(cfg.datasets.vla_data.data_root_dir)

    # same window rule as eval_openloop.select_windows (first N episodes)
    by_traj = {}
    for index in range(len(dataset)):
        _, traj_id, step = dataset.sample_step(index)
        by_traj.setdefault(int(traj_id), []).append((int(step), index))
    sel = []
    for traj_id in sorted(by_traj)[: args.num_episodes]:
        items = sorted(by_traj[traj_id])
        picks = np.linspace(0, len(items) - 1, args.windows_per_episode).round().astype(int)
        for p in sorted(set(picks.tolist())):
            sel.append((traj_id, items[p][0], items[p][1]))
    print(f"{len(sel)} windows over {args.num_episodes} episodes", flush=True)

    print("building model ...", flush=True)
    model = build_framework(cfg)
    sd = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing {len(missing)} / unexpected {len(unexpected)}", flush=True)
    model = model.cuda().eval()

    items = [dataset[i] for _, _, i in sel]
    gt = np.stack([np.asarray(it["action"], dtype=np.float64) for it in items])

    # state-hold baseline on exactly these windows
    # same accessor as eval_openloop.action_stats: entries are dict-or-object
    stats = dataset.datasets[0].metadata.statistics.action
    keys = list(stats.keys())

    def field(entry, name):
        value = entry[name] if isinstance(entry, dict) else getattr(entry, name)
        return float(np.ravel(value)[0])

    lo = np.array([field(stats[k], "min") for k in keys], dtype=np.float64)
    hi = np.array([field(stats[k], "max") for k in keys], dtype=np.float64)
    cache, states = {}, []
    for traj, step, _ in sel:
        if traj not in cache:
            cache[traj] = pd.read_parquet(
                data_root / "data" / "chunk_00000" / f"episode_{traj:06d}.parquet")
        states.append(np.asarray(cache[traj]["observation.state"].iloc[step].tolist(),
                                 dtype=np.float64))
    state_norm = 2.0 * (np.stack(states) - lo) / (hi - lo) - 1.0
    hold = np.repeat(state_norm[:, None, :], gt.shape[1], axis=1)

    res = {"config_yaml": args.config_yaml, "checkpoint": args.checkpoint,
           "num_windows": len(sel), "num_seeds": args.num_seeds,
           "num_episodes": args.num_episodes,
           "hold_mae_full": float(np.abs(hold - gt).mean()),
           "hold_mae_step0": float(np.abs(hold[:, 0] - gt[:, 0]).mean()),
           "hold_mae_per_step": np.abs(hold - gt).mean(axis=(0, 2)).tolist(),
           "by_timesteps": {}}

    for n_steps in args.timestep_list:
        model.action_model.num_inference_timesteps = int(n_steps)
        runs = []
        for k in range(args.num_seeds):
            preds = []
            for start in range(0, len(sel), args.batch_size):
                batch = sel[start:start + args.batch_size]
                bitems = [dataset[i] for _, _, i in batch]
                set_seed(1000 * k + start)   # same noise per repeat across n_steps
                with torch.no_grad():
                    out = model.predict_action(
                        batch_images=[it["image"] for it in bitems],
                        instructions=[it["lang"] for it in bitems],
                        state=[it["state"] for it in bitems])  # NOTE: state= (see doc §8.2)
                preds.append(np.asarray(out["normalized_actions"], dtype=np.float64))
            runs.append(np.concatenate(preds))
            print(f"  steps={n_steps} seed pass {k + 1}/{args.num_seeds}", flush=True)

        stack = np.stack(runs)
        single = [float(np.abs(r - gt).mean()) for r in runs]
        single0 = [float(np.abs(r[:, 0] - gt[:, 0]).mean()) for r in runs]
        mean_pred = stack.mean(axis=0)
        res["by_timesteps"][str(n_steps)] = {
            "single_run_mae_full": single,
            "single_run_mae_full_mean": float(np.mean(single)),
            "single_run_mae_step0_mean": float(np.mean(single0)),
            "ensemble_mae_full": float(np.abs(mean_pred - gt).mean()),
            "ensemble_mae_step0": float(np.abs(mean_pred[:, 0] - gt[:, 0]).mean()),
            "sampling_std_overall": float(stack.std(axis=0).mean()),
            "sampling_std_step0": float(stack.std(axis=0)[:, 0].mean()),
            "ensemble_mae_per_step": np.abs(mean_pred - gt).mean(axis=(0, 2)).tolist(),
        }
        r = res["by_timesteps"][str(n_steps)]
        print(f"  >> steps={n_steps}: single MAE {r['single_run_mae_full_mean']:.4f} "
              f"(step0 {r['single_run_mae_step0_mean']:.4f}) | "
              f"ensemble({args.num_seeds}) MAE {r['ensemble_mae_full']:.4f} "
              f"(step0 {r['ensemble_mae_step0']:.4f}) | std {r['sampling_std_overall']:.4f}",
              flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print("written ->", args.output)


if __name__ == "__main__":
    main()
