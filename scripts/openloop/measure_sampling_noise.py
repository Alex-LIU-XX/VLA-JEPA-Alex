#!/usr/bin/env python
"""Measure the flow-matching sampling noise floor of the action head.

Motivation: the open-loop report shows that (a) the model's MAE is almost flat
across the 7-step chunk and (b) its step-0 error (~0.024-0.029 normalized) is
larger than the state-hold baseline on 7 of 8 tasks. Both are consistent with the
action head having an intrinsic sampling variance: ``predict_action`` starts from
``torch.randn`` and denoises with only ``num_inference_timesteps=4`` steps.

This script quantifies that directly: the *same* observation windows are passed
through the model K times with different seeds, and the standard deviation of the
predictions across seeds is the noise floor (the irreducible part of the error
that even a perfect policy would show under this sampler).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/openloop/measure_sampling_noise.py \
        --config_yaml checkpoints/iclr_sponge_wipe/config.yaml \
        --checkpoint  checkpoints/iclr_sponge_wipe/checkpoints/steps_20000_pytorch_model.pt \
        --num_windows 32 --num_seeds 8 \
        --output eval_openloop/report_data/sampling_noise.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
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
    ap.add_argument("--num_windows", type=int, default=32)
    ap.add_argument("--num_seeds", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_inference_timesteps", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    print("building dataset ...", flush=True)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data, mode="val",
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames)

    # deterministic windows spread over episodes (same rule as select_windows)
    by_traj = {}
    for index in range(len(dataset)):
        _, traj_id, step = dataset.sample_step(index)
        by_traj.setdefault(int(traj_id), []).append((int(step), index))
    picks = []
    for traj_id in sorted(by_traj):
        items = sorted(by_traj[traj_id])
        for p in np.linspace(0, len(items) - 1, 4).round().astype(int):
            picks.append(items[p][1])
        if len(picks) >= args.num_windows:
            break
    picks = picks[: args.num_windows]
    print(f"{len(picks)} windows selected", flush=True)

    print("building model ...", flush=True)
    model = build_framework(cfg)
    sd = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"missing {len(missing)} / unexpected {len(unexpected)}", flush=True)
    model.action_model.num_inference_timesteps = int(args.num_inference_timesteps)
    model = model.cuda().eval()

    runs = []  # [K][W, T, D]
    for k in range(args.num_seeds):
        preds = []
        for start in range(0, len(picks), args.batch_size):
            batch = picks[start:start + args.batch_size]
            items = [dataset[i] for i in batch]
            set_seed(1000 * k + start)  # independent noise per repeat
            with torch.no_grad():
                # NOTE: the keyword must be ``state=`` -- ``VLA_JEPA.predict_action``
                # absorbs unknown kwargs silently (doc §8.2), so ``states=`` would
                # drop proprioception without any error and wreck the metrics.
                out = model.predict_action(
                    batch_images=[it["image"] for it in items],
                    instructions=[it["lang"] for it in items],
                    state=[it["state"] for it in items])
            preds.append(np.asarray(out["normalized_actions"], dtype=np.float64))
        runs.append(np.concatenate(preds))
        print(f"  seed pass {k + 1}/{args.num_seeds}", flush=True)

    stack = np.stack(runs)                       # [K, W, T, D]
    gt = np.stack([np.asarray(dataset[i]["action"], dtype=np.float64) for i in picks])
    sd_across = stack.std(axis=0)                # [W, T, D]
    mean_pred = stack.mean(axis=0)

    # how much of the step-0 error could be explained by sampling noise alone
    step0_err = np.abs(mean_pred[:, 0] - gt[:, 0]).mean()
    res = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "num_windows": len(picks),
        "num_seeds": args.num_seeds,
        "num_inference_timesteps": args.num_inference_timesteps,
        "sampling_std_overall": float(sd_across.mean()),
        "sampling_std_per_step": sd_across.mean(axis=(0, 2)).tolist(),
        "sampling_std_per_dim": sd_across.mean(axis=(0, 1)).tolist(),
        "mean_pred_step0_mae_vs_gt": float(step0_err),
        "mean_pred_mae_vs_gt": float(np.abs(mean_pred - gt).mean()),
        "single_run_mae_vs_gt": [
            float(np.abs(r - gt).mean()) for r in runs],
        "mean_pred_step0_mae_per_dim": np.abs(mean_pred[:, 0] - gt[:, 0]).mean(axis=0).tolist(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("single_run_mae_vs_gt",)}, indent=2))
    print("written ->", args.output)


if __name__ == "__main__":
    main()
