"""
Open-loop evaluation for a trained VLA-JEPA policy on a LeRobot v2.1 dataset.

The script samples windows that cover every episode, runs ``predict_action``
(the action head is a flow-matching sampler starting from ``torch.randn``, so a
fixed ``--seed`` is used to keep runs reproducible) and compares the predicted
action chunk against the ground-truth chunk stored in the dataset.

Outputs (written to --output_dir):
  * metrics.json        : overall / per-dimension / per-horizon-step errors,
                          per-episode breakdown, raw-unit errors, R^2, baseline comparison
  * scatter.png         : predicted vs ground-truth per action dimension
  * horizon.png         : error as a function of the chunk position
  * trajectories.png    : GT vs predicted action time series for a few episodes

Usage:
    python scripts/eval_openloop.py \
        --config_yaml checkpoints/pick_open_place_0724/config.yaml \
        --checkpoint  checkpoints/pick_open_place_0724/final_model/pytorch_model.pt \
        --output_dir  eval_openloop/pick_open_place_0724_final \
        --windows_per_episode 8 --batch_size 4
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def parse_args():
    p = argparse.ArgumentParser(description="VLA-JEPA open-loop evaluation")
    p.add_argument("--config_yaml", required=True, help="training config used for the run")
    p.add_argument("--checkpoint", required=True, help="path to a saved pytorch_model.pt")
    p.add_argument("--output_dir", required=True, help="directory for metrics/plots")
    p.add_argument("--windows_per_episode", type=int, default=4,
                   help="how many observation windows to test per episode")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_inference_timesteps", type=int, default=4,
                   help="flow-matching denoising steps of the action head "
                        "(the checkpoint config uses 4; there is no DDIM branch)")
    p.add_argument("--seed", type=int, default=0,
                   help="action sampling starts from torch.randn, so a fixed seed "
                        "is required for reproducible numbers")
    p.add_argument("--num_episodes", type=int, default=0, help="0 = all episodes")
    p.add_argument("--dense_stride", type=int, default=0,
                   help="if >0, ignore --windows_per_episode and sweep every "
                        "N-th frame of each episode (use this for trajectory plots)")
    p.add_argument("--dense_oversample", type=int, default=4,
                   help="with --dense_stride: scan N x len(dataset) indices so the "
                        "frame coverage is ~98%% instead of ~63%%")
    return p.parse_args()


def set_seed(seed):
    """The action head samples its initial noise from ``torch.randn``; without a
    fixed seed two runs of this script would report different numbers."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# model / data
# --------------------------------------------------------------------------- #
def build_model(cfg, checkpoint, num_inference_timesteps):
    model = build_framework(cfg)
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    critical = [k for k in missing if "action_model" in k or "qwen" in k or "vj_" in k]
    print(f"checkpoint       : {checkpoint}")
    print(f"missing keys     : {len(missing)} (critical: {len(critical)})")
    print(f"unexpected keys  : {len(unexpected)}")
    if critical:
        raise RuntimeError(f"critical weights missing from checkpoint, e.g. {critical[:5]}")
    # ``VLA_JEPA.predict_action`` swallows **kwargs, so the old ``num_ddim_steps``
    # argument never reached the action head. The real knob is this attribute.
    model.action_model.num_inference_timesteps = int(num_inference_timesteps)
    print(f"inference steps  : {model.action_model.num_inference_timesteps} "
          f"(flow matching, no DDIM)")
    model = model.cuda().eval()
    return model


def select_windows(dataset, windows_per_episode, num_episodes):
    """Pick deterministic windows spread over every episode (val mode sampling)."""
    by_traj = {}
    for index in range(len(dataset)):
        _, traj_id, step = dataset.sample_step(index)
        by_traj.setdefault(int(traj_id), []).append((int(step), index))

    traj_ids = sorted(by_traj)[: num_episodes or None]
    selected = []
    for traj_id in traj_ids:
        items = sorted(by_traj[traj_id])
        picks = np.linspace(0, len(items) - 1, windows_per_episode).round().astype(int)
        for p in sorted(set(picks.tolist())):
            step, index = items[p]
            selected.append((traj_id, step, index))
    return selected


def select_windows_dense(dataset, stride, num_episodes, oversample=4):
    """Every ``stride``-th frame of each episode.

    ``select_windows`` only guarantees the requested number of windows per
    episode, which is far too sparse to draw a continuous curve. This gives an
    (almost) frame-regular sweep so the predicted chunks can be overlaid on the
    ground-truth trajectory for plotting.

    ``LeRobotMixtureDataset.sample_step`` maps an index to a (trajectory, step)
    pair by *sampling with replacement*, so visiting ``len(dataset)`` indices
    only reaches ~63% of the frames (coupon collector). Iterating
    ``oversample * len(dataset)`` indices lifts that to ~98%.
    """
    by_traj = {}
    for index in range(oversample * len(dataset)):
        _, traj_id, step = dataset.sample_step(index)
        by_traj.setdefault(int(traj_id), {})[int(step)] = index

    traj_ids = sorted(by_traj)[: num_episodes or None]
    selected = []
    for traj_id in traj_ids:
        steps = sorted(by_traj[traj_id])
        for s in steps[::stride]:
            selected.append((traj_id, s, by_traj[traj_id][s]))
    selected.sort(key=lambda w: (w[0], w[1]))
    return selected


def action_stats(dataset):
    """min/max per action dimension, used to map normalized actions back to raw units."""
    stats = dataset.datasets[0].metadata.statistics.action
    keys = list(stats.keys())

    def field(entry, name):
        value = entry[name] if isinstance(entry, dict) else getattr(entry, name)
        return float(np.ravel(value)[0])

    lo = np.array([field(stats[k], "min") for k in keys], dtype=np.float64)
    hi = np.array([field(stats[k], "max") for k in keys], dtype=np.float64)
    return lo, hi, keys


def denormalize(x, lo, hi):
    return (x + 1.0) / 2.0 * (hi - lo) + lo


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def run_inference(model, dataset, windows, batch_size, seed):
    preds, gts, meta = [], [], []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        items = [dataset[index] for _, _, index in chunk]
        images = [it["image"] for it in items]
        langs = [it["lang"] for it in items]
        states = [it["state"] for it in items]
        # Reseed per batch so the result depends only on (seed, batch composition)
        # and not on how the work happened to be split across batches.
        set_seed(seed + start)
        with torch.no_grad():
            out = model.predict_action(
                batch_images=images,
                instructions=langs,
                state=states,
            )
        preds.append(np.asarray(out["normalized_actions"], dtype=np.float64))
        gts.append(np.asarray([it["action"] for it in items], dtype=np.float64))
        meta.extend([(t, s) for t, s, _ in chunk])
        done = min(start + batch_size, len(windows))
        print(f"  inference {done}/{len(windows)}", end="\r", flush=True)
    print()
    return np.concatenate(preds), np.concatenate(gts), meta


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def compute_metrics(pred, gt, lo, hi, meta=None):
    err = pred - gt
    m = {
        "num_windows": int(gt.shape[0]),
        "chunk_len": int(gt.shape[1]),
        "action_dim": int(gt.shape[2]),
        "normalized": {
            "mae": float(np.abs(err).mean()),
            "mse": float((err ** 2).mean()),
            "mae_per_dim": np.abs(err).mean(axis=(0, 1)).tolist(),
            "mae_per_horizon_step": np.abs(err).mean(axis=(0, 2)).tolist(),
        },
        "normalized_baseline_predict_zero": {
            "mae": float(np.abs(gt).mean()),
            "mse": float((gt ** 2).mean()),
        },
    }
    m["normalized"]["mae_improvement_over_zero_baseline"] = float(
        1.0 - m["normalized"]["mae"] / m["normalized_baseline_predict_zero"]["mae"]
    )

    pred_raw, gt_raw = denormalize(pred, lo, hi), denormalize(gt, lo, hi)
    err_raw = pred_raw - gt_raw
    m["raw"] = {
        "mae_per_dim": np.abs(err_raw).mean(axis=(0, 1)).tolist(),
        "rmse_per_dim": np.sqrt((err_raw ** 2).mean(axis=(0, 1))).tolist(),
        "gt_range_per_dim": (hi - lo).tolist(),
        "mae_per_dim_over_gt_range": (
            np.abs(err_raw).mean(axis=(0, 1)) / np.maximum(hi - lo, 1e-8)
        ).tolist(),
    }
    r2 = []
    for d in range(gt.shape[2]):
        g, p = gt[:, :, d].ravel(), pred[:, :, d].ravel()
        ss_res, ss_tot = float(((g - p) ** 2).sum()), float(((g - g.mean()) ** 2).sum())
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    m["normalized"]["r2_per_dim"] = r2

    # Per-episode breakdown: an aggregate MAE hides episodes where the policy
    # fails at the decisive moment (grasp, drawer approach).
    if meta is not None:
        traj = np.array([t for t, _ in meta])
        per_ep = {}
        for ep in sorted(set(traj.tolist())):
            sel = traj == ep
            per_ep[str(int(ep))] = {
                "num_windows": int(sel.sum()),
                "mae": float(np.abs(err[sel]).mean()),
                "mae_gripper": float(np.abs(err[sel, :, 6]).mean()),
                "mae_xyz": float(np.abs(err[sel, :, :3]).mean()),
            }
        m["per_episode"] = per_ep
        worst = sorted(per_ep.items(), key=lambda kv: -kv[1]["mae"])[:5]
        m["worst_episodes"] = [{"episode": int(k), **v} for k, v in worst]
        m["episode_mae_std"] = float(np.std([v["mae"] for v in per_ep.values()]))
    return m


def plot_scatter(pred, gt, r2, out_path):
    n = gt.shape[2]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for d in range(n):
        ax = axes.ravel()[d]
        g, p = gt[:, :, d].ravel(), pred[:, :, d].ravel()
        ax.scatter(g, p, s=4, alpha=0.35)
        lim = [min(g.min(), p.min()), max(g.max(), p.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(f"{DIM_NAMES[d]}  R²={r2[d]:.3f}")
        ax.set_xlabel("ground truth")
        ax.set_ylabel("predicted")
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Open-loop: predicted vs ground-truth actions (min-max normalized)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_horizon(per_step, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(per_step) + 1), per_step, "o-")
    ax.set_xlabel("position inside the predicted action chunk")
    ax.set_ylabel("MAE (normalized)")
    ax.set_title("Open-loop error vs prediction horizon")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectories(pred, gt, meta, out_path, episodes=(0, 1, 2)):
    traj = np.array([m[0] for m in meta])
    step = np.array([m[1] for m in meta])
    n = gt.shape[2]
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.0 * n), sharex=True)
    for d in range(n):
        ax = axes[d]
        for ep in episodes:
            sel = np.where(traj == ep)[0]
            if len(sel) == 0:
                continue
            order = np.argsort(step[sel])
            sel = sel[order]
            ax.plot(step[sel], gt[sel, 0, d], "-o", ms=3, lw=1.2,
                    label=f"ep{ep} GT" if d == 0 else None, alpha=0.9)
            ax.plot(step[sel], pred[sel, 0, d], "--x", ms=3, lw=1.2,
                    label=f"ep{ep} pred" if d == 0 else None, alpha=0.9)
        ax.set_ylabel(DIM_NAMES[d])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("frame index inside episode")
    fig.suptitle("Open-loop rollout (chunk step 0): ground truth vs prediction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_yaml)
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)

    print("=" * 70)
    print("building dataset (mode=val, deterministic windows) ...")
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
    )
    if args.dense_stride > 0:
        windows = select_windows_dense(dataset, args.dense_stride, args.num_episodes,
                                       oversample=args.dense_oversample)
        print(f"dense sweep: every {args.dense_stride}-th frame "
              f"(oversample x{args.dense_oversample})")
    else:
        windows = select_windows(dataset, args.windows_per_episode, args.num_episodes)
    print(f"selected {len(windows)} windows over {len({w[0] for w in windows})} episodes")

    print("building model ...")
    model = build_model(cfg, args.checkpoint, args.num_inference_timesteps)

    print("running inference ...")
    pred, gt, meta = run_inference(model, dataset, windows, args.batch_size, args.seed)

    lo, hi, keys = action_stats(dataset)
    print(f"action keys: {keys}")
    metrics = compute_metrics(pred, gt, lo, hi, meta)
    metrics["config"] = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "windows_per_episode": args.windows_per_episode,
        "dense_stride": args.dense_stride,
        "dense_oversample": args.dense_oversample,
        "num_inference_timesteps": args.num_inference_timesteps,
        "seed": args.seed,
        "num_episodes": len({w[0] for w in windows}),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_scatter(pred, gt, metrics["normalized"]["r2_per_dim"], out_dir / "scatter.png")
    plot_horizon(metrics["normalized"]["mae_per_horizon_step"], out_dir / "horizon.png")
    plot_trajectories(pred, gt, meta, out_dir / "trajectories.png")
    np.savez(out_dir / "predictions.npz", pred=pred, gt=gt,
             meta=np.array(meta), action_min=lo, action_max=hi)

    print("-" * 70)
    print(f"windows                 : {metrics['num_windows']} "
          f"over {metrics['config']['num_episodes']} episodes")
    print(f"normalized MAE / MSE    : {metrics['normalized']['mae']:.4f} / "
          f"{metrics['normalized']['mse']:.4f}")
    print(f"MAE of zero baseline    : {metrics['normalized_baseline_predict_zero']['mae']:.4f}"
          f"  -> improvement {100 * metrics['normalized']['mae_improvement_over_zero_baseline']:.1f}%")
    print(f"raw MAE per joint       : "
          f"{np.round(metrics['raw']['mae_per_dim'], 4).tolist()}")
    print(f"R^2 per dim             : {np.round(metrics['normalized']['r2_per_dim'], 3).tolist()}")
    print(f"per-episode MAE std     : {metrics['episode_mae_std']:.4f}")
    print("worst 5 episodes (normalized MAE):")
    for w in metrics["worst_episodes"]:
        print(f"  ep {w['episode']:>3}  mae={w['mae']:.4f}  "
              f"xyz={w['mae_xyz']:.4f}  gripper={w['mae_gripper']:.4f}  n={w['num_windows']}")
    print(f"artifacts               : {out_dir}")


if __name__ == "__main__":
    main()
