"""
Extra analysis for an open-loop run produced by ``scripts/eval_openloop.py``.

``eval_openloop.py`` already reports the zero baseline ("always predict 0").
That baseline is weak, because a min-max normalised action of 0 is the middle
of the range. This script adds the two comparisons that actually decide whether
a chunk prediction is useful:

  * persistence baseline -- repeat the *current* ground-truth action across the
    whole chunk (i.e. "assume the robot keeps doing what it is doing now").
    A policy that cannot beat this has learned nothing about the future.
  * step-0 metrics -- a receding-horizon controller only executes the first few
    actions of a chunk, so the error at chunk position 0 matters far more than
    the error at position 6.
  * gripper open/close decision accuracy -- for pick-and-place the gripper
    command is the most consequential and most discrete channel.

Usage:
    python scripts/analyze_openloop.py \
        --predictions eval_openloop/pick_open_place_0724_final/predictions.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def mae(a, b):
    return float(np.abs(a - b).mean())


def state_hold_baseline(meta, lo, hi, data_root_dir):
    """Repeat the *measured state* across the whole chunk.

    This is the honest "do nothing" baseline: at inference time the policy only
    has the current state, not the ground-truth current action, so this is what
    a hold-position controller would actually command.

    ``state`` is deliberately left un-normalised by ``PiperDataConfig`` (only the
    action keys go through ``StateActionTransform``), so it is mapped into the
    action's min-max range here before comparing.
    """
    import pandas as pd

    cache, rows = {}, []
    for traj, step in meta:
        traj, step = int(traj), int(step)
        if traj not in cache:
            path = Path(data_root_dir) / "data" / "chunk_00000" / f"episode_{traj:06d}.parquet"
            cache[traj] = pd.read_parquet(path)
        rows.append(np.asarray(cache[traj]["observation.state"].iloc[step].tolist(),
                               dtype=np.float64))
    state = np.stack(rows)
    state_norm = 2.0 * (state - lo) / (hi - lo) - 1.0
    return state_norm


def analyze(npz_path, out_dir=None, gripper_threshold=0.0, data_root_dir=None):
    data = np.load(npz_path, allow_pickle=True)
    pred, gt, meta = data["pred"], data["gt"], data["meta"]
    lo, hi = data["action_min"], data["action_max"]
    assert pred.shape == gt.shape, (pred.shape, gt.shape)

    traj = np.array([m[0] for m in meta])
    steps = np.array([m[1] for m in meta])
    rep = {"predictions": str(npz_path), "num_windows": int(pred.shape[0]),
           "chunk_len": int(pred.shape[1]), "action_dim": int(pred.shape[2])}

    # ---- model vs the two baselines, on the full chunk and on step 0 only ----
    zero = np.zeros_like(gt)
    persist = np.repeat(gt[:, 0:1, :], gt.shape[1], axis=1)  # hold current action

    rep["full_chunk"] = {
        "model_mae": mae(pred, gt),
        "zero_baseline_mae": mae(zero, gt),
        "persistence_baseline_mae": mae(persist, gt),
    }
    rep["full_chunk"]["model_vs_persistence_improvement"] = float(
        1.0 - rep["full_chunk"]["model_mae"] / rep["full_chunk"]["persistence_baseline_mae"]
    )
    rep["full_chunk"]["model_vs_zero_improvement"] = float(
        1.0 - rep["full_chunk"]["model_mae"] / rep["full_chunk"]["zero_baseline_mae"]
    )

    # Receding-horizon view: execute the first k actions, average over k = 1..3.
    rep["by_prefix"] = {}
    for k in (1, 2, 3, 7):
        rep["by_prefix"][f"first_{k}_steps"] = {
            "model_mae": mae(pred[:, :k], gt[:, :k]),
            "persistence_mae": mae(persist[:, :k], gt[:, :k]),
            "zero_mae": mae(zero[:, :k], gt[:, :k]),
        }

    # ---- the honest "do nothing" baseline: hold the current measured state ----
    if data_root_dir:
        try:
            state_norm = state_hold_baseline(meta, lo, hi, data_root_dir)
            hold = np.repeat(state_norm[:, None, :], gt.shape[1], axis=1)
            rep["state_hold_baseline"] = {
                "source": str(data_root_dir),
                "model_mae": mae(pred, gt),
                "hold_mae": mae(hold, gt),
                "improvement": float(1.0 - mae(pred, gt) / mae(hold, gt)),
                "per_step": [
                    {"step": k,
                     "model_mae": mae(pred[:, k], gt[:, k]),
                     "hold_mae": mae(hold[:, k], gt[:, k]),
                     "model_better": bool(mae(pred[:, k], gt[:, k]) < mae(hold[:, k], gt[:, k]))}
                    for k in range(gt.shape[1])
                ],
                "window_slices": {
                    name: {
                        "model_mae": mae(pred[:, sl], gt[:, sl]),
                        "hold_mae": mae(hold[:, sl], gt[:, sl]),
                        "improvement": float(
                            1.0 - mae(pred[:, sl], gt[:, sl]) / mae(hold[:, sl], gt[:, sl])
                        ),
                    }
                    for name, sl in (("step0", slice(0, 1)),
                                     ("steps0_2", slice(0, 3)),
                                     ("steps3_6", slice(3, 7)),
                                     ("full_chunk", slice(None)))
                },
                "per_dim": {
                    name: {
                        "model_mae": mae(pred[:, :, d], gt[:, :, d]),
                        "hold_mae": mae(hold[:, :, d], gt[:, :, d]),
                        "improvement": float(
                            1.0 - mae(pred[:, :, d], gt[:, :, d]) / mae(hold[:, :, d], gt[:, :, d])
                        ),
                    }
                    for d, name in enumerate(DIM_NAMES)
                },
            }
            if out_dir:
                np.savez(Path(out_dir) / "baselines.npz", hold=hold, state_norm=state_norm)
        except Exception as exc:  # dataset moved / not available -> skip, not fatal
            rep["state_hold_baseline"] = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- per-dimension model vs persistence ----
    rep["per_dim"] = {}
    for d, name in enumerate(DIM_NAMES):
        rep["per_dim"][name] = {
            "model_mae": mae(pred[:, :, d], gt[:, :, d]),
            "persistence_mae": mae(persist[:, :, d], gt[:, :, d]),
            "model_mae_step0": mae(pred[:, 0, d], gt[:, 0, d]),
            "persistence_mae_step0": mae(persist[:, 0, d], gt[:, 0, d]),
        }

    # ---- gripper as a discrete decision ----
    g_pred, g_gt = pred[:, :, 6], gt[:, :, 6]
    open_pred, open_gt = g_pred > gripper_threshold, g_gt > gripper_threshold
    rep["gripper_decision"] = {
        "threshold": gripper_threshold,
        "accuracy": float((open_pred == open_gt).mean()),
        "gt_open_fraction": float(open_gt.mean()),
        "pred_open_fraction": float(open_pred.mean()),
        "false_open_rate": float((open_pred & ~open_gt).mean()),   # predicted open, really closed
        "false_closed_rate": float((~open_pred & open_gt).mean()),  # predicted closed, really open
    }

    # ---- error growth along the chunk, averaged over windows ----
    per_step = np.abs(pred - gt).mean(axis=(0, 2))
    rep["mae_per_horizon_step"] = per_step.tolist()
    rep["mae_growth_step0_to_last"] = float(per_step[-1] / max(per_step[0], 1e-9))

    rep["per_episode_range"] = {
        "min": float(min(np.abs(pred[traj == e] - gt[traj == e]).mean()
                         for e in sorted(set(traj.tolist())))),
        "max": float(max(np.abs(pred[traj == e] - gt[traj == e]).mean()
                         for e in sorted(set(traj.tolist())))),
    }

    # ---- sanity: does the model just echo the current state? ----
    # Correlation between the prediction and the current GT action, per window.
    p_flat = pred.reshape(pred.shape[0], -1)
    c_flat = persist.reshape(persist.shape[0], -1)
    cors = []
    for i in range(pred.shape[0]):
        a, b = p_flat[i], c_flat[i]
        if a.std() > 1e-9 and b.std() > 1e-9:
            cors.append(float(np.corrcoef(a, b)[0, 1]))
    rep["pred_vs_persistence_corr"] = {
        "mean": float(np.mean(cors)) if cors else float("nan"),
        "median": float(np.median(cors)) if cors else float("nan"),
    }

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "extra_metrics.json", "w") as f:
            json.dump(rep, f, indent=2)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="predictions.npz from eval_openloop.py")
    ap.add_argument("--out_dir", default=None, help="defaults to the npz directory")
    ap.add_argument("--data_root_dir", default=None,
                    help="LeRobot v2.1 dataset dir; defaults to the value recorded "
                         "in the sibling metrics.json")
    args = ap.parse_args()

    out_dir = args.out_dir or str(Path(args.predictions).parent)

    data_root_dir = args.data_root_dir
    if data_root_dir is None:
        metrics_json = Path(out_dir) / "metrics.json"
        if metrics_json.exists():
            with open(metrics_json) as f:
                data_root_dir = json.load(f).get("config", {}).get("data_root_dir")

    r = analyze(args.predictions, out_dir, data_root_dir=data_root_dir)

    print("=" * 68)
    print(f"windows: {r['num_windows']}   chunk: {r['chunk_len']}   dim: {r['action_dim']}")
    print("-" * 68)
    f = r["full_chunk"]
    print(f"full chunk  model MAE      : {f['model_mae']:.4f}")
    print(f"            zero baseline  : {f['zero_baseline_mae']:.4f}"
          f"   -> improvement {100 * f['model_vs_zero_improvement']:.1f}%")
    print(f"            persist baseline: {f['persistence_baseline_mae']:.4f}"
          f"   -> improvement {100 * f['model_vs_persistence_improvement']:.1f}%")
    print("-" * 68)
    print("receding-horizon (only the first k actions get executed):")
    for k, v in r["by_prefix"].items():
        print(f"  {k:<14} model={v['model_mae']:.4f}  persist={v['persistence_mae']:.4f}"
              f"  zero={v['zero_mae']:.4f}")
    print("-" * 68)
    print("per dimension (model vs persistence):")
    for d in DIM_NAMES:
        v = r["per_dim"][d]
        print(f"  {d:<8} model={v['model_mae']:.4f}  persist={v['persistence_mae']:.4f}"
              f"   step0: model={v['model_mae_step0']:.4f} persist={v['persistence_mae_step0']:.4f}")
    print("-" * 68)
    sh = r.get("state_hold_baseline", {})
    if "error" in sh:
        print(f"state-hold baseline unavailable: {sh['error']}")
    elif sh:
        print("state-hold baseline (repeat the CURRENT MEASURED STATE -- the honest "
              "'do nothing'):")
        print(f"  full chunk  model {sh['model_mae']:.4f} vs hold {sh['hold_mae']:.4f}"
              f"  -> model better by {100 * sh['improvement']:.1f}%")
        for name, v in sh["window_slices"].items():
            flag = "" if v["improvement"] > 0 else "   <-- WORSE THAN DOING NOTHING"
            print(f"  {name:<12} model {v['model_mae']:.4f} vs hold {v['hold_mae']:.4f}"
                  f"  -> {100 * v['improvement']:6.1f}%{flag}")
        print("  per step (model vs hold):")
        for s in sh["per_step"]:
            flag = "" if s["model_better"] else "  <-- worse than hold"
            print(f"    step {s['step']}: model {s['model_mae']:.4f}  hold {s['hold_mae']:.4f}{flag}")
    print("-" * 68)
    g = r["gripper_decision"]
    print(f"gripper open/close accuracy: {100 * g['accuracy']:.1f}%"
          f"  (gt open {100 * g['gt_open_fraction']:.1f}%, pred open {100 * g['pred_open_fraction']:.1f}%)")
    print(f"  false-open {100 * g['false_open_rate']:.1f}%   false-closed {100 * g['false_closed_rate']:.1f}%")
    print("-" * 68)
    print(f"MAE horizon growth step0 -> step6 : {r['mae_growth_step0_to_last']:.2f}x")
    print(f"per-episode MAE range             : {r['per_episode_range']['min']:.4f} .. "
          f"{r['per_episode_range']['max']:.4f}")
    print(f"pred-vs-persistence corr          : mean {r['pred_vs_persistence_corr']['mean']:.3f}")
    print(f"written -> {Path(out_dir) / 'extra_metrics.json'}")


if __name__ == "__main__":
    main()
