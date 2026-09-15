"""
Parse a VLA-JEPA training log and summarise / plot the loss history.

The trainer only writes ``action_loss`` / ``wm_loss`` to the log file (tensorboard
receives just ``mae_score`` / ``mse_score``), so this script reads the log text.

Usage:
    python scripts/analyze_loss.py --log /tmp/adjust_cup_10k.log \
        --output_dir eval_openloop/adjust_cup_10k
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(r"Step (\d+), Loss: \{(.*?)\}")
KV_RE = re.compile(r"'([a-z_]+)': ([-0-9.eE+]+)")


def parse_log(path):
    raw = Path(path).read_text(errors="ignore")
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    raw = re.sub(r"train_starvla\.py:\d+", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    rows = []
    for step, body in STEP_RE.findall(raw):
        row = {"step": int(step)}
        row.update({k: float(v) for k, v in KV_RE.findall(body)})
        rows.append(row)
    return rows


def moving_average(y, w):
    """Return (x-index offsets, smoothed values) with the window centred."""
    y = np.asarray(y, dtype=float)
    w = max(1, min(int(w), len(y)))
    if w <= 1:
        return np.arange(len(y)), y
    sm = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.arange(len(y) - w + 1) + (w - 1) // 2, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"no step records found in {args.log}")

    steps = np.array([r["step"] for r in rows])
    action = np.array([r.get("action_loss", np.nan) for r in rows])
    wm = np.array([r.get("wm_loss", np.nan) for r in rows])
    mae = np.array([r.get("mae_score", np.nan) for r in rows])
    mse = np.array([r.get("mse_score", np.nan) for r in rows])
    lr = np.array([r.get("learning_rate", np.nan) for r in rows])

    def summarize(y, name, n_bins=10):
        valid = ~np.isnan(y)
        s, v = steps[valid], y[valid]
        if len(v) == 0:
            return {}
        edges = np.linspace(0, len(v), n_bins + 1).astype(int)
        bins = []
        for i in range(n_bins):
            seg = v[edges[i]:edges[i + 1]]
            if len(seg):
                bins.append({"step_range": [int(s[edges[i]]), int(s[edges[i + 1] - 1])],
                             "mean": float(seg.mean())})
        return {
            "first": float(v[0]), "last": float(v[-1]), "min": float(v.min()),
            "mean_first_10pct": float(v[: max(1, len(v) // 10)].mean()),
            "mean_last_10pct": float(v[-max(1, len(v) // 10):].mean()),
            "bins": bins,
        }

    summary = {k: summarize(v, k) for k, v in
               [("action_loss", action), ("wm_loss", wm), ("mae_score", mae), ("mse_score", mse)]}
    summary["num_logged_steps"] = int(len(rows))
    summary["step_range"] = [int(steps.min()), int(steps.max())]
    with open(out_dir / "loss_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for ax, series, name in [(axes[0], action, "action_loss"), (axes[1], wm, "wm_loss")]:
        ax.plot(steps, series, lw=0.6, alpha=0.35, label=name)
        idx, sm = moving_average(series, args.window)
        ax.plot(steps[idx], sm, lw=2, label=f"moving avg ({min(args.window, len(series))})")
        ax.set_ylabel(name)
        ax.set_yscale("log")
    axes[2].plot(steps, lr, lw=1.5, color="tab:green", label="learning rate")
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("training step")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    ev = ~np.isnan(mae)
    if ev.any():
        ax2 = axes[2].twinx()
        ax2.plot(steps[ev], mae[ev], "o-", color="tab:red", ms=3, label="mae_score (eval)")
        ax2.set_ylabel("mae_score")
        ax2.legend(loc="upper right")
    fig.suptitle("VLA-JEPA fine-tuning on adjust_cup (10k steps)")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=120)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"\nplot -> {out_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
