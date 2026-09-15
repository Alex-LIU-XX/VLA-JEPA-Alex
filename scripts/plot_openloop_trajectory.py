"""
Trajectory-fitting visualisations for an open-loop run.

Consumes the ``predictions.npz`` written by ``scripts/eval_openloop.py`` (run it
with ``--dense_stride`` so the windows sweep the episode nearly frame by frame).

Figures written to --out_dir:

  trajectory_fit.png      ground truth vs predicted action for every joint, one
                          column per episode. Each window's whole 7-step chunk is
                          drawn as a faint ribbon, plus the step-0 prediction as a
                          dashed curve -- so you can see both the fit and how the
                          model extrapolates forward.
  trajectory_zoom.png     a few individual chunks in detail: the predicted 7-step
                          chunk against the ground-truth chunk it was compared to.
  trajectory_2d.png       2-D phase plots of selected joint pairs, coloured by time.
  error_over_time.png     per-joint |error| against frame index, to expose *where*
                          in an episode the policy breaks down.
  gripper_timeline.png    gripper channel only -- the most consequential channel
                          for pick-and-place.

Usage:
    python scripts/plot_openloop_trajectory.py \
        --predictions eval_openloop/adjust_cup_5k_dense/predictions.npz \
        --out_dir      eval_openloop/adjust_cup_5k_dense
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DIM_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
DIM_NAMES_CN = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]
PAIRS = [(1, 2), (3, 4), (0, 5)]  # (joint2,joint3), (joint4,joint5), (joint1,joint6)


def setup_fonts():
    """Build a font stack that renders both CJK and ASCII.

    Some CJK fonts (notably ``Droid Sans Fallback``) ship without ASCII glyphs,
    so putting one first turns every axis number into a tofu box. Matplotlib
    falls back per glyph, but only when the list is assigned to ``font.family``
    -- assigning ``font.sans-serif`` resolves to a single font and drops the
    fallback chain.
    """
    available = {f.name for f in fm.fontManager.ttflist}
    stack = ["DejaVu Sans"]
    for name in ("Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                 "Microsoft YaHei", "SimHei", "Droid Sans Fallback"):
        if name in available:
            stack.append(name)
            break
    plt.rcParams["font.family"] = stack
    plt.rcParams["axes.unicode_minus"] = False
    return len(stack) > 1


def denormalize(x, lo, hi):
    return (np.asarray(x, dtype=np.float64) + 1.0) / 2.0 * (hi - lo) + lo


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    lo, hi = d["action_min"], d["action_max"]
    pred = denormalize(d["pred"], lo, hi)
    gt = denormalize(d["gt"], lo, hi)
    meta = d["meta"]
    traj = np.array([int(m[0]) for m in meta])
    step = np.array([int(m[1]) for m in meta])
    return pred, gt, traj, step, lo, hi


def _episode_axes(traj):
    return sorted(set(traj.tolist()))


# --------------------------------------------------------------------------- #
def plot_fit(pred, gt, traj, step, episodes, out_path, cn):
    names = DIM_NAMES_CN if cn else DIM_NAMES
    n_ep = len(episodes)
    fig, axes = plt.subplots(7, n_ep, figsize=(5.2 * n_ep, 2.05 * 7),
                             sharex="col", squeeze=False)
    for col, ep in enumerate(episodes):
        sel = np.where(traj == ep)[0]
        order = np.argsort(step[sel])
        sel = sel[order]
        t = step[sel]
        for d in range(7):
            ax = axes[d][col]
            # every predicted chunk as a faint ribbon. Keep alpha low: with ~137
            # overlapping ribbons anything higher saturates into a solid slab that
            # hides the curves drawn on top of it.
            for i in sel:
                ax.plot(step[i] + np.arange(pred.shape[1]), pred[i, :, d],
                        color="tab:blue", alpha=0.07, lw=0.9, zorder=2)
            ax.plot(t, gt[sel, 0, d], color="black", lw=2.0, zorder=4,
                    label="真值 (GT)" if (d == 0 and col == 0) else None)
            ax.plot(t, pred[sel, 0, d], color="tab:red", lw=1.5, ls="--", zorder=5,
                    label="预测 (chunk step 0)" if (d == 0 and col == 0) else None)
            if d == 0:
                ax.set_title(f"episode {ep}", fontsize=11)
            if col == 0:
                ax.set_ylabel(names[d], fontsize=9)
            if d == 6:
                ax.set_xlabel("帧序号 (frame index)")
            ax.grid(alpha=0.3)
    handles = [Line2D([], [], color="black", lw=1.7, label="真值 (GT)" if cn else "ground truth"),
               Line2D([], [], color="tab:red", lw=1.3, ls="--",
                      label="预测 step 0" if cn else "predicted (chunk step 0)"),
               Line2D([], [], color="tab:blue", lw=1.0, alpha=0.35,
                      label="预测完整 chunk（7 步）" if cn else "predicted full chunk (7 steps)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1.003))
    fig.suptitle("开环轨迹拟合：真值 vs 预测（每个窗口预测未来 7 步）" if cn
                 else "Open-loop trajectory fit: ground truth vs prediction "
                      "(each window predicts 7 steps ahead)",
                 fontsize=13, y=1.012)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def plot_zoom(pred, gt, traj, step, episode, out_path, n_chunks=3, cn=True):
    names = DIM_NAMES_CN if cn else DIM_NAMES
    sel = np.where(traj == episode)[0]
    order = np.argsort(step[sel])
    sel = sel[order]
    # spread the zoomed windows over the episode
    picks = np.linspace(0, len(sel) - 1, n_chunks).round().astype(int)
    fig, axes = plt.subplots(7, n_chunks, figsize=(4.6 * n_chunks, 2.05 * 7), squeeze=False)
    k = np.arange(pred.shape[1])
    for col, p in enumerate(picks):
        i = sel[p]
        for d in range(7):
            ax = axes[d][col]
            ax.plot(k, gt[i, :, d], "o-", color="black", lw=1.9, ms=5,
                    label="真值" if (d == 0 and col == 0) else None)
            ax.plot(k, pred[i, :, d], "s--", color="tab:red", lw=1.9, ms=5,
                    label="预测" if (d == 0 and col == 0) else None)
            ax.fill_between(k, gt[i, :, d], pred[i, :, d], color="tab:red", alpha=0.12)
            e = np.abs(pred[i, :, d] - gt[i, :, d]).mean()
            if d == 0:
                ax.set_title(f"帧 {step[i]}  (MAE={e:.3f})", fontsize=10)
            if col == 0:
                ax.set_ylabel(names[d], fontsize=9)
            if d == 6:
                ax.set_xlabel("chunk 内位置 (0→6)")
            ax.grid(alpha=0.3)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.004))
    fig.suptitle(f"单个动作块放大对比（episode {episode}）" if cn else
                 f"Individual chunk zoom (episode {episode})", fontsize=13, y=1.012)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def plot_2d(pred, gt, traj, step, episodes, out_path, cn=True):
    n_ep = len(episodes)
    fig, axes = plt.subplots(len(PAIRS), n_ep, figsize=(5.0 * n_ep, 4.6 * len(PAIRS)),
                             squeeze=False)
    for col, ep in enumerate(episodes):
        sel = np.where(traj == ep)[0]
        sel = sel[np.argsort(step[sel])]
        tt = step[sel]
        for row, (a, b) in enumerate(PAIRS):
            ax = axes[row][col]
            sc = ax.scatter(gt[sel, 0, a], gt[sel, 0, b], c=tt, cmap="viridis",
                            s=13, zorder=3, label="真值" if cn else "GT")
            ax.plot(pred[sel, 0, a], pred[sel, 0, b], color="tab:red", lw=1.2,
                    alpha=0.85, zorder=4, label="预测" if cn else "pred")
            ax.set_xlabel(f"{DIM_NAMES[a]} (rad)")
            ax.set_ylabel(f"{DIM_NAMES[b]} (rad)")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"episode {ep}", fontsize=11)
            if row == 0 and col == 0:
                ax.legend(fontsize=9, loc="best")
    fig.suptitle("关节空间 2D 相图（颜色 = 时间推进，深→浅）" if cn else
                 "2-D phase plots (colour = time)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def plot_error_over_time(pred, gt, traj, step, episodes, out_path, cn=True):
    names = DIM_NAMES_CN if cn else DIM_NAMES
    n_ep = len(episodes)
    fig, axes = plt.subplots(7, n_ep, figsize=(5.2 * n_ep, 1.85 * 7),
                             sharex="col", squeeze=False)
    for col, ep in enumerate(episodes):
        sel = np.where(traj == ep)[0]
        sel = sel[np.argsort(step[sel])]
        t = step[sel]
        err = np.abs(pred[sel, 0] - gt[sel, 0])  # (n, 7)
        for d in range(7):
            ax = axes[d][col]
            ax.plot(t, err[:, d], color="tab:purple", lw=1.2)
            ax.axhline(err[:, d].mean(), color="gray", ls=":", lw=1.2)
            if d == 0:
                ax.set_title(f"episode {ep}", fontsize=11)
            if col == 0:
                ax.set_ylabel(names[d], fontsize=9)
            if d == 6:
                ax.set_xlabel("帧序号 (frame index)")
            ax.grid(alpha=0.3)
    fig.suptitle("逐关节绝对误差随时间的分布（虚线 = 该 episode 均值）" if cn else
                 "Per-joint absolute error over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def plot_gripper(pred, gt, traj, step, episodes, out_path, cn=True):
    n_ep = len(episodes)
    fig, axes = plt.subplots(n_ep, 1, figsize=(11, 2.5 * n_ep), squeeze=False)
    for r, ep in enumerate(episodes):
        ax = axes[r][0]
        sel = np.where(traj == ep)[0]
        sel = sel[np.argsort(step[sel])]
        t = step[sel]
        ax.plot(t, gt[sel, 0, 6], "o-", color="black", ms=3.5, lw=1.5,
                label="真值夹爪" if cn else "GT gripper")
        ax.plot(t, pred[sel, 0, 6], "x--", color="tab:red", ms=3.5, lw=1.5,
                label="预测夹爪" if cn else "pred gripper")
        acc = ((pred[sel, 0, 6] > 0) == (gt[sel, 0, 6] > 0)).mean()
        ax.set_title(f"episode {ep}  开合判定准确率 {100 * acc:.1f}%" if cn else
                     f"episode {ep}  open/close accuracy {100 * acc:.1f}%", fontsize=10)
        ax.set_ylabel("夹爪 (归一化)" if cn else "gripper (norm.)")
        ax.grid(alpha=0.3)
        if r == 0:
            ax.legend(fontsize=9, ncol=2)
    axes[-1][0].set_xlabel("帧序号 (frame index)")
    fig.suptitle("夹爪通道时序对比" if cn else "Gripper channel over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="episode ids to draw; defaults to the first 3 in the npz")
    ap.add_argument("--zoom_episode", type=int, default=None,
                    help="episode used for the chunk-zoom figure")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or Path(args.predictions).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    cn = setup_fonts()
    print(f"CJK font available: {cn}")

    pred, gt, traj, step, lo, hi = load(args.predictions)
    all_eps = _episode_axes(traj)
    episodes = args.episodes if args.episodes else all_eps[:3]
    episodes = [e for e in episodes if e in all_eps]
    if not episodes:
        raise SystemExit(f"none of the requested episodes are in the file; have {all_eps}")
    print(f"episodes in npz : {all_eps}")
    print(f"drawing         : {episodes}")
    print(f"windows         : {len(traj)}  chunk={pred.shape[1]}  dim={pred.shape[2]}")

    plot_fit(pred, gt, traj, step, episodes, out_dir / "trajectory_fit.png", cn)
    zoom_ep = args.zoom_episode if args.zoom_episode is not None else episodes[0]
    plot_zoom(pred, gt, traj, step, zoom_ep, out_dir / "trajectory_zoom.png", cn=cn)
    plot_2d(pred, gt, traj, step, episodes, out_dir / "trajectory_2d.png", cn)
    plot_error_over_time(pred, gt, traj, step, episodes, out_dir / "error_over_time.png", cn)
    plot_gripper(pred, gt, traj, step, episodes, out_dir / "gripper_timeline.png", cn)

    # quick numeric summary over the drawn episodes
    summary = {}
    for ep in episodes:
        sel = traj == ep
        summary[str(ep)] = {
            "num_windows": int(sel.sum()),
            "mae_step0": float(np.abs(pred[sel, 0] - gt[sel, 0]).mean()),
            "mae_step0_per_dim": np.abs(pred[sel, 0] - gt[sel, 0]).mean(axis=0).tolist(),
            "gripper_accuracy": float(((pred[sel, 0, 6] > 0) == (gt[sel, 0, 6] > 0)).mean()),
        }
    with open(out_dir / "trajectory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 60)
    for ep, v in summary.items():
        print(f"ep {ep:>3}: n={v['num_windows']:>4}  MAE(step0)={v['mae_step0']:.4f}"
              f"  gripper acc={100 * v['gripper_accuracy']:.1f}%")
    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
