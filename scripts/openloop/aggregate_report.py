#!/usr/bin/env python
"""Aggregate all 8-task ICLR open-loop runs into tables + comparison figures.

Reads ``eval_openloop/<run>/steps_<N>/{metrics,extra_metrics}.json`` and the
training-time tensorboard curves (``report_data/train_curves.json``), then writes

  eval_openloop/report_data/summary_table.csv     one row per (run, step)
  eval_openloop/report_data/report_data.json      nested, everything a report needs
  doc/reports/figures/iclr_8tasks/*.png           comparison figures
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA")
EVAL = ROOT / "eval_openloop"
DATA = EVAL / "report_data"
FIGS = ROOT / "doc" / "reports" / "figures" / "iclr_8tasks"
# small machine-readable artifacts are committed next to the report so its numbers
# can be checked without regenerating the (gitignored) eval_openloop/ tree
PUBLISH = ROOT / "doc" / "reports" / "data" / "iclr_8tasks"
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# Chinese labels need a font that has ASCII *and* CJK glyphs; matplotlib only does
# per-glyph fallback when the list is assigned to font.family (see doc §8.5).
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["axes.unicode_minus"] = False

RUNS = {
    "iclr_adjust_cup": "adjust_cup（扶正杯子）",
    "iclr_open_cabinet": "open_cabinet（开柜子）",
    "iclr_pick_banana_newtable": "pick_banana_newtable（抓香蕉→碗）",
    "iclr_pick_banana_pot": "pick_banana_pot（香蕉入罐）",
    "iclr_pick_block": "pick_block（抓双色积木）",
    "iclr_pick_eggplant_cluttered": "pick_eggplant_cluttered（杂乱中抓茄子）",
    "iclr_pick_eggplant_drawer": "pick_eggplant_drawer（茄子入抽屉）",
    "iclr_sponge_wipe": "sponge_wipe（海绵擦桌）",
}
SHORT = {k: v.split("（")[0] for k, v in RUNS.items()}


def load_row(run, step):
    d = EVAL / run / f"steps_{step}"
    mp, ep = d / "metrics.json", d / "extra_metrics.json"
    if not mp.exists():
        return None
    m = json.load(open(mp))
    row = {
        "run": run,
        "step": step,
        "num_windows": m["num_windows"],
        "num_episodes": m["config"]["num_episodes"],
        "mae": m["normalized"]["mae"],
        "mse": m["normalized"]["mse"],
        "zero_baseline_mae": m["normalized_baseline_predict_zero"]["mae"],
        "improvement_vs_zero": m["normalized"]["mae_improvement_over_zero_baseline"],
        "mae_per_dim": m["normalized"]["mae_per_dim"],
        "mae_gripper": m["normalized"]["mae_per_dim"][6],
        "mae_xyz": float(np.mean(m["normalized"]["mae_per_dim"][:3])),
        "r2_per_dim": m["normalized"]["r2_per_dim"],
        "mae_per_horizon_step": m["normalized"]["mae_per_horizon_step"],
        "episode_mae_std": m.get("episode_mae_std"),
        "episode_mae_min": min(v["mae"] for v in m["per_episode"].values()),
        "episode_mae_max": max(v["mae"] for v in m["per_episode"].values()),
        "worst_episodes": m.get("worst_episodes", []),
        "raw_mae_per_dim": m["raw"]["mae_per_dim"],
        "raw_mae_over_range": m["raw"]["mae_per_dim_over_gt_range"],
        "has_extra": ep.exists(),
    }
    if ep.exists():
        e = json.load(open(ep))
        sh = e.get("state_hold_baseline", {})
        row["persist_mae"] = e["full_chunk"]["persistence_baseline_mae"]
        row["improvement_vs_persist"] = e["full_chunk"]["model_vs_persistence_improvement"]
        row["gripper_acc"] = e["gripper_decision"]["accuracy"]
        row["gripper_gt_open"] = e["gripper_decision"]["gt_open_fraction"]
        row["gripper_pred_open"] = e["gripper_decision"]["pred_open_fraction"]
        row["gripper_false_open"] = e["gripper_decision"]["false_open_rate"]
        row["gripper_false_closed"] = e["gripper_decision"]["false_closed_rate"]
        row["horizon_growth"] = e["mae_growth_step0_to_last"]
        row["pred_vs_persist_corr"] = e["pred_vs_persistence_corr"]["mean"]
        row["prefix"] = {k: v for k, v in e["by_prefix"].items()}
        if "error" not in sh and sh:
            row["hold_mae"] = sh["hold_mae"]
            row["improvement_vs_hold"] = sh["improvement"]
            row["hold_per_step"] = [
                {"step": s["step"], "model": s["model_mae"], "hold": s["hold_mae"],
                 "model_better": s["model_better"]} for s in sh["per_step"]]
            row["hold_slices"] = sh["window_slices"]
            row["hold_per_dim"] = sh["per_dim"]
        else:
            row["hold_error"] = sh.get("error", "missing")
    return row


def collect():
    rows = []
    for run in RUNS:
        d = EVAL / run
        if not d.exists():
            continue
        # skip the *_dense dirs: they are 4-episode per-frame sweeps used only for
        # trajectory plots, and would otherwise duplicate rows in the summary
        steps = sorted(int(p.name.split("_")[1]) for p in d.glob("steps_*")
                       if (p / "metrics.json").exists()
                       and not p.name.endswith("_dense"))
        for s in steps:
            r = load_row(run, s)
            if r:
                rows.append(r)
    return rows


def attach_train_mae(rows):
    tc = json.load(open(DATA / "train_curves.json"))
    for r in rows:
        c = tc.get(r["run"], {}).get("mae_score", {})
        # tensorboard logs at eval_interval=500 -> take the closest logged step
        if c:
            key = min(c, key=lambda k: abs(int(k) - r["step"]))
            r["train_mae_at_step"] = c[key]
            r["train_mae_step"] = int(key)
            r["train_mae_gap"] = r["mae"] - c[key]
            r["train_curve"] = c
    return rows


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_final_bar(rows, out):
    final = {r["run"]: r for r in rows if r["step"] == max(
        x["step"] for x in rows if x["run"] == r["run"])}
    order = sorted(final, key=lambda k: final[k]["mae"])
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    ax = axes[0]
    ax.bar(x, [final[k]["mae"] for k in order], color="#3b6ea5", label="开环 MAE")
    ax.plot(x, [final[k].get("train_mae_at_step", np.nan) for k in order], "rD--",
            ms=6, label="训练期 mae_score")
    for i, k in enumerate(order):
        ax.text(i, final[k]["mae"], f"{final[k]['mae']:.4f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("normalized MAE (chunk 7 步)")
    ax.set_title("各任务最终权重：开环 MAE vs 训练期 mae_score")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    vals = [100 * final[k].get("improvement_vs_hold", np.nan) for k in order]
    cols = ["#2e7d32" if v >= 20 else ("#f9a825" if v > 0 else "#c62828") for v in vals]
    ax.bar(x, vals, color=cols)
    ax.axhline(20, color="g", ls="--", lw=1)
    ax.axhline(0, color="k", lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("整段 chunk 相对 state-hold 提升 (%)")
    ax.set_title("是否跑赢「保持不动」基线（绿≥20%，黄>0，红<0）")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    w = 0.38
    ax.bar(x - w / 2, [100 * final[k]["gripper_acc"] for k in order], w,
           label="夹爪开合判定准确率", color="#6a4c93")
    ax.bar(x + w / 2, [100 * final[k].get("improvement_vs_hold", np.nan) for k in order],
           w, label="vs state-hold 提升", color="#8ecae6")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in order], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("夹爪判定准确率 vs 整体提升")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig1_final_summary.png", dpi=130)
    plt.close(fig)


def fig_learning_curves(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    for run in RUNS:
        rs = sorted([r for r in rows if r["run"] == run], key=lambda r: r["step"])
        if not rs:
            continue
        ax.plot([r["step"] for r in rs], [r["mae"] for r in rs], "o-",
                label=SHORT[run], ms=5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("开环 normalized MAE")
    ax.set_title("开环 MAE 随训练步数（8 任务）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    for run in RUNS:
        rs = sorted([r for r in rows if r["run"] == run], key=lambda r: r["step"])
        if not rs or "train_curve" not in rs[0]:
            continue
        c = rs[0]["train_curve"]
        ks = sorted(int(k) for k in c)
        ax.plot(ks, [c[str(k)] if str(k) in c else c[k] for k in ks], "-", lw=1.2,
                label=SHORT[run])
    ax.set_xlabel("训练步数")
    ax.set_ylabel("训练期 mae_score")
    ax.set_title("训练日志 mae_score（eval_interval=500）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[2]
    fin = [r for r in rows if r["step"] == max(
        x["step"] for x in rows if x["run"] == r["run"])]
    ax.scatter([r.get("train_mae_at_step", np.nan) for r in fin],
               [r["mae"] for r in fin], s=45, c="#c62828")
    lo = min([r["mae"] for r in fin] + [r.get("train_mae_at_step", 1) for r in fin]) * 0.85
    hi = max([r["mae"] for r in fin] + [r.get("train_mae_at_step", 0) for r in fin]) * 1.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    for r in fin:
        ax.annotate(SHORT[r["run"]], (r.get("train_mae_at_step", np.nan), r["mae"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("训练期 mae_score（对应 step）")
    ax.set_ylabel("本次开环 MAE")
    ax.set_title("交叉验证：开环 vs 训练日志（虚线为 y=x）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_learning_curves.png", dpi=130)
    plt.close(fig)


def fig_horizon_and_hold(rows, out):
    fin = {}
    for r in rows:
        run = r["run"]
        if run not in fin or r["step"] > fin[run]["step"]:
            fin[run] = r
    order = sorted(fin, key=lambda k: fin[k]["mae"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    ax = axes[0]
    for k in order:
        r = fin[k]
        ax.plot(range(7), r["mae_per_horizon_step"], "o-", ms=4,
                label=f"{SHORT[k]}（{r['mae']:.4f}）")
    ax.set_xlabel("chunk 内位置 (step 0..6)")
    ax.set_ylabel("normalized MAE")
    ax.set_title("误差随预测步长增长")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    w = 0.8 / len(order)
    x = np.arange(7)
    for i, k in enumerate(order):
        r = fin[k]
        if "hold_per_step" not in r:
            continue
        diff = [100 * (1 - s["model"] / s["hold"]) for s in r["hold_per_step"]]
        ax.bar(x + i * w, diff, w, label=SHORT[k])
    ax.axhline(0, color="k", lw=1)
    ax.axhline(20, color="g", ls="--", lw=1)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([f"step {i}" for i in range(7)])
    ax.set_ylabel("相对 state-hold 提升 (%)")
    ax.set_title("逐 step：模型 vs 保持不动（负值=不如不动）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig3_horizon_and_hold.png", dpi=130)
    plt.close(fig)


def write_tables(rows, out):
    cols = ["run", "step", "num_windows", "num_episodes", "mae", "mse",
            "improvement_vs_zero", "improvement_vs_persist", "improvement_vs_hold",
            "hold_mae", "r2_min", "r2_mean", "gripper_acc", "mae_gripper",
            "mae_xyz", "horizon_growth", "episode_mae_std", "train_mae_at_step",
            "train_mae_gap", "pred_vs_persist_corr"]
    with open(out / "summary_table.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["run"], r["step"])):
            r2 = [v for v in r["r2_per_dim"] if not np.isnan(v)]
            wr.writerow([
                r["run"], r["step"], r["num_windows"], r["num_episodes"],
                round(r["mae"], 5), round(r["mse"], 5),
                round(r["improvement_vs_zero"], 4),
                round(r.get("improvement_vs_persist", float("nan")), 4),
                round(r.get("improvement_vs_hold", float("nan")), 4),
                round(r.get("hold_mae", float("nan")), 5),
                round(min(r2), 4) if r2 else "", round(float(np.mean(r2)), 4) if r2 else "",
                round(r.get("gripper_acc", float("nan")), 4),
                round(r.get("mae_gripper", float("nan")), 5),
                round(r.get("mae_xyz", float("nan")), 5),
                round(r.get("horizon_growth", float("nan")), 3),
                round(r.get("episode_mae_std", float("nan")), 5),
                round(r.get("train_mae_at_step", float("nan")), 5),
                round(r.get("train_mae_gap", float("nan")), 5),
                round(r.get("pred_vs_persist_corr", float("nan")), 4),
            ])
    json.dump(rows, open(out / "report_data.json", "w"), indent=2, default=str)


def publish():
    PUBLISH.mkdir(parents=True, exist_ok=True)
    for name in ("summary_table.csv", "train_curves.json", "sampling_noise.json",
                 "sampler_options_banana_pot.json", "latency.json"):
        src = DATA / name
        if src.exists():
            shutil.copy(src, PUBLISH / name)
    print("published ->", PUBLISH)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = attach_train_mae(collect())
    print(f"collected {len(rows)} evaluated checkpoints")
    write_tables(rows, DATA)
    fig_final_bar(rows, FIGS)
    fig_learning_curves(rows, FIGS)
    fig_horizon_and_hold(rows, FIGS)
    print("figures ->", FIGS)
    print("tables  ->", DATA / "summary_table.csv")
    publish()
    missing = [r["run"] for r in rows if not r["has_extra"]]
    if missing:
        print("!! extra_metrics.json missing for:", sorted(set(missing)))


if __name__ == "__main__":
    main()
