#!/usr/bin/env python
"""Render the data-heavy part of the 8-task report from report_data.json.

Emits ``eval_openloop/report_data/report_facts.md``: one section per task with
the checkpoint table, state-hold per-step table, gripper stats and worst
episodes, plus the cross-task summary table. The narrative report
(``doc/reports/openloop_iclr_8tasks.md``) is assembled from these facts.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("/share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA")
DATA = ROOT / "eval_openloop" / "report_data"
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
RUNS = {
    "iclr_adjust_cup": "adjust_cup（扶正杯子）",
    "iclr_open_cabinet": "open_cabinet（开柜子）",
    "iclr_pick_banana_newtable": "pick_banana_newtable（抓香蕉放碗）",
    "iclr_pick_banana_pot": "pick_banana_pot（香蕉入罐）",
    "iclr_pick_block": "pick_block（抓双色积木）",
    "iclr_pick_eggplant_cluttered": "pick_eggplant_cluttered（杂乱中抓茄子）",
    "iclr_pick_eggplant_drawer": "pick_eggplant_drawer（茄子入抽屉）",
    "iclr_sponge_wipe": "sponge_wipe（海绵擦桌）",
}
DATASETS = {
    "iclr_adjust_cup": "adjust_cup_0409_1_offset_state_v2_1",
    "iclr_open_cabinet": "open_cabinet_all_0423_1_offset_state_v2_1",
    "iclr_pick_banana_newtable": "pick_banana_100_newTable_1_offset_state_v2_1",
    "iclr_pick_banana_pot": "pick_banana_pot_0730_1_offset_state_v2_1",
    "iclr_pick_block": "pick_block_100_1_offset_state_v2_1",
    "iclr_pick_eggplant_cluttered": "pick_eggplant_from_cluttered_0414_1_offset_state_v2_1",
    "iclr_pick_eggplant_drawer": "pick_eggplant_drawer_0730_1_offset_state_v2_1",
    "iclr_sponge_wipe": "sponge_wipe_0423_1_offset_state_v2_1",
}
DATASET_SIZE = {  # episodes / frames, from meta/info.json
    "iclr_adjust_cup": (50, 6837), "iclr_open_cabinet": (50, 6275),
    "iclr_pick_banana_newtable": (100, 12209), "iclr_pick_banana_pot": (48, 12224),
    "iclr_pick_block": (100, 18572), "iclr_pick_eggplant_cluttered": (87, 12884),
    "iclr_pick_eggplant_drawer": (50, 6809), "iclr_sponge_wipe": (49, 5619),
}


def pct(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100*v:.1f}%"


def f4(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def main():
    rows = json.load(open(DATA / "report_data.json"))
    for r in rows:  # json round-trips None for NaN via default=str
        for k, v in list(r.items()):
            if v == "nan":
                r[k] = float("nan")
    out = []
    w = out.append

    finals = {}
    for r in rows:
        run = r["run"]
        if run not in finals or r["step"] > finals[run]["step"]:
            finals[run] = r
    order = sorted(finals, key=lambda k: finals[k]["mae"])

    w("# 事实表（自动生成）\n")
    w("## 总表 · 最终权重\n")
    w("| # | 任务 | 数据集(ep/帧) | 窗口 | 训练mae_score | **开环MAE** | 差值 | vs零基线 | vs state-hold | 逐ep标准差 | 夹爪准确率 | step6/step0 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, k in enumerate(order, 1):
        r = finals[k]
        ep, fr = DATASET_SIZE[k]
        w(f"| {i} | {RUNS[k]} | {ep}/{fr} | {r['num_windows']} | "
          f"{f4(r.get('train_mae_at_step'))} | **{f4(r['mae'])}** | {f4(r.get('train_mae_gap'))} | "
          f"{pct(r['improvement_vs_zero'])} | {pct(r.get('improvement_vs_hold'))} | "
          f"{f4(r.get('episode_mae_std'))} | {pct(r.get('gripper_acc'))} | "
          f"{f4(r.get('horizon_growth'))}x |")
    w("")

    w("## 总表 · 全部 checkpoint\n")
    w("| 任务 | step | 开环MAE | 训练mae_score | R²(min~max) | vs state-hold | step0 model/hold | 夹爪准确率 |")
    w("|---|---|---|---|---|---|---|---|")
    for k in RUNS:
        for r in sorted([x for x in rows if x["run"] == k], key=lambda x: x["step"]):
            r2s = [v for v in r["r2_per_dim"] if not (isinstance(v, float) and np.isnan(v))]
            s0 = ""
            if "hold_per_step" in r and r["hold_per_step"]:
                s0 = f"{r['hold_per_step'][0]['model']:.4f} / {r['hold_per_step'][0]['hold']:.4f}"
            w(f"| {RUNS[k].split('（')[0]} | {r['step']} | {f4(r['mae'])} | "
              f"{f4(r.get('train_mae_at_step'))} | "
              f"{f4(min(r2s))}~{f4(max(r2s))} | {pct(r.get('improvement_vs_hold'))} | {s0} | "
              f"{pct(r.get('gripper_acc'))} |")
    w("")

    for k in order:
        r = finals[k]
        ep, fr = DATASET_SIZE[k]
        w(f"---\n\n## {RUNS[k]}\n")
        w(f"- 数据集：`{DATASETS[k]}`（{ep} episodes / {fr} 帧 / 10 fps）")
        w(f"- 权重：`checkpoints/{k}/checkpoints/steps_{r['step']}_pytorch_model.pt`"
          f"（= `final_model/pytorch_model.pt`）")
        w(f"- 测试窗口：{r['num_windows']}（覆盖全部 {r['num_episodes']} episodes）")
        w("")
        w("### 各 checkpoint\n")
        w("| step | 开环MAE | 训练mae_score | 差值 | vs零基线 | vs state-hold | 逐ep std |")
        w("|---|---|---|---|---|---|---|")
        for x in sorted([y for y in rows if y["run"] == k], key=lambda y: y["step"]):
            w(f"| {x['step']} | {f4(x['mae'])} | {f4(x.get('train_mae_at_step'))} | "
              f"{f4(x.get('train_mae_gap'))} | {pct(x['improvement_vs_zero'])} | "
              f"{pct(x.get('improvement_vs_hold'))} | {f4(x.get('episode_mae_std'))} |")
        w("")
        if "hold_per_step" in r:
            w("### 逐 step：模型 vs state-hold（最终权重）\n")
            w("| step | 模型 MAE | state-hold MAE | 提升 | 判定 |")
            w("|---|---|---|---|---|")
            for s in r["hold_per_step"]:
                imp = 1 - s["model"] / s["hold"]
                w(f"| {s['step']} | {s['model']:.4f} | {s['hold']:.4f} | {pct(imp)} | "
                  f"{'✅ 赢' if s['model_better'] else '❌ 输'} |")
            w("")
            if "hold_slices" in r:
                w("### 分窗口对比（最终权重）\n")
                w("| 窗口 | 模型 MAE | state-hold MAE | 提升 |")
                w("|---|---|---|---|")
                for name, v in r["hold_slices"].items():
                    w(f"| {name} | {v['model_mae']:.4f} | {v['hold_mae']:.4f} | "
                      f"{pct(v['improvement'])} |")
                w("")
        w("### 逐维（最终权重）\n")
        w("| 维度 | 归一化MAE | R² | raw MAE | raw MAE/量程 | vs state-hold |")
        w("|---|---|---|---|---|---|")
        for d, name in enumerate(DIM_NAMES):
            r2 = r["r2_per_dim"][d]
            hp = r.get("hold_per_dim", {}).get(name, {})
            w(f"| {name} | {r['mae_per_dim'][d]:.4f} | "
              f"{'nan' if isinstance(r2, float) and np.isnan(r2) else f'{r2:.3f}'} | "
              f"{r['raw_mae_per_dim'][d]:.4f} | {100*r['raw_mae_over_range'][d]:.1f}% | "
              f"{pct(hp.get('improvement'))} |")
        w("")
        w("### 夹爪 / 误差增长（最终权重）\n")
        w(f"- 开合判定准确率 **{pct(r.get('gripper_acc'))}**；GT 张开比例 "
          f"{pct(r.get('gripper_gt_open'))} vs 预测张开比例 {pct(r.get('gripper_pred_open'))}")
        w(f"- 误开(false-open) {pct(r.get('gripper_false_open'))}；"
          f"误闭(false-closed) {pct(r.get('gripper_false_closed'))}")
        w(f"- 误差增长 step0→step6：**{f4(r.get('horizon_growth'))}x**；"
          f"预测与 persistence 相关 {f4(r.get('pred_vs_persist_corr'))}")
        w(f"- 逐 episode MAE：{f4(r.get('episode_mae_min'))} ~ "
          f"{f4(r.get('episode_mae_max'))}（std {f4(r.get('episode_mae_std'))}）")
        worst = r.get("worst_episodes") or []
        if worst:
            w("- 最差 5 个 episode：")
            for we in worst:
                w(f"  - ep {we['episode']}: MAE {we['mae']:.4f}（xyz {we['mae_xyz']:.4f} / "
                  f"gripper {we['mae_gripper']:.4f}, n={we['num_windows']}）")
        w("")
        w(f"- 图：`eval_openloop/{k}/steps_{r['step']}/`（scatter/horizon/trajectories）"
          f"；密集轨迹图 `eval_openloop/{k}/steps_{r['step']}_dense/`")

    (DATA / "report_facts.md").write_text("\n".join(out) + "\n")
    print("written ->", DATA / "report_facts.md", f"({len(out)} lines)")


if __name__ == "__main__":
    main()
