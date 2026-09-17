#!/usr/bin/env python
"""Build a 8-panel contact sheet from the dense trajectory figures of the 8 tasks.

Each task's ``steps_<N>_dense/trajectory_fit.png`` already carries its own axes;
here they are placed on one page for the cross-task comparison section of the
report, alongside a second sheet with the 2-D phase plots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path("/share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA")
FIGS = ROOT / "doc" / "reports" / "figures" / "iclr_8tasks"
RUNS = [
    ("iclr_adjust_cup", "adjust_cup 扶正杯子"),
    ("iclr_open_cabinet", "open_cabinet 开柜子"),
    ("iclr_pick_banana_newtable", "pick_banana_newtable 抓香蕉放碗"),
    ("iclr_pick_banana_pot", "pick_banana_pot 香蕉入罐"),
    ("iclr_pick_block", "pick_block 抓双色积木"),
    ("iclr_pick_eggplant_cluttered", "pick_eggplant_cluttered 杂乱中抓茄子"),
    ("iclr_pick_eggplant_drawer", "pick_eggplant_drawer 茄子入抽屉"),
    ("iclr_sponge_wipe", "sponge_wipe 海绵擦桌"),
]
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]


def final_step(run):
    return max(int(p.name.split("_")[1])
               for p in (ROOT / "checkpoints" / run / "checkpoints").glob(
                   "steps_*_pytorch_model.pt"))


def sheet(fname, title, out_name):
    fig, axes = plt.subplots(4, 2, figsize=(20, 26))
    for ax, (run, label) in zip(axes.ravel(), RUNS):
        img = ROOT / "eval_openloop" / run / f"steps_{final_step(run)}_dense" / fname
        if img.exists():
            ax.imshow(mpimg.imread(img))
            ax.set_title(f"{label}  |  steps_{final_step(run)}", fontsize=15)
        else:
            ax.text(0.5, 0.5, f"{label}\n(missing {fname})", ha="center", va="center",
                    fontsize=14)
        ax.axis("off")
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FIGS / out_name, dpi=90)
    plt.close(fig)
    print("written ->", FIGS / out_name)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    sheet("trajectory_fit.png",
          "8 个 ICLR 真机任务 · 开环轨迹拟合（密集逐帧扫描，ep0-2）",
          "fig4_trajectory_contact.png")
    sheet("trajectory_2d.png",
          "8 个 ICLR 真机任务 · 关节空间 2D 相图（预测 vs 真值）",
          "fig5_phase_contact.png")
