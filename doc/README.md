# VLA-JEPA 文档索引

真机（Piper）数据从**采集格式转换** → **训练** → **开环测试**的完整流程文档。

---

## 三份主文档

按流水线顺序阅读：

| # | 文档 | 什么时候看 |
|---|---|---|
| 1 | [**01_data_conversion.md**](./01_data_conversion.md) · 训练前数据转换 | 拿到新采集的 v3.0 数据，要转成 v2.1 并注册机器人类型 |
| 2 | [**02_training.md**](./02_training.md) · 训练启动命令说明 | 数据就绪，要开始训练 / 续训 / 排查训练报错 |
| 3 | [**03_openloop_testing.md**](./03_openloop_testing.md) · 开环测试 | 训练完成，要评估权重效果 |

每份文档都包含**可直接复制运行的完整命令**。

---

## 全流程速览

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# ── 步骤 1：数据转换（v3.0 → v2.1）───────────────────────
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /path/to/xxx_1_offset_state \
  --output /path/to/xxx_1_offset_state_v2_1 \
  --fps 10 --chunk-size 100

# ── 步骤 2：训练 ────────────────────────────────────────
# 2a. 先冒烟（1 GPU，2 分钟，验证数据+模型+保存全链路）
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml

# 2b. 正式训练（4 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml

# ── 步骤 3：开环测试 ────────────────────────────────────
# 3a. 主测试（指标）
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/<run_id>/config.yaml \
  --checkpoint  checkpoints/<run_id>/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/<run_id> \
  --windows_per_episode 8 --batch_size 4 --seed 0

# 3b. 基线分析（必做，否则会被虚高的指标误导）
/opt/conda/envs/VLA_JEPA/bin/python scripts/analyze_openloop.py \
  --predictions eval_openloop/<run_id>/predictions.npz

# 3c. 轨迹可视化（可选）
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/<run_id>/config.yaml \
  --checkpoint  checkpoints/<run_id>/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/<run_id>_dense \
  --dense_stride 1 --num_episodes 4 --dense_oversample 4 --batch_size 4 --seed 0

/opt/conda/envs/VLA_JEPA/bin/python scripts/plot_openloop_trajectory.py \
  --predictions eval_openloop/<run_id>_dense/predictions.npz \
  --episodes 0 1 2 --zoom_episode 0
```

---

## 目录结构

```
doc/
├── README.md                    # 本文件
├── 01_data_conversion.md        # ① 训练前：数据转换 + 机器人注册
├── 02_training.md               # ② 训练：启动命令 + 配置 + 排错
├── 03_openloop_testing.md       # ③ 开环测试：流程 + 指标解读 + 坑位
│
├── reports/                     # 实测报告（用 03 的流程跑出的结果）
│   ├── openloop_adjust_cup_5k.md
│   └── openloop_pick_open_place_0724.md
│
└── archive/                     # 已被上面三份取代的原始文档（保留备查）
    ├── debug_journal.md         # Piper 适配调试全记录（所有原始报错）
    ├── training_pipeline.md     # 训练流程与架构完整详解
    ├── training_phase2.md       # 第二阶段训练完整教程（逐项配置注释 + FAQ）
    └── adjust_cup_real_world.md # adjust_cup 真机数据接入与冒烟测试报告
```

---

## 关键脚本

| 脚本 | 用途 | 详见 |
|---|---|---|
| `scripts/convert_v3_to_v2_1_aligned.py` | v3.0 → v2.1 数据转换（**推荐**） | [01](./01_data_conversion.md#3-转换命令) |
| `scripts/convert_v3_to_v2_1.py` | 旧版转换脚本（有 bug，勿用于新数据） | [01](./01_data_conversion.md#31-用哪个脚本) |
| `starVLA/training/train_starvla.py` | 训练入口（阶段二） | [02](./02_training.md#5-启动命令) |
| `scripts/vlajepa_robot_ft.sh` | 8 卡训练启动脚本 | [02](./02_training.md#53-多卡正式训练推荐) |
| `scripts/eval_openloop.py` | 开环测试主脚本 | [03](./03_openloop_testing.md#3-eval_openlooppy-参数) |
| `scripts/analyze_openloop.py` | 开环测试补充基线分析 | [03](./03_openloop_testing.md#4-analyze_openlooppy-参数) |
| `scripts/plot_openloop_trajectory.py` | 轨迹拟合可视化 | [03](./03_openloop_testing.md#5-plot_openloop_trajectorypy-参数) |
| `scripts/analyze_loss.py` | 从训练日志提取 loss 曲线 | [02](./02_training.md#71-训练日志) |

---

## 三条最要紧的经验

1. **数据必须是 LeRobot v2.1**。v3.0 要先转换，否则加载器直接报错。
   （[01 §1](./01_data_conversion.md#1-什么时候需要转换)）

2. **训练必须用 `accelerate launch`**，即使只有 1 张卡。
   直接 `python train_starvla.py` 会在分布式初始化上失败。
   单卡 24 GB 还需要 ZeRO-2 + CPU Offload 和 `pip install ninja`。
   （[02 §8.1](./02_training.md#81-valueerror-default-process-group-has-not-been-initialized) / [§8.2](./02_training.md#82-cuda-out-of-memory)）

3. **看开环指标别只看「相比全预测 0 提升多少」**——那个基线太弱，会严重高估模型。
   要看 `analyze_openloop.py` 算出的 **state-hold 基线**（"什么都不做"）。
   （[03 §7.1](./03_openloop_testing.md#71-先看跑赢了哪个基线)）

---

## 相关文档（仓库其它位置）

| 位置 | 说明 |
|---|---|
| `README.md`（仓库根目录） | 项目总览、环境安装、官方评测（LIBERO / SimplerEnv） |
| `usage.md`（仓库根目录） | v3 → v2.1 转换命令 ⚠️ 内容已并入 [01](./01_data_conversion.md) |
| `启动脚本.md`（仓库根目录） | 训练启动与参数说明 ⚠️ 内容已并入 [02](./02_training.md) |
| `checkpoints/<run_id>/config.yaml` | 每个权重的训练配置快照（`data_root_dir` 说明它用的是哪个数据集） |
