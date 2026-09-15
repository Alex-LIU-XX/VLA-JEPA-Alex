# 03 · 训练完成后的开环测试

本文档说明如何对一个训练好的 VLA-JEPA 权重做**开环测试**（open-loop evaluation）：
用数据集里已有的观测帧喂给模型，让它预测未来 7 步动作，再与真值比对。
不接真机、不起仿真，纯离线评估策略拟合质量。

**在流水线中的位置**：
[`01_data_conversion.md`](./01_data_conversion.md) → [`02_training.md`](./02_training.md) → **本文档**

**实测报告**（用本文档的流程跑出来的结果）：

| 报告 | 内容 |
|---|---|
| [`reports/openloop_adjust_cup_5k.md`](./reports/openloop_adjust_cup_5k.md) | adjust_cup_0409（steps_5000）：MAE 0.0410、夹爪判定 98.4% |
| [`reports/openloop_pick_open_place_0724.md`](./reports/openloop_pick_open_place_0724.md) | pick_open_place_0724（10k / 20k 对比） |

---

## 0. 三个脚本的分工

全部位于 `scripts/`，按顺序使用：

| 脚本 | 作用 | 是否必需 |
|---|---|---|
| **`eval_openloop.py`** | **主测试**。加载权重 → 批量推理 → 计算指标 → 输出 `metrics.json` / `predictions.npz` / 3 张基础图 | ✅ 必需 |
| **`analyze_openloop.py`** | **补充分析**。在 `predictions.npz` 上补算 persistence 基线、state-hold 基线、滚动时域指标、夹爪判定准确率 | 推荐 |
| **`plot_openloop_trajectory.py`** | **轨迹可视化**。5 张轨迹拟合图，需要先用 `--dense_stride` 跑出密集窗口 | 需要看图时用 |

> `eval_openloop.py` 是仓库已有的脚本（本说明编写时已由使用者修正过若干问题，见 §7）。

---

## 1. 前置条件

### 1.1 数据集必须是 LeRobot **v2.1**

piper 这条线（训练与评测的全部现有配置）都走 `dataset_py: lerobot_datasets`，
**只吃 v2.1**。仓库里虽然另有 `lerobot_v3_datasets.py`，但它需要 `img_keys` /
`state_key` / `action_key` / `resize_size` 等额外配置项，目前没有任何 piper 配置接入它。

如果你的数据是 v3.0（`meta/` 下有 `episodes/` 目录、`data/chunk-000/file-000.parquet`），
必须先转换：

```bash
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /path/to/xxx_1_offset_state \
  --output /path/to/xxx_1_offset_state_v2_1 \
  --fps 10 --chunk-size 100
```

转换后应存在这些文件（缺一不可）：

```
<dataset>/meta/info.json          # codebase_version 必须是 "v2.1"
<dataset>/meta/modality.json      # 定义 state/action/video 的字段切片
<dataset>/meta/episodes.jsonl
<dataset>/meta/stats_gr00t.json
<dataset>/data/chunk_00000/episode_000000.parquet
<dataset>/videos/observation.images.image/chunk_00000/episode_000000.mp4
```

**重要**：`meta/steps_2d5a34b904d2.pkl` 是采样索引缓存，文件名**硬编码**。
复制数据集时务必删掉它，否则会加载到别的数据集的索引（串味）。

### 1.2 需要一份训练配置 `config.yaml`

直接复用训练时落盘的那份：`checkpoints/<run_id>/config.yaml`。

这份配置里的 `datasets.vla_data.data_root_dir` 指向**训练时用的数据集**。
若要测别的数据集，改这一行即可（其余不用动）：

```yaml
datasets:
  vla_data:
    data_root_dir: /path/to/your_dataset_v2_1    # ← 只改这里
    data_mix: piper_pick_place                   # piper 机器人固定用这个
```

> `mixtures.py` 里 `piper_pick_place` 的子目录名是**空串**，所以 `data_root_dir` 要
> **直接指向数据集目录本身**，不要再套一层。

### 1.3 环境与显存

```bash
/opt/conda/envs/VLA_JEPA/bin/python    # Python 3.10
```

- 单卡 RTX 4090 24 GB 足够；`--batch_size 4` 时显存占用约 **7.5 GB**
- 加载权重需要约 7 GB 内存（先 load 到 CPU 再 `.cuda()`）
- 一次全量测试（400 窗口）约 7 分钟

---

## 2. 快速开始

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# ① 主测试
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/adjust_cup_10k/config.yaml \
  --checkpoint  checkpoints/adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt \
  --output_dir  eval_openloop/adjust_cup_5k \
  --windows_per_episode 8 --batch_size 4 --seed 0

# ② 补充基线分析（自动从 metrics.json 读数据集路径）
/opt/conda/envs/VLA_JEPA/bin/python scripts/analyze_openloop.py \
  --predictions eval_openloop/adjust_cup_5k/predictions.npz

# ③ 需要轨迹图时：先跑密集扫描，再画图
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/adjust_cup_10k/config.yaml \
  --checkpoint  checkpoints/adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt \
  --output_dir  eval_openloop/adjust_cup_5k_dense \
  --dense_stride 1 --num_episodes 4 --dense_oversample 4 --batch_size 4 --seed 0

/opt/conda/envs/VLA_JEPA/bin/python scripts/plot_openloop_trajectory.py \
  --predictions eval_openloop/adjust_cup_5k_dense/predictions.npz \
  --out_dir eval_openloop/adjust_cup_5k_dense \
  --episodes 0 1 2 --zoom_episode 0
```

---

## 3. `eval_openloop.py` 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config_yaml` | 必填 | 训练配置路径 |
| `--checkpoint` | 必填 | `pytorch_model.pt` 或 `steps_N_pytorch_model.pt` |
| `--output_dir` | 必填 | 输出目录，不存在会自动创建 |
| `--windows_per_episode` | 4 | 每个 episode 取几个时间窗。**建议 8**（统计更稳） |
| `--batch_size` | 8 | 推理批大小。24 GB 卡上 4 稳妥；OOM 就降到 2 |
| `--num_inference_timesteps` | 4 | 动作头 flow matching 的去噪步数。**默认 4 与训练配置一致，一般不用改** |
| `--seed` | 0 | 采样种子。**必须固定**，否则每次结果都不同（见 §7.1） |
| `--num_episodes` | 0 | 测前 N 个 episode，0 = 全部 |
| `--dense_stride` | 0 | >0 时忽略 `--windows_per_episode`，每 N 帧取一个窗口（画轨迹图用），配合 `--num_episodes` 限制范围 |
| `--dense_oversample` | 4 | 密集模式下扫描 `N × len(dataset)` 个索引。**别设成 1**（见 §7.4） |

### 两种窗口选择模式

- **默认模式**（`--dense_stride 0`）：每个 episode 均匀取 `windows_per_episode` 个点。
  适合**算指标**，覆盖全部 episode。
- **密集模式**（`--dense_stride >0`）：逐帧扫描指定的前 N 个 episode。
  适合**画轨迹图**，因为只有点足够密才能连成曲线。

---

## 4. `analyze_openloop.py` 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--predictions` | 必填 | `eval_openloop.py` 产出的 `predictions.npz` |
| `--out_dir` | npz 所在目录 | 写 `extra_metrics.json` 和 `baselines.npz` 的位置 |
| `--data_root_dir` | 自动读取 | 从同目录 `metrics.json` 的 `config.data_root_dir` 取；取不到才需要手动指定 |

---

## 5. `plot_openloop_trajectory.py` 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--predictions` | 必填 | **必须是密集模式**产出的 `predictions.npz` |
| `--out_dir` | npz 所在目录 | 图片输出位置 |
| `--episodes` | npz 里前 3 个 | 要画哪些 episode，如 `--episodes 0 1 2` |
| `--zoom_episode` | `episodes[0]` | 单块放大图用哪个 episode |

---

## 6. 输出文件说明

```
<output_dir>/
├── metrics.json            # 主指标：整体/逐维/逐 step/逐 episode/最差 episode
├── predictions.npz         # 原始预测与真值（pred / gt / meta / action_min / action_max）
├── scatter.png             # 预测 vs 真值散点，逐维 R²
├── horizon.png             # 误差随 chunk 位置变化
├── trajectories.png        # 基础轨迹图（只有 step 0，点稀疏）
│
├── extra_metrics.json      # ← analyze_openloop.py 产出
├── baselines.npz           # state-hold 基线数值（hold / state_norm）
│
├── trajectory_summary.json # ← plot_openloop_trajectory.py 产出
├── trajectory_fit.png      # ★ 主图：全 episode 轨迹拟合
├── trajectory_zoom.png     # ★ 单个动作块放大
├── trajectory_2d.png       # ★ 关节空间 2D 相图
├── error_over_time.png     # 误差随时间的分布
└── gripper_timeline.png    # 夹爪通道时序
```

---

## 7. 指标怎么看（重点）

### 7.1 先看「跑赢了哪个基线」

`metrics.json` 里的 `mae_improvement_over_zero_baseline`（相比"全预测 0"）**会严重高估模型**：
本项目的动作是**绝对关节位置**，相邻帧高度自相关，所以"预测 0"这个基线本身就很弱。

**真正该看的是 `extra_metrics.json` 里的 `state_hold_baseline`** ——
把当前测得的 state 重复整段 chunk，即"什么都不做"控制器实际会发出的指令。

判断标准：

| 现象 | 含义 |
|---|---|
| 整段 chunk 输给 state-hold | ⚠️ 模型没学到有效策略（高 R² 是自相关假象） |
| 整段 chunk 赢 state-hold 20%+ | ✅ 学到了真实策略 |
| step 0 输给 state-hold | **这不是"基线作弊"**——state 也是模型的输入，两者信息对等，属公平比较。说明模型对"下一帧目标位置"的预测不如"直接把当前关节位置当目标"这个平凡估计（10 Hz 下两者本就极接近，模型还额外引入 flow matching 的采样方差）。只执行首步的滚动时域控制会受影响 |
| step 1 起稳定赢 state-hold | ✅ 近程响应有效，可放心做滚动时域部署 |
| 直到 step 2-3 才反超，或 step 0 差距 >100% | ⚠️ 近程欠响应明显，建议整段执行或改用增量动作表示 |

**务必逐 step 看**（`state_hold_baseline.per_step`），整体 MAE 会掩盖"前几步特别差"的问题。

### 7.2 夹爪单独看

夹爪是抓取-放置任务最关键、也通常是最弱的通道。看两个数：

- `metrics.json` → `normalized.mae_per_dim[6]`（MAE 通常最高）
- `extra_metrics.json` → `gripper_decision.accuracy`（开合判定准确率）

还要看 `pred_open_fraction` 与 `gt_open_fraction` 是否接近——差太多说明有系统性偏置
（例如一直偏"张开"，会让机器人抓不住东西）。

### 7.3 误差随步长增长

`normalized.mae_per_horizon_step`：

- 近似线性增长且末尾没爆 → chunk 长度合理
- `step6/step0` 超过 3 倍 → 后段预测不可信，考虑缩短 chunk

### 7.4 与训练日志交叉验证

`checkpoints/<run_id>/tensorboard/` 里有训练期的 `mae_score`（同一套归一化 MAE）。
**开环测出来的 MAE 应该和它非常接近**，否则说明测试协议有问题。实测参考：

| 测试 | 训练期 mae_score | 开环 MAE | 差异 |
|---|---|---|---|
| adjust_cup step 5000 | 0.0408 | 0.0410 | 0.0002 |
| pick_open_place_0724 step 20000 | 0.0682 | 0.0743 | 0.0061 |

差在 0.01 以内都算正常（开环用的是确定性窗口，训练期用的是随机 batch，采样分布略有不同）。

```bash
/opt/conda/envs/VLA_JEPA/bin/python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
for ev in sorted(glob.glob('checkpoints/<run_id>/tensorboard/events.out.*')):
    ea=EventAccumulator(ev,size_guidance={'scalars':0}); ea.Reload()
    if 'mae_score' in ea.Tags().get('scalars',[]):
        s=ea.Scalars('mae_score'); print(ev.split('/')[-1], s[-1].step, round(s[-1].value,4))
"
```

---

## 8. 已知坑位清单

### 8.1 动作头是随机采样，不固定种子结果不可复现 ⚠️

动作头（`GR00T_ActionHeader.predict_action`）从 `torch.randn` 起始做 flow matching，
**每次运行结果都不同**。必须传 `--seed`。

### 8.2 `--ddim_steps` 是历史遗留的无效参数 ⚠️

`VLA_JEPA.predict_action` 只把 2 个参数透传给动作头，
老的 `use_ddim=True, num_ddim_steps=20` 会被 `**kwargs` **静默吃掉**，
代码里也**根本没有 DDIM 分支**。真正生效的是
`config.framework.action_model.num_inference_timesteps`（本仓库配置为 4）。
现脚本已改为真正设置该属性，参数名也换成了 `--num_inference_timesteps`。

### 8.3 语言指令可能是无意义的数字 ⚠️

部分转换脚本会把 `meta/tasks.jsonl` 的任务文本写成裸数字（`"0"`, `"22"`），
模型训练时看到的指令就是数字。这种情况下**语言条件实际无效**，
测试结论只覆盖「视觉 + 状态 → 动作」。检查方法：

```bash
head -3 <dataset>/meta/tasks.jsonl
# {"task_index": 0, "task": "0"}      ← 数字占位符，语言通道被浪费
# {"task_index": 0, "task": "把杯子..."}  ← 正常
```

### 8.4 `--dense_oversample` 不能设成 1

`LeRobotMixtureDataset.sample_step` 用 `rng.choice(...)` **带放回抽样**把索引映射到
`(trajectory, step)`。所以遍历 `len(dataset)` 个索引只能覆盖约 **63%** 的帧
（优惠券收集问题），轨迹图会有明显断点。过采样 4 倍后覆盖率达 ~98%。

### 8.5 中文字体渲染成方块

系统唯一的中文字体 `Droid Sans Fallback` **完全没有 ASCII 字形**，
直接用会让所有坐标数字变豆腐块。且必须把字体列表赋给 **`font.family`**，
赋给 `font.sans-serif` **不会**触发 matplotlib 的逐字形回退：

```python
plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]  # ✅
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]  # ❌ 无效
```

`plot_openloop_trajectory.py` 的 `setup_fonts()` 已处理。验证是否还有缺字形：

```bash
python -W all scripts/plot_openloop_trajectory.py ... 2>&1 | grep -c "missing from font"   # 应为 0
```

### 8.6 后台任务用 `pkill -f`，不要用 `kill $!`

`CUDA_VISIBLE_DEVICES=0 nohup python xxx &` 这种写法，`$!` 拿到的是 **bash 子 shell 的 PID**，
不是 python 的。`kill $!` 只会杀掉外壳，python 继续跑并写同一个输出目录/日志，造成数据串台。
建议：

```bash
nohup python scripts/eval_openloop.py ... > log/xxx.log 2>&1 &
echo $!                                    # 记下这个（但它是子 shell，不一定是 python）
ps -eo pid,cmd | grep eval_openloop        # 确认真实 python PID，对不上就杀这个
pkill -f "eval_openloop/xxx"               # 最可靠：按输出目录特征杀
```

### 8.7 权重加载出现 missing / unexpected keys

正常情况应该打印 `missing keys: 0 (critical: 0)` 和 `unexpected keys: 0`。
若非 0，脚本会检查是否缺 `action_model` / `qwen` / `vj_` 关键权重并直接报错。
常见原因：配置与权重不匹配（例如拿 A 任务的配置去加载 B 任务的权重）。

### 8.8 确认权重与数据集是配套的

不同 run 的权重互不通用。核对方法：

```bash
grep -E "run_id|data_root_dir" checkpoints/<run_id>/config.yaml
```

`data_root_dir` 就是该权重训练时用的数据集。**用别的数据集测属于跨任务零样本测试，
结果差不代表权重有问题。**

---

## 9. 典型工作流：测一个新的 checkpoint

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# 0) 确认权重与数据集配套
grep -E "run_id|data_root_dir" checkpoints/<run_id>/config.yaml

# 1) 冒烟（2 episode × 2 窗口，约 3 分钟，验证权重能加载、不 OOM）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_openloop.py \
  --config_yaml checkpoints/<run_id>/config.yaml \
  --checkpoint  checkpoints/<run_id>/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/<run_id>_smoke \
  --num_episodes 2 --windows_per_episode 2 --batch_size 2 --seed 0
# 期望看到：missing keys: 0 (critical: 0) / unexpected keys: 0

# 2) 全量指标
CUDA_VISIBLE_DEVICES=0 python scripts/eval_openloop.py \
  --config_yaml checkpoints/<run_id>/config.yaml \
  --checkpoint  checkpoints/<run_id>/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/<run_id> \
  --windows_per_episode 8 --batch_size 4 --seed 0

# 3) 基线分析（关键！）
python scripts/analyze_openloop.py --predictions eval_openloop/<run_id>/predictions.npz

# 4) 轨迹图
CUDA_VISIBLE_DEVICES=1 python scripts/eval_openloop.py \
  --config_yaml checkpoints/<run_id>/config.yaml \
  --checkpoint  checkpoints/<run_id>/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/<run_id>_dense \
  --dense_stride 1 --num_episodes 4 --dense_oversample 4 --batch_size 4 --seed 0
python scripts/plot_openloop_trajectory.py \
  --predictions eval_openloop/<run_id>_dense/predictions.npz \
  --episodes 0 1 2 --zoom_episode 0
```

---

## 10. 相关文档

| 文档 | 内容 |
|---|---|
| [`reports/openloop_adjust_cup_5k.md`](./reports/openloop_adjust_cup_5k.md) | adjust_cup_0409（steps_5000）开环测试报告 |
| [`reports/openloop_pick_open_place_0724.md`](./reports/openloop_pick_open_place_0724.md) | pick_open_place_0724（10k/20k）开环测试报告 |
| [`01_data_conversion.md`](./01_data_conversion.md) | 训练前的数据转换（v3.0 → v2.1）与机器人注册 |
| [`02_training.md`](./02_training.md) | 训练启动命令说明 |
| [`archive/adjust_cup_real_world.md`](./archive/adjust_cup_real_world.md) | adjust_cup 真机数据接入与训练全过程 |
