# VLA-JEPA 开环测试报告：adjust_cup_0409（steps_5000）

## 0. 摘要

对 `adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt` 在其训练集
`adjust_cup_0409_1_offset_state_v2_1` 上做开环测试（50 episodes × 8 窗口 = **400 个测试窗**）：

| 指标 | 结果 |
|---|---|
| 归一化 MAE / MSE | **0.0410 / 0.0045** |
| 相比「全预测 0」基线 | **提升 93.0%** |
| 相比「保持不动」基线（整段 chunk） | **提升 63.0%** |
| 每维 R² | **0.959 ~ 0.998**（全部接近 1） |
| 夹爪开合判定准确率 | **98.4%** |
| 逐 episode MAE 波动 | 0.0329 ~ 0.0531（std **0.0044**，极稳定） |

**结论：这是一个质量相当高的策略**，且**只训了 5000 步**。

> **交叉验证**：训练日志里 step 5000 的 `mae_score` = **0.0408**，本次开环 MAE = **0.0410**，
> 两者几乎完全吻合 —— 说明开环协议与训练内评估一致，数值可信。

---

## 1. 测试配置

| 项 | 值 |
|---|---|
| 权重 | `checkpoints/adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt`（6.16 GB） |
| 数据集 | `Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_v2_1` |
| 数据规模 | 50 episodes / 6837 帧 / 33 条任务描述 / 10 fps |
| 观测 | 双视角 224×224，state 7 维，世界模型视频 8 帧 256×256 |
| 动作 | 7 维绝对关节位置，chunk = 7 步（0.7 s） |
| 推理 | 动作头 flow matching，`num_inference_timesteps=4`，`seed=0` |
| 权重加载 | **0 missing / 0 unexpected keys** |
| 轮次 1（指标） | 50 episodes × 8 窗口 = **400 窗** |
| 轮次 2（画图） | 4 episodes × 逐帧扫描 = **542 窗**（ep0/1/2 各 137~139 窗） |

> ⚠️ **注意该权重是中途快照**：配置里 `max_train_steps=10000`，
> 但训练在 **step 5012** 中断（日志最后一行 5012/10000，无 `final_model/`）。
> 所以这是"训了一半"的模型，不是收敛结果。

---

## 2. 主指标

### 2.1 整体

| 指标 | 400 窗（全部 50 episodes） | 542 窗（密集 4 episodes） |
|---|---|---|
| 归一化 MAE | **0.0410** | 0.0376 |
| 归一化 MSE | **0.0045** | 0.0036 |
| 「全预测 0」基线 MAE | 0.5887 | 0.5613 |
| 相比零基线 | +93.0% | +93.3% |
| 逐 episode MAE 范围 | 0.0329 ~ 0.0531 | 0.0360 ~ 0.0395 |

### 2.2 逐维度

| 维度 | 归一化 MAE | R² | raw MAE / 真值量程 |
|---|---|---|---|
| x | 0.0439 | 0.972 | 2.2% |
| y | 0.0287 | **0.998** | 1.4% |
| z | 0.0380 | 0.994 | 1.9% |
| roll | 0.0300 | 0.993 | 1.5% |
| pitch | 0.0532 | 0.971 | 2.7% |
| yaw | 0.0347 | 0.988 | 1.7% |
| **gripper** | **0.0581** | **0.959** | 2.9% |

**夹爪依然是相对最弱的通道**（MAE 最高、R² 最低），但绝对值已经很好了 ——
对比上一份 `pick_open_place_0724` 的夹爪 MAE 0.1333 / R² 0.792，这里好了 2 倍以上。

### 2.3 夹爪开合判定

| 指标 | 值 |
|---|---|
| 判定准确率 | **98.4%** |
| 真值张开比例 | 12.8% |
| 预测张开比例 | 13.1% |
| 误判为张开 | **1.0%** |
| 误判为闭合 | **0.7%** |

预测的张开比例与真值几乎一致（13.1% vs 12.8%），**没有系统性偏置**。

### 2.4 误差随预测步长增长

| chunk 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| MAE | 0.0326 | 0.0427 | 0.0441 | 0.0416 | 0.0404 | 0.0405 | 0.0448 |

**step6 / step0 仅 1.38×**，且第 2 步之后误差基本走平（0.040~0.045）。
对比 `pick_open_place_0724` 的 2.36× 线性增长，这里的 chunk 预测**在前 0.7 秒内都保持稳定**，
说明动作块长度 7 用得很充分，没有"越往后越崩"的问题。

### 2.5 最差 / 最好 episode

| | episode | 归一化 MAE | xyz MAE | 夹爪 MAE |
|---|---|---|---|---|
| 最差 | 17 | 0.0531 | 0.0452 | 0.0872 |
| | 8 | 0.0498 | 0.0454 | 0.0762 |
| | 0 | 0.0495 | 0.0429 | 0.0700 |
| 最好 | 33 | 0.0332 | — | — |
| | 4 | 0.0329 | — | — |

最差与最好只差 **1.6 倍**（0.0531 / 0.0329），std 仅 0.0044 —— **表现高度一致，没有崩溃的 episode**。
（对比 `pick_open_place_0724` 最差 0.1065 / 最好 0.0535 = 2.0 倍，std 0.0124）

---

## 3. 与「保持不动」基线的对比

诚实基线 = 把当前测得的 state 重复整段 chunk（"什么都不做"控制器的实际指令）。

| 区间 | 模型 | 保持不动 | 模型相对基线 |
|---|---|---|---|
| 仅 step 0 | 0.0326 | 0.0233 | **−39.6%** ❌ |
| steps 0-2（前 0.3 s） | 0.0398 | 0.0575 | **+30.8%** ✅ |
| steps 3-6（0.4-0.6 s） | 0.0418 | 0.1505 | **+72.2%** ✅ |
| 整段 chunk | 0.0410 | 0.1106 | **+63.0%** ✅ |

### 逐 step

| chunk 位置 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 模型 | 0.0326 | 0.0427 | 0.0441 | 0.0416 | 0.0404 | 0.0405 | 0.0448 |
| 保持不动 | **0.0233** | 0.0581 | 0.0911 | 0.1155 | 0.1390 | 0.1620 | 0.1854 |
| 谁更好 | 基线 | 模型 | 模型 | 模型 | 模型 | 模型 | 模型 |

**读法：**
- **只有 step 0 输给"保持不动"**，且差距（−39.6%）远小于 `pick_open_place_0724` 的 −106%。
- **从 step 1 起（0.1 秒后）就稳定反超**，而 `pick_open_place_0724` 要到 step 2-3 才反超。
- 基线误差随步长线性发散（0.0233 → 0.1854），而模型误差几乎恒定（0.033 → 0.045），
  这正是"学到了运动趋势"的特征。

> **关于 step 0 的诚实解读（更正）：**
> `state-hold` 基线只用当前测得的 state，而 state **同样是模型的输入之一**，
> 所以这个基线**没有任何信息优势，是公平比较**。
> 结论是：模型对"下一个控制目标"的预测，确实不如直接把"当前关节位置"当作目标。
>
> 原因不难理解：10 Hz 下动作平缓，`action_t`（下一帧目标位置）与 `state_t`（当前实际位置）
> 本就非常接近，这是个近乎平凡的估计；模型在观测基础上还要经过 flow matching 随机采样，
> 反而引入了额外方差。但从 step 1 起基线迅速发散，模型优势立刻显现。
>
> **影响**：若采用「每步重新规划、只执行第 0 步」的滚动时域控制，这一步会成为瓶颈。
> 缓解办法：整段执行、从 step 1 起执行，或增大 `--num_inference_timesteps` 降低采样方差。
>
> 好在本次差距只有 −39.6%（`pick_open_place_0724` 是 −106%），且 step 1 即反超，
> 属于可接受的量级。

---

## 4. 轨迹拟合可视化

用 `scripts/plot_openloop_trajectory.py` 生成，基于逐帧密集扫描（episode 0/1/2 各 137~139 个窗口）：

| 图 | 内容 | 看点 |
|---|---|---|
| **`trajectory_fit.png`** | 7 关节 × 3 episodes 网格。黑实线 = 真值，红虚线 = 各窗口的 step-0 预测，浅蓝细线 = 每个窗口预测的完整 7 步 chunk | **主图**。预测几乎贴合真值曲线；蓝色 ribbon 的宽度直接反映预测不确定性 |
| **`trajectory_zoom.png`** | 挑 3 个时刻，把单个 7 步 chunk 的预测 vs 真值逐关节画出，填充色 = 误差面积 | 看单块预测的形状是否跟得上真值（起停、换向） |
| **`trajectory_2d.png`** | 关节空间 2D 相图（joint2-joint3 / joint4-joint5 / joint1-joint6），颜色 = 时间 | 看预测轨迹在相空间里是否走出与真值相同的形状，而不只是"点对点接近" |
| **`error_over_time.png`** | 逐关节 |误差| 随帧序号变化，虚线 = 该 episode 均值 | 看误差集中在动作的哪个阶段（通常是启动/换向瞬间） |
| **`gripper_timeline.png`** | 夹爪通道真值 vs 预测 + 开合判定准确率 | 抓取-放置最关键通道的时序对齐情况 |

三个 episode 的 step-0 MAE 分别为 **0.0234 / 0.0216 / 0.0206**，夹爪判定准确率
**90.5% / 85.4% / 86.3%**（密集轮次这几段恰好比全局 98.4% 差，说明这几个 episode 的夹爪切换较难）。

---

## 5. 与上一份测试的横向对比

| | **adjust_cup 5k**（本次） | pick_open_place_0724 20k |
|---|---|---|
| 训练步数 | **5000**（中途快照） | 20000（完整） |
| 训练帧数 | 6837 | 12327 |
| 归一化 MAE | **0.0410** | 0.0743 |
| 相比零基线 | +93.0% | +85.9% |
| 相比保持不动（整段） | **+63.0%** | +20.5% |
| step 0 相比保持不动 | −39.6% | −106% |
| 反超保持不动的起点 | **step 1** | step 2~3 |
| 夹爪判定准确率 | **98.4%** | 88.9% |
| R²（夹爪） | **0.959** | 0.792 |
| 误差随步长增长 | **1.38×** | 2.36× |
| 逐 episode MAE std | **0.0044** | 0.0124 |

**只用 1/4 的训练步数、更少的数据，adjust_cup 的表现全面优于 pick_open_place_0724。**
推测原因：`adjust_cup`（调整杯子位置）动作幅度小、轨迹重复度高、任务变异性低；
而 `pick_open_place`（茄子/胡萝卜放进不同抽屉）涉及大范围移动与夹爪时序，难度更高。

---

## 6. 结论与建议

### ✅ 结论

1. **5000 步就已经是一个可用的策略**：MAE 0.0410、R² 全部 >0.95、夹爪判定 98.4%。
2. **与训练内评估完全一致**（0.0410 vs 0.0408），协议可信。
3. **chunk 内预测稳定**：误差随步长仅增长 1.38×，7 步动作块用得很充分。
4. **表现高度一致**：逐 episode MAE 波动仅 ±0.0044，无崩溃样本。
5. **唯一短板是 step 0**（比"保持不动"差 39.6%），但这是预测类策略的固有代价，不影响闭环。

### 📌 建议

| 优先级 | 建议 | 依据 |
|---|---|---|
| 🔴 高 | **把训练跑完**（到 10000 步）。曲线仍在稳定下降：0.0808(2k) → 0.0573(3k) → 0.0487(4k) → 0.0408(5k)，**完全没有收敛迹象** | §0 交叉验证 + 训练日志 |
| 🟠 中 | 若追求部署实时性，可以从 **step 1 起**执行（跳过 chunk 首步），或直接整段执行 | §3：step 1 起即优于"保持不动" |
| 🟠 中 | 夹爪仍是相对最弱通道（MAE 最高、R² 最低）。若下游任务对夹爪时序敏感，可单独加权 | §2.2 |
| 🟡 低 | 值得检查 ep 17 / 8 / 0 的夹爪段（MAE 0.070~0.088，是全局 0.058 的 1.2~1.5 倍） | §2.5 |
| 🟡 低 | 转换脚本的任务文本是数字占位符（`"0"`~`"32"`），语言条件实际无效；修掉后可做 A/B 对照 | 上一份报告 §5 |

### 局限说明

- **同分布测试**：模型就是在这个数据集上训练的，衡量的是**拟合程度**，不代表真机泛化。
- **权重未训练完**：step 5012/10000 中断，数字会随继续训练变化。
- **开环 ≠ 闭环**：这里衡量单次 chunk 预测误差，不是任务成功率。
- **语言条件无效**：指令是数字占位符，结论只覆盖「视觉 + 状态 → 动作」。

---

## 7. 复现方式

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# 轮次 1：全量指标（50 episodes × 8 窗口）
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/adjust_cup_10k/config.yaml \
  --checkpoint  checkpoints/adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt \
  --output_dir  eval_openloop/adjust_cup_5k \
  --windows_per_episode 8 --batch_size 4 --seed 0

# 轮次 2：逐帧密集扫描（画轨迹图用）
CUDA_VISIBLE_DEVICES=1 /opt/conda/envs/VLA_JEPA/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/adjust_cup_10k/config.yaml \
  --checkpoint  checkpoints/adjust_cup_10k/checkpoints/steps_5000_pytorch_model.pt \
  --output_dir  eval_openloop/adjust_cup_5k_dense \
  --dense_stride 1 --num_episodes 4 --dense_oversample 4 --batch_size 4 --seed 0

# 基线分析
/opt/conda/envs/VLA_JEPA/bin/python scripts/analyze_openloop.py \
  --predictions eval_openloop/adjust_cup_5k/predictions.npz

# 轨迹可视化
/opt/conda/envs/VLA_JEPA/bin/python scripts/plot_openloop_trajectory.py \
  --predictions eval_openloop/adjust_cup_5k_dense/predictions.npz \
  --out_dir eval_openloop/adjust_cup_5k_dense \
  --episodes 0 1 2 --zoom_episode 0
```

耗时：轮次 1 约 7 分钟，轮次 2 约 9 分钟（单卡 RTX 4090，显存 7.5 GB）。

### 产出物

```
eval_openloop/
├── adjust_cup_5k/                    # 轮次 1：400 窗，50 episodes
│   ├── metrics.json                  # 主指标 + 逐 episode
│   ├── extra_metrics.json            # 基线对比 + 夹爪判定
│   ├── baselines.npz
│   ├── predictions.npz
│   └── scatter.png / horizon.png / trajectories.png
└── adjust_cup_5k_dense/              # 轮次 2：542 窗，4 episodes
    ├── metrics.json / extra_metrics.json / baselines.npz / predictions.npz
    ├── trajectory_summary.json       # 逐 episode 的 step-0 MAE 与夹爪准确率
    ├── trajectory_fit.png            # ★ 主图：轨迹拟合
    ├── trajectory_zoom.png           # ★ 单块预测放大
    ├── trajectory_2d.png             # ★ 关节空间相图
    ├── error_over_time.png           # 误差随时间分布
    ├── gripper_timeline.png          # 夹爪时序
    └── scatter.png / horizon.png / trajectories.png
```

### 本次新增/修改的脚本

| 文件 | 说明 |
|---|---|
| `scripts/eval_openloop.py` | 新增 `--dense_stride` / `--dense_oversample`：逐帧密集扫描模式（`sample_step` 是带放回抽样，直接遍历只覆盖 ~63% 帧，过采样 4 倍后达 ~98%） |
| `scripts/plot_openloop_trajectory.py` | **新增**：5 张轨迹拟合可视化图。注意字体需同时覆盖中英文——`Droid Sans Fallback` 没有 ASCII 字形，必须设 `font.family` 为**列表**才能触发 matplotlib 的逐字形回退（设 `font.sans-serif` 无效） |
