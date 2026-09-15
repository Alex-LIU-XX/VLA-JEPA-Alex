# 02 · 训练启动命令说明

VLA-JEPA 在 Piper 真机数据上的训练启动方式、配置要点与排错。

> **前置步骤**：数据必须先转成 LeRobot v2.1 并注册机器人类型，
> 见 [`01_data_conversion.md`](./01_data_conversion.md)。
> **后续步骤**：训练完成后做开环测试，见 [`03_openloop_testing.md`](./03_openloop_testing.md)。

---

## 目录

1. [总览](#1-总览)
2. [环境准备](#2-环境准备)
3. [预训练权重](#3-预训练权重)
4. [配置文件关键项](#4-配置文件关键项)
5. [启动命令](#5-启动命令)
6. [训练输出](#6-训练输出)
7. [监控与日志](#7-监控与日志)
8. [常见问题](#8-常见问题)
9. [原理速览](#9-原理速览)
10. [命令速查](#10-命令速查)

---

## 1. 总览

VLA-JEPA 有两阶段训练，**两阶段可独立运行**：

| 阶段 | 入口脚本 | 配置文件 | 数据 | 本项目是否使用 |
|---|---|---|---|---|
| **① Co-training** | `starVLA/training/train_vlajepa_cotrain.py` | `scripts/config/vlajepa_cotrain.yaml` | 机器人数据 + 人类视频 | ❌ 否 |
| **② Robot Fine-tuning** | `starVLA/training/train_starvla.py` | `scripts/config/*.yaml` | 仅机器人数据 | ✅ **是** |

**本项目走的是阶段二**：直接从 Qwen3-VL-2B + V-JEPA2 原始权重开始微调，
不需要阶段一的 checkpoint（`pretrained_checkpoint` 是可选的）。

阶段一只有在你想用大规模人类视频预热世界模型时才需要。

---

## 2. 环境准备

### 2.1 conda 环境

```bash
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

本机已有环境：**`/opt/conda/envs/VLA_JEPA`**（Python 3.10.20），下面命令统一用它。

关键依赖版本：

| 包 | 版本 | 用途 |
|---|---|---|
| `transformers` | 4.57.0 | Qwen3-VL 加载 |
| `accelerate` | 1.5.2 | 分布式启动 |
| `deepspeed` | 0.16.9 | ZeRO 优化 |
| `av` | 12.3.0 | 视频解码（AV1 必需） |
| `ninja` | — | **编译 DeepSpeed CPU Adam 算子，必须装** |

### 2.2 验证

```bash
/opt/conda/envs/VLA_JEPA/bin/python -c \
  "import torch, transformers, accelerate, deepspeed, av; print('OK')"
```

> 若 GPU 不支持 FlashAttention 2，把 YAML 里的 `attn_implementation` 从
> `flash_attention_2` 改成 `eager`。

---

## 3. 预训练权重

### 3.1 必需的（约 5.5 GB）

| 权重 | 大小 | 地址 |
|---|---|---|
| Qwen3-VL-2B-Instruct | ~4 GB | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| V-JEPA2 VIT-L (`vjepa2-vitl-fpc64-256`) | ~1.5 GB | https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 |

本项目已下载到：

```
checkpoints/Qwen/Qwen3-VL-2B-Instruct
checkpoints/facebook/vjepa2-vitl-fpc64-256
```

### 3.2 可选的 Co-training 权重（6.16 GB）

https://huggingface.co/ginwind/VLA-JEPA （`Pretrain/` 目录）。
加载后相当于在更多数据上预热过，微调效果可能更好；不加载也能正常训练。

### 3.3 在 YAML 中指向本地路径

```yaml
framework:
  qwenvl:
    base_vlm: /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA/checkpoints/Qwen/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA/checkpoints/facebook/vjepa2-vitl-fpc64-256
```

---

## 4. 配置文件关键项

复制一份现成配置改，例如 `scripts/config/adjust_cup_train.yaml`。

### 4.1 每次训练基本都要改

| 参数 | 位置（约） | 说明 |
|---|---|---|
| `run_id` | 1 | 输出目录名 → `checkpoints/<run_id>/` |
| `datasets.vla_data.data_root_dir` | 53 | v2.1 数据集路径（**直接指向数据集目录本身**） |
| `datasets.vla_data.data_mix` | 54 | `mixtures.py` 中注册的 key，piper 用 `piper_pick_place` |
| `trainer.max_train_steps` | 65 | 总步数 |
| `trainer.save_interval` | 67 | 保存间隔，**必须 < `max_train_steps`** 才会存 |
| `datasets.vla_data.per_device_batch_size` | 59 | 每卡 batch，见 §5.5 |

### 4.2 偶尔调整

| 参数 | 说明 |
|---|---|
| `trainer.num_warmup_steps` | LR 预热步数（正式训练建议 5000） |
| `trainer.learning_rate.action_model` | 动作头 LR（默认 1e-4，通常比 VLM 学得快） |
| `trainer.learning_rate.base` | VLM backbone LR（默认 3e-5） |
| `trainer.freeze_modules` | 冻结模块加速：`qwen_vl` / `action_model` / `vj_encoder` / `vj_predictor` |
| `framework.vj2_model.num_frames` | V-JEPA 输入帧数，显存不足可从 8 改 4 |
| `trainer.eval_interval` | 评估间隔（每次评估跑 20 次前向，太密会拖慢） |

### 4.3 基本不动

`action_dim` / `state_dim`（取决于机器人）、`action_horizon`、
`repetition_diffusion_steps`、`loss_scale`、`noise_beta_*`。

### 4.4 piper 的参考配置

```yaml
framework:
  name: VLA_JEPA
  action_model:
    action_dim: 7
    state_dim: 7
    action_horizon: 7
    future_action_window_size: 6
    num_inference_timesteps: 4
  vj2_model:
    num_frames: 8
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_mix: piper_pick_place
    action_type: absolute
    resolution_size: 224
    video_resolution_size: 256
    with_state: true
```

---

## 5. 启动命令

> ⚠️ **必须用 `accelerate launch` 启动**，直接 `python train_starvla.py` 会因为
> `dist.get_rank()` 在 `accelerator.prepare()` 之前调用而报
> `ValueError: Default process group has not been initialized`。

### 5.1 冒烟测试（1 GPU，约 2 分钟）

用于验证「数据 + 模型 + 保存」全链路能否跑通。**正式训练前务必先跑这个。**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml
```

对应的测试配置（`adjust_cup_test.yaml`）：`max_train_steps: 20`、
`per_device_batch_size: 2`、`num_warmup_steps: 5`、`eval_interval: 10`。

期望输出：

```
参数量      : 2770.332 M
dataloader  : 6837 steps / 50 trajectories
              image list[2]×(224,224), video (2,8,256,256,3) uint8,
              action (7,7), state (1,7)
step  1  action_loss=1.4233  wm_loss=0.1934
...
step 20  action_loss=1.4485  wm_loss=0.1857  mse=0.1087  mae=0.9131
final_model/pytorch_model.pt 6.16 GB 已保存
```

### 5.2 单卡正式训练

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml
```

> 单卡 24 GB **必须**开 ZeRO-2 + CPU Offload，
> 且需要 `pip install ninja`（DeepSpeed CPU Adam 要编译 C++ 扩展），否则会 OOM。
> 见 §8.2。

### 5.3 多卡正式训练（推荐）

**4 卡**：

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml
```

**8 卡**（用仓库自带 shell 脚本）：

```bash
bash scripts/vlajepa_robot_ft.sh
```

`scripts/vlajepa_robot_ft.sh` 内容：

```bash
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft.yaml
```

> 换成自己的配置时，把最后一行 `--config_yaml` 指向你的 YAML 即可。

**后台运行 + 记录日志**：

```bash
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml' \
  > log/adjust_cup_10k.log 2>&1 &
```

> ⚠️ 后台任务的 **`$!` 拿到的是子 shell 的 PID，不是 python 的**。
> 要停止训练请用 `pkill -f "train_starvla"`，或先 `ps -eo pid,cmd | grep train_starvla`
> 找到真实 PID 再 kill。

### 5.4 断点续训

在 YAML 中改：

```yaml
trainer:
  is_resume: true
  resume_epoch: null
  resume_step: 5000        # 或 null = 自动找最近的 checkpoint
```

然后**用完全相同的 `run_id` 和命令**重新启动即可
（`run_id` 必须一致，否则找不到 checkpoint 目录）。

### 5.5 batch size 与显存

```
总 batch size = per_device_batch_size × num_processes × gradient_accumulation_steps
```

| GPU 数 | `per_device_batch_size` | `grad_accum` | 总 batch | 用途 |
|---|---|---|---|---|
| 1 | 2 | 1 | 2 | 冒烟测试 |
| 1 | 4 | 1 | 4 | 单卡训练（需 CPU Offload） |
| 4 | 4 | 1 | 16 | 稳妥 |
| 4 | 8 | 1 | 32 | **推荐** |
| 8 | 8 | 1 | 64 | 大规模 |

**OOM 时的调整顺序**：① 调小 `per_device_batch_size` → ② 增大
`gradient_accumulation_steps`（保持等效 batch）→ ③ `num_frames` 从 8 改 4 →
④ 开 `freeze_modules`。

### 5.6 阶段一 Co-training（本项目未使用）

```bash
bash scripts/vlajepa_cotrain.sh
# 等价于：
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_vlajepa_cotrain.py \
  --config_yaml ./scripts/config/vlajepa_cotrain.yaml
```

---

## 6. 训练输出

```
checkpoints/<run_id>/
├── config.yaml                     # 本次训练使用的配置（快照）
├── config.json                     # 同上，JSON 格式
├── dataset_statistics.json         # 数据集统计量（action/state 的 min/max/mean/std）
├── summary.jsonl                   # 每次保存记录一行 {"steps": N}
├── tensorboard/                    # mae_score / mse_score 曲线
├── checkpoints/
│   ├── steps_5000_pytorch_model.pt     # 每 save_interval 步存一个（各 6.16 GB）
│   └── steps_10000_pytorch_model.pt
└── final_model/
    └── pytorch_model.pt            # 训练结束无条件保存
```

**注意**：

- `config.yaml` 里的 `data_root_dir` 就是该权重训练用的数据集。
  日后核对「权重与数据是否配套」看这里。
- 每个 checkpoint **6.16 GB**，注意磁盘占用。`save_interval` 不要设太小。
- 即使只是冒烟测试，`_finalize_training()` 也会无条件保存 `final_model/`，记得清理。

---

## 7. 监控与日志

### 7.1 训练日志

```bash
# 看 loss 曲线
tail -f log/adjust_cup_10k.log

# 提取 loss 并画图
python scripts/analyze_loss.py \
  --log log/adjust_cup_10k.log \
  --output_dir eval_openloop/adjust_cup_10k
```

日志里每 `logging_frequency` 步打印一次：

```
Step 5000, Loss: {'action_loss': ..., 'wm_loss': ..., 'learning_rate': ...,
                  'mse_score': ..., 'mae_score': ...}
2.77B 参数 / ZeRO-2 / 4 卡约 3.6 s/step
```

### 7.2 TensorBoard

```bash
tensorboard --logdir checkpoints/<run_id>/tensorboard
```

或用脚本直接读取（**开环测试时做交叉验证要用，见 `03_openloop_testing.md` §7.4**）：

```bash
/opt/conda/envs/VLA_JEPA/bin/python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
for ev in sorted(glob.glob('checkpoints/<run_id>/tensorboard/events.out.*')):
    ea=EventAccumulator(ev,size_guidance={'scalars':0}); ea.Reload()
    for t in ea.Tags().get('scalars',[]):
        s=ea.Scalars(t); print(t, 'last step', s[-1].step, '=', round(s[-1].value,4))
"
```

### 7.3 判断训练是否健康

| 指标 | 期望 |
|---|---|
| `action_loss` | 从 ~1.4 稳步下降 |
| `wm_loss` | 从 ~0.19 缓降 |
| `mae_score` | 持续下降；**若已走平就可以停** |
| 显存 | 无 OOM；4 卡 ZeRO-2 大约每卡 20 GB |

**实测收敛速度参考**（adjust_cup，6837 帧）：

| 步数 | 250 | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|---|
| `mae_score` | 0.2191 | 0.2090 | 0.0808 | 0.0573 | 0.0487 | 0.0408 |

5000 步时仍在明显下降，**远未收敛**。这类小数据集（几千帧）建议至少 10000~20000 步。

---

## 8. 常见问题

### 8.1 `ValueError: Default process group has not been initialized`

**原因**：直接 `python train_starvla.py` 单进程运行，但代码在
`accelerator.prepare()` 之前就调用了 `dist.get_rank()`。

**解决**：一定用 `accelerate launch`，即使只有 1 张卡：

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --num_processes 1 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/piper_test.yaml
```

### 8.2 CUDA Out of Memory

```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 7.93 GiB. GPU has 23.52 GiB total.
```

**原因**：2.77B 参数的模型，显存需求远超 24 GB：

| 项 | 占用 |
|---|---|
| 模型权重 (bf16) | ~5.5 GB |
| 梯度 (bf16) | ~5.5 GB |
| Adam 状态 (fp32) | ~22 GB（momentum + variance） |
| **合计** | **>> 24 GB** |

**解决**：
1. 用 **ZeRO-2 + CPU Offload**（把优化器状态挪到 CPU 内存）→ GPU 显存降到 **~5.7 GB**；
2. **`pip install ninja`** —— DeepSpeed 的 CPU Adam 算子需要编译 C++ 扩展，缺了会失败；
3. 或者直接用 4 卡以上，ZeRO-2 分片后可以关掉 CPU Offload。

配置在 `starVLA/config/deepseeds/deepspeed_zero2.yaml` 和 `ds_config_test.json`：

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

### 8.3 `ValueError: No suitable position columns found`

见 [`01_data_conversion.md` §6.3](./01_data_conversion.md#63-valueerror-no-suitable-position-columns-found)。

### 8.4 `KeyError: 'info'`

`info.json` 的 video feature 缺 `info` 块，
见 [`01_data_conversion.md` §6.6](./01_data_conversion.md#66-valueerror-channel-is-not-in-list--keyerror-info)。

### 8.5 加载 checkpoint 时 key 不匹配

**原因**：配置与 checkpoint 不一致（例如 `action_dim` 不同）。

**解决**：确保 `action_dim` / `state_dim` / `action_horizon` 与训练时一致。
推理脚本（`eval_openloop.py`）用的是 `strict=False`，会打印
`missing keys` / `unexpected keys` 数量，正常应为 **0 / 0**。

### 8.6 多卡数据重复

`get_vla_dataset()` 的 `seed` 固定，各 rank 的 `(epoch, index, seed)` 相同。
仓库 `build_dataloader()` 已做了 `seed = cfg.seed + rank` 的偏移来缓解。
若你发现多卡数据多样性仍异常，检查这一段。

### 8.7 `eval_interval` 在 step 0 会触发一次评估

`completed_steps % eval_interval == 0` 在 step 0 成立，所以启动后会先跑一次
评估（内部 20 次前向），首次启动会慢一点，属正常现象。

---

## 9. 原理速览

> 完整原理见 [`archive/training_pipeline.md`](./archive/training_pipeline.md)。

### 9.1 架构

```
输入 (image, video, lang, action, state)
  │
  ├─ Qwen3-VL-2B ──→ 文本/图像编码 ──→ action_tokens + embodied_action_tokens
  │                       │                        │
  │                       ▼                        ▼
  │               V-JEPA Predictor          DiT Action Head (Flow Matching)
  │               (预测未来视频token)       (预测未来动作轨迹)
  │                       │                        │
  │                  wm_loss (L1)            action_loss (MSE)
  │                   × 0.1                  × 1.0
  └───────────────────────┴────────────────────────┘
                          total_loss
```

| 组件 | 来源 | 说明 |
|---|---|---|
| Qwen3-VL-2B | Qwen | VLM 骨干 |
| V-JEPA2 VIT-L | Meta/FAIR | 视频编码器（训练时 frozen） |
| V-JEPA Predictor | 自定义 | Action-Conditioned Transformer |
| DiT-B Action Head | NVIDIA GR00T N1.5 | Flow Matching 动作头 |

### 9.2 损失与优化器

| 损失 | 公式 | 权重 |
|---|---|---|
| `action_loss` | MSE(pred_velocity, velocity) | × 1.0 |
| `wm_loss` | L1(pred_states, gt_states) | × 0.1 |

| 参数组 | 学习率 | 覆盖 |
|---|---|---|
| `base` | 3e-5 | Qwen3-VL backbone |
| `qwen_vl_interface` | 1e-5 | VLM 接口层 |
| `action_model` | 1e-4 | DiT 动作头 |

AdamW (betas=[0.9,0.95]) + Cosine with min_lr (warmup 5000, min_lr 1e-6)、
bf16 混合精度、grad clip 1.0、gradient checkpointing 开启。

### 9.3 动作头推理：flow matching，不是 DDIM

训练时对动作加噪学习速度场；**推理时从 `torch.randn` 起始做欧拉积分**，
步数由 `framework.action_model.num_inference_timesteps` 控制（本项目为 **4**）。

> ⚠️ 代码里**没有 DDIM 分支**。有些旧文档/脚本里写"DDIM 4 步"是不准确的表述；
> 开环测试脚本里曾经的 `--ddim_steps` 参数也从未生效，
> 详见 [`03_openloop_testing.md` §8.2](./03_openloop_testing.md#82---ddim_steps-是历史遗留的无效参数-)。

### 9.4 核心文件索引

| 功能 | 文件 |
|---|---|
| 模型定义 | `starVLA/model/framework/VLA_JEPA.py` |
| 动作头 | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |
| 世界模型 Predictor | `starVLA/model/modules/world_model/vj2_predictor.py` |
| 数据加载入口 | `starVLA/dataloader/__init__.py` |
| 机器人配置注册 | `starVLA/dataloader/gr00t_lerobot/data_config.py` |
| 数据混合注册 | `starVLA/dataloader/gr00t_lerobot/mixtures.py` |
| 训练入口 | `starVLA/training/train_starvla.py` |

---

## 10. 命令速查

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# ---- 冒烟测试（1 GPU，2 分钟）----
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml

# ---- 正式训练（4 GPU）----
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml

# ---- 正式训练（8 GPU）----
bash scripts/vlajepa_robot_ft.sh

# ---- 后台跑 + 记日志 ----
nohup bash -c 'CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml' > log/run.log 2>&1 &

# ---- 停止训练（不要用 kill $!）----
pkill -f "train_starvla"

# ---- 查看进度 ----
tail -f log/run.log
```

**下一步** → 训练完成后做开环测试，见 [`03_openloop_testing.md`](./03_openloop_testing.md)。

---

## 相关文档

| 文档 | 说明 |
|---|---|
| [`01_data_conversion.md`](./01_data_conversion.md) | 训练前的数据转换与机器人注册 |
| [`03_openloop_testing.md`](./03_openloop_testing.md) | 训练完成后的开环测试 |
| [`archive/training_pipeline.md`](./archive/training_pipeline.md) | 训练流程与模型架构完整详解 |
| [`archive/training_phase2.md`](./archive/training_phase2.md) | 第二阶段训练完整教程（含逐项配置注释、FAQ） |
| [`archive/debug_journal.md`](./archive/debug_journal.md) | Piper 适配调试记录（含所有原始报错与解决过程） |
