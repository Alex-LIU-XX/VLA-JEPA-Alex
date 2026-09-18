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
   - [5.7 ICLR_real_world 八个任务：逐个训练](#57-iclr_real_world-八个任务逐个训练)
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

### 2.1 uv 环境（本仓库用 `uv` 管理，不用 conda）

```bash
# 首次搭建 / 换机器重建：完整步骤见 doc/04_environment_uv.md
uv venv --python 3.10 .venv
uv sync --python .venv/bin/python     # 依赖规格在 pyproject.toml（torch 走 cu128 索引）
```

本机已有环境：**仓库内 `.venv`**（Python 3.10.21，uv 管理，`torch 2.7.1+cu128`），
下面命令统一用它。完整环境说明（代理、CUDA 组合、FlashAttention、权重目录）见
[04_environment_uv.md](./04_environment_uv.md)。

> ⚠️ **不要直接敲 `accelerate` / `python`**。这两个名字解析到哪个解释器取决于当前 shell 的
> PATH（登录 shell 可能是 conda base 或系统 python，那里**没有 accelerate / deepspeed**），
> 会直接报：
>
> ```
> bash: line 2: accelerate: command not found
> ```
>
> 两个正确姿势（本文所有命令统一用 ①）：
>
> ```bash
> # ① 直接用仓库内 venv 的可执行文件，不依赖 shell 处于哪个环境（推荐）
> .venv/bin/accelerate launch ...
>
> # ② 或先激活 venv——写进脚本时同样有效
> source .venv/bin/activate
> ```
>
> 自查：`ls .venv/bin/accelerate` 存在、`readlink -f $(which accelerate)` 指向 `.venv/bin/accelerate`。

关键依赖版本：

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` / `torchvision` | 2.7.1+cu128 / 0.22.1+cu128 | 训练与推理（cu128 见 §04 说明） |
| `transformers` | 4.57.0 | Qwen3-VL 加载 |
| `accelerate` | 1.5.2 | 分布式启动 |
| `deepspeed` | 0.16.9 | ZeRO 优化 |
| `flash-attn` | 2.8.3.post1 | `attn_implementation=flash_attention_2` |
| `av` | 12.3.0 | 视频解码（AV1 必需） |
| `ninja` | — | **编译 DeepSpeed CPU Adam 算子，必须装** |

### 2.2 验证

```bash
.venv/bin/python -c \
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

> 📌 **本节所有命令都带下面两个前缀，照抄即可**（后文各节不再重复解释）：
>
> | 前缀 | 为什么必须有 |
> |---|---|
> | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 24 GB 卡上显存**贴着天花板**，不开这个会因**碎片化**在 step 10 左右 OOM——报错典型特征是「只差 20 MB，但缓存里还空着 2 GB」。详见 §8.2 |
> | `.venv/bin/accelerate` | 不要依赖 PATH 里的 `accelerate`（可能是 conda base / 系统 python，里面没有）。详见 §2.1 |
>
> 完整模板（把 `<...>` 换掉即可）：
>
> ```bash
> cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
> CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
>   --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
>   --num_processes 4 --main_process_port 29500 \
>   ./starVLA/training/train_starvla.py \
>   --config_yaml ./scripts/config/<你的配置>.yaml \
>   > log/<run_id>.log 2>&1
> ```

### 5.1 冒烟测试（1 GPU，约 2 分钟）

用于验证「数据 + 模型 + 保存」全链路能否跑通。**正式训练前务必先跑这个。**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch \
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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch \
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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
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

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft.yaml
```

> 换成自己的配置时，把最后一行 `--config_yaml` 指向你的 YAML 即可。

**后台运行 + 记录日志**：

```bash
nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
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
| 4 | 4 | 1 | 16 | **本机基准（实测 5012 步零 OOM）** |
| 4 | 8 | 1 | 32 | ❌ 24 GB 卡放不下 |
| 8 | 8 | 1 | 64 | 仅 40 GB+ 卡 |

> **本机是 4×RTX 4090，可用显存 23.52 GiB/卡。**
> 2.77B 模型 + ZeRO-2 的稳态占用是 `MA 7.74 GB / CA 10.42~10.47 GB`
> （三次训练日志的读数一致），加上激活值和 allreduce bucket 后**刚好贴在 24 GB 天花板**。
>
> 因此 **`per_device_batch_size: 4`（4 卡 → 总 batch 16）就是这台机器的上限**，别再往上调。
> 实测 `checkpoints/adjust_cup_10k`（4 卡 / bs 4 / accum 1 / 计划 10000 步）连续跑满
> **5012 步零 OOM**（后因故中断，非 OOM），是本项目唯一验证过跑得动的配方——
> **要对齐它，照抄这一行即可，不要动 batch size**。
>
> 该配方在换节点后会偶发 OOM：23:36 与 23:43 两次启动都在 **step 10~11** 崩，
> 报错都是「想申请 938 MiB、只剩 917 MiB，但**缓存里还空着 2.1 GiB**」——
> 典型碎片化，**加 §5 开头的 `expandable_segments` 即可**，不必动 batch size。

**OOM 时的调整顺序**：① 先确认启动命令带了
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（§5 开头，**通常只加这一个变量就够了**）
→ ② 调小 `per_device_batch_size` → ③ 增大 `gradient_accumulation_steps`（保持等效 batch）→
④ `num_frames` 从 8 改 4 → ⑤ 开 `freeze_modules` → ⑥ 最后才上 ZeRO-2 + CPU Offload（§8.2）。

### 5.6 阶段一 Co-training（本项目未使用）

```bash
bash scripts/vlajepa_cotrain.sh
# 等价于：
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_vlajepa_cotrain.py \
  --config_yaml ./scripts/config/vlajepa_cotrain.yaml
```

---

### 5.7 ICLR_real_world 八个任务：逐个训练

`Datasets/ICLR_real_world/` 下的 8 个数据集**各训一个模型**，
每个 **20000 步**，每 **5000 步**存一个权重（共 4 个：5000 / 10000 / 15000 / 20000）。

#### 5.7.1 数据集与配置文件对照

配置文件已生成在 `scripts/config/`，无需再改：

| # | 数据集（`_v2_1`） | episodes | frames | 配置文件 | `run_id` / 输出目录 |
|---|---|---|---|---|---|
| 1 | `adjust_cup_0409_1_offset_state` | 50 | 6837 | `iclr_adjust_cup.yaml` | `checkpoints/iclr_adjust_cup/` |
| 2 | `open_cabinet_all_0423_1_offset_state` | 50 | 6275 | `iclr_open_cabinet.yaml` | `checkpoints/iclr_open_cabinet/` |
| 3 | `pick_banana_100_newTable_1_offset_state` | 100 | 12209 | `iclr_pick_banana_newtable.yaml` | `checkpoints/iclr_pick_banana_newtable/` |
| 4 | `pick_banana_pot_0730_1_offset_state` | 48 | 12224 | `iclr_pick_banana_pot.yaml` | `checkpoints/iclr_pick_banana_pot/` |
| 5 | `pick_block_100_1_offset_state` | 100 | 18572 | `iclr_pick_block.yaml` | `checkpoints/iclr_pick_block/` |
| 6 | `pick_eggplant_drawer_0730_1_offset_state` | 50 | 6809 | `iclr_pick_eggplant_drawer.yaml` | `checkpoints/iclr_pick_eggplant_drawer/` |
| 7 | `pick_eggplant_from_cluttered_0414_1_offset_state` | 87 | 12884 | `iclr_pick_eggplant_cluttered.yaml` | `checkpoints/iclr_pick_eggplant_cluttered/` |
| 8 | `sponge_wipe_0423_1_offset_state` | 49 | 5619 | `iclr_sponge_wipe.yaml` | `checkpoints/iclr_sponge_wipe/` |

每份配置的关键项（8 份完全一致，只有 `run_id` 和 `data_root_dir` 不同）：

```yaml
run_id: iclr_<短名>
datasets:
  vla_data:
    data_root_dir: /share/.../Datasets/ICLR_real_world/<数据集名>_v2_1
    data_mix: piper_pick_place          # 该 mix 指向 data_root_dir 本身，逐个训练无需改 mixtures.py
    per_device_batch_size: 4
trainer:
  max_train_steps: 20000
  save_interval: 5000                   # → steps_{5000,10000,15000,20000}_pytorch_model.pt
  num_warmup_steps: 1000
  eval_interval: 500
```

#### 5.7.2 通用命令模板

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA
mkdir -p log

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29500 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/<配置文件> \
  > log/<run_id>.log 2>&1
```

> - 上面按 **4 卡**写（`CUDA_VISIBLE_DEVICES=0,1,2,3` + `--num_processes 4`）。
>   卡数不同时这两处要**同步修改**，保持一致
> - 单卡跑不了这个配置（2.77B 模型 + Adam 状态放不下 24 GB），必须 ≥2 卡；
>   单卡要加 CPU Offload，见 §8.2
> - **`per_device_batch_size` 保持 4，不要改**：本机 4×RTX 4090（23.52 GiB/卡）上
>   bs 4 已经是上限，调到 8 必 OOM，见 §5.5
> - 命令里那两个前缀（`PYTORCH_CUDA_ALLOC_CONF` + 绝对路径 `accelerate`）**不能省**，
>   原因见 §5 开头

#### 5.7.3 八个任务的启动命令

**① adjust_cup_0409**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29500 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_adjust_cup.yaml \
  > log/iclr_adjust_cup.log 2>&1
```

**② open_cabinet_all_0423**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29501 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_open_cabinet.yaml \
  > log/iclr_open_cabinet.log 2>&1
```

**③ pick_banana_100_newTable**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29502 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_pick_banana_newtable.yaml \
  > log/iclr_pick_banana_newtable.log 2>&1
```

**④ pick_banana_pot_0730**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29503 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_pick_banana_pot.yaml \
  > log/iclr_pick_banana_pot.log 2>&1
```

**⑤ pick_block_100**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_pick_block.yaml \
  > log/iclr_pick_block.log 2>&1
```

**⑥ pick_eggplant_drawer_0730**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_pick_eggplant_drawer.yaml \
  > log/iclr_pick_eggplant_drawer.log 2>&1
```

**⑦ pick_eggplant_from_cluttered_0414**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29506 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_pick_eggplant_cluttered.yaml \
  > log/iclr_pick_eggplant_cluttered.log 2>&1
```

**⑧ sponge_wipe_0423**

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29507 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/iclr_sponge_wipe.yaml \
  > log/iclr_sponge_wipe.log 2>&1
```

> 每个任务用**不同的 `--main_process_port`**（29500~29507），
> 这样万一你想同时跑两个也不会端口冲突。

#### 5.7.4 一条命令跑完全部 8 个（串行）

训练脚本之间互相独立，直接循环即可：

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && mkdir -p log

for name in adjust_cup open_cabinet pick_banana_newtable pick_banana_pot \
            pick_block pick_eggplant_drawer pick_eggplant_cluttered sponge_wipe
do
  echo "===== [$(date '+%F %T')] START iclr_${name} ====="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
    --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 4 --main_process_port 29500 \
    ./starVLA/training/train_starvla.py \
    --config_yaml ./scripts/config/iclr_${name}.yaml \
    > log/iclr_${name}.log 2>&1
  echo "===== [$(date '+%F %T')] DONE  iclr_${name} (exit $?) ====="
done
```

建议放到 tmux / screen 里跑，避免 SSH 断连中断训练：

```bash
tmux new -s iclr_train
# 粘贴上面的循环，然后 Ctrl+B 再按 D 脱离
tmux attach -t iclr_train          # 重新连回查看
```

#### 5.7.5 时间预算

每个任务固定 **4 卡**，参考实测数据（4 卡、`per_device_batch_size: 4`）：**约 3.6 s/step**

```
单模型 20000 步 = 20000 × 3.6 s ≈ 20 小时
```

所以 8 个模型的总墙钟时间**取决于机器上有几组 4 卡**：

| 机器规模 | 并行方式 | 总墙钟时间 |
|---|---|---|
| 4 卡 | 只能串行，一次跑 1 个 | **~6.7 天** |
| 8 卡 | 2 个任务并行（各 4 卡） | **~3.4 天** |
| 16 卡 | 4 个任务并行（各 4 卡） | **~1.7 天** |
| 32 卡 | 8 个任务全并行 | **~20 小时** |

> ⚠️ **4 卡串行是 6.7 天的任务，提前规划好。**
> 如果机器卡多，最划算的是**每个任务分一组 4 卡并行跑**——
> 比如 32 卡时可以 8 个任务同时开，一晚上就跑完。
>
> 多任务并行时记得给每个任务**分配不同的 `CUDA_VISIBLE_DEVICES` 和 `--main_process_port`**，
> 且 `--num_processes` 始终是 4：

```bash
# 例：16 卡机器，4 个任务并行（每个 4 卡）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3   .venv/bin/accelerate launch --num_processes 4 --main_process_port 29500 \
  ... --config_yaml ./scripts/config/iclr_adjust_cup.yaml            > log/iclr_adjust_cup.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=4,5,6,7   .venv/bin/accelerate launch --num_processes 4 --main_process_port 29501 \
  ... --config_yaml ./scripts/config/iclr_open_cabinet.yaml          > log/iclr_open_cabinet.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=8,9,10,11 .venv/bin/accelerate launch --num_processes 4 --main_process_port 29502 \
  ... --config_yaml ./scripts/config/iclr_pick_banana_newtable.yaml  > log/iclr_pick_banana_newtable.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=12,13,14,15 .venv/bin/accelerate launch --num_processes 4 --main_process_port 29503 \
  ... --config_yaml ./scripts/config/iclr_pick_banana_pot.yaml       > log/iclr_pick_banana_pot.log 2>&1 &
wait
```

#### 5.7.6 训练过程中怎么盯

```bash
# 实时看日志
tail -f log/iclr_pick_block.log

# 只看进度条（带 s/it 和预计剩余时间）
tail -c 400 log/iclr_pick_block.log | tr '\r' '\n' | tail -2

# 看已保存的权重
ls -la checkpoints/iclr_pick_block/checkpoints/

# 看评估曲线
tensorboard --logdir checkpoints/iclr_pick_block/tensorboard
```

#### 5.7.7 训练完成后

每个任务的产出：

```
checkpoints/iclr_<短名>/
├── config.yaml
├── dataset_statistics.json
├── tensorboard/
├── checkpoints/
│   ├── steps_5000_pytorch_model.pt      # 每个 6.16 GB
│   ├── steps_10000_pytorch_model.pt
│   ├── steps_15000_pytorch_model.pt
│   └── steps_20000_pytorch_model.pt
└── final_model/
    └── pytorch_model.pt
```

> ⚠️ **磁盘**：每个 run 约 **37 GB**（6 个权重 × 6.16 GB），8 个任务合计约 **300 GB**，提前确认空间。

开环测试（每个任务用自己 run 目录下的 `config.yaml`）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_openloop.py \
  --config_yaml checkpoints/iclr_pick_block/config.yaml \
  --checkpoint  checkpoints/iclr_pick_block/final_model/pytorch_model.pt \
  --output_dir  eval_openloop/iclr_pick_block \
  --windows_per_episode 8 --batch_size 4 --seed 0
```

详见 [`03_openloop_testing.md`](./03_openloop_testing.md)。

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
.venv/bin/python -c "
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
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --num_processes 1 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/piper_test.yaml
```

### 8.2 CUDA Out of Memory

先看报错里的两个数字，**决定走哪条路**：

| 症状 | 报错特征 | 解决 |
|---|---|---|
| **A. 碎片化**（24 GB 卡上最常见） | 想申请的块只差一点点，且 **`reserved by PyTorch but unallocated` 还有 1~2 GiB** | 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **B. 真放不下** | `Tried to allocate 7.93 GiB` 这种，差的是几 GB 量级 | ZeRO-2 + CPU Offload |

#### A. 碎片化 —— 先试这个，不用改任何超参

真实报错长这样（`iclr_open_cabinet`，4 卡 / bs 4，跑到第 11 步崩）：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 938.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 917.62 MiB is free.
Process 3593698 has 22.61 GiB memory in use.
Of the allocated memory 19.86 GiB is allocated by PyTorch,
and 2.15 GiB is reserved by PyTorch but unallocated.
```

**注意 `2.15 GiB is reserved by PyTorch but unallocated`** —— PyTorch 手里还攥着
2.1 GiB 空闲缓存，却拿不出一块**连续**的 938 MiB。这是碎片化，不是容量不够。

**解决**：启动命令加上环境变量即可，**batch size 一个字都不用改**：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  ...
```

> 这套配置本身是跑得完的：`checkpoints/adjust_cup_10k` 用同样的
> 4 卡 / bs 4 / accum 1 跑满 **5012 步零 OOM**，而两次 OOM 的显存读数
> （`MA 7.74 GB` / `CA 10.42 GB`）跟它**几乎完全相同**——说明**不是 batch size 太大，别急着调小**。

#### B. 真的放不下

**原因**：2.77B 参数的模型，ZeRO-2 不分片权重和优化器状态时需求远超 24 GB：

| 项 | 占用 |
|---|---|
| 模型权重 (bf16) | ~5.5 GB |
| 梯度 (bf16) | ~5.5 GB |
| Adam 状态 (fp32) | ~22 GB（momentum + variance） |
| **合计** | **>> 24 GB** |

**解决**：
1. 用 **ZeRO-2 + CPU Offload**（把优化器状态挪到 CPU 内存）→ GPU 显存降到 **~5.7 GB**；
2. **`pip install ninja`** —— DeepSpeed 的 CPU Adam 算子需要编译 C++ 扩展，缺了会失败；
3. 或者直接用 4 卡以上（ZeRO-2 分片后可以关掉 CPU Offload，本机就是这么跑的，见 §5.5）；
   若仍 OOM，再降低 `per_device_batch_size` 并同步增大 `gradient_accumulation_steps`
   以保持等效 batch。

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

> 下面每条命令都带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 和
> `accelerate` 的绝对路径——**这两个前缀都不能省**（原因见 §5 开头、§8.2）。
> 要加自己的任务，把 `--config_yaml` 换成 `./scripts/config/iclr_<name>.yaml` 即可。

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

# ---- 冒烟测试（1 GPU，2 分钟）----
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml

# ---- 正式训练（4 GPU）----
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_train.yaml

# ---- 正式训练（8 GPU）----
bash scripts/vlajepa_robot_ft.sh

# ---- 后台跑 + 记日志 ----
nohup bash -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/accelerate launch \
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
