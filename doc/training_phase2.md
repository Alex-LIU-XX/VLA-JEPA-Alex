# VLA-JEPA 第二阶段训练教程（Robot Fine-tuning）

## 概述

第二阶段（Robot Fine-tuning）只使用机器人操控数据（LeRobot v2.1 格式），对 VLA-JEPA 模型进行微调。本阶段不需要人类视频数据，可直接从 Qwen3-VL + V-JEPA2 的原始预训练权重开始训练，也可以加载第一阶段（Co-training）的 checkpoint 作为初始化。

**入口脚本**：`starVLA/training/train_starvla.py`  
**Shell 脚本**：`scripts/vlajepa_robot_ft.sh`  
**参考配置**：`scripts/config/vlajepa_robot_ft.yaml`

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型权重大小与下载](#2-模型权重大小与下载)
3. [数据集准备](#3-数据集准备)
4. [注册自定义数据集](#4-注册自定义数据集)
5. [配置文件详解](#5-配置文件详解)
6. [启动训练](#6-启动训练)
7. [断点续训](#7-断点续训)
8. [加载第一阶段权重](#8-加载第一阶段权重)
9. [常见问题](#9-常见问题)

---

## 1. 环境准备

### 1.1 创建 conda 环境

```bash
git clone https://github.com/ginwind/VLA-JEPA
cd VLA-JEPA

conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA
```

### 1.2 安装依赖

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

`requirements.txt` 包含 29 个包，关键依赖：

| 包 | 版本 | 用途 |
|---|---|---|
| `transformers` | ==4.57.0 | Qwen3-VL 模型加载 |
| `accelerate` | ==1.5.2 | 分布式训练 |
| `deepspeed` | ==0.16.9 | ZeRO 优化 |
| `decord` | ==0.6.0 | 视频解码 |
| `av` | ==12.3.0 | 视频解码（pyav 后端） |
| `diffusers` | 最新 | DiT 组件 |
| `pyarrow` / `fastparquet` | - | parquet 数据读取 |
| `qwen-vl-utils` | - | QwenVL 工具函数 |

> **注意**：`flash-attn` 需要单独编译安装（`--no-build-isolation`）。如果 GPU 较老不支持 FlashAttention 2，可在 YAML 中将 `attn_implementation` 改为 `"eager"`。

### 1.3 验证安装

```bash
python -c "import torch; import transformers; import accelerate; import deepspeed; print('All dependencies OK')"
```

---

## 2. 模型权重大小与下载

训练需要准备两个（或三个）预训练权重：

### 2.1 必需的权重

| 权重 | 大小 | 下载地址 |
|---|---|---|
| **Qwen3-VL-2B-Instruct** | ~4 GB | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct |
| **V-JEPA2 VIT-L** (vjepa2-vitl-fpc64-256) | ~1.5 GB | https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 |

**合计约 5.5 GB**，这是启动第二阶段训练的最低要求。

### 2.2 可选的权重

| 权重 | 大小 | 下载地址 | 用途 |
|---|---|---|---|
| **Co-training checkpoint** | 6.16 GB | https://huggingface.co/ginwind/VLA-JEPA (`Pretrain/`) | 作为微调初始化，可提升性能 |

### 2.3 已发布的完整模型列表

所有权重托管在 Hugging Face：https://huggingface.co/ginwind/VLA-JEPA

| 目录 | 大小 | 说明 |
|---|---|---|
| `Pretrain/` | 6.16 GB | 第一阶段 Co-training 权重 |
| `LIBERO/` | 6.16 GB | LIBERO 微调权重 |
| `SimplerEnv/` | 6.16 GB | SimplerEnv 微调权重 |
| `Real-world/` | 6.16 GB | 真实场景微调权重 |

> 每个 checkpoint 约 6.16 GB，包含完整的模型参数和训练配置。

### 2.4 配置 YAML 中的权重路径

下载后在 YAML 中设置：

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: /path/to/vjepa2-vitl-fpc64-256
```

---

## 3. 数据集准备

### 1.1 数据格式

本阶段使用 **LeRobot v2.1** 格式。数据目录结构如下：

```
<data_root>/
├── meta/
│   ├── modality.json       # [必需] 列映射文件
│   ├── episodes.jsonl      # [必需] episode 元数据
│   ├── tasks.jsonl         # [必需] 任务描述
│   └── info.json           # [必需] 数据集信息
├── data/
│   └── chunk_0/
│       └── episode_0.parquet  # state / action / timestamp 等低维数据
└── videos/
    └── chunk_0/
        └── episode_0_<camera>.mp4  # 视频文件
```

### 3.2 modality.json 编写

`modality.json` 定义了 parquet 文件中的列与 VLA-JEPA 内部字段之间的映射关系。

**参考模板**（对应 7 维 delta_qpos 动作 + 8 维状态 + 双视角视频）：

```json
{
    "state": {
        "x": { "start": 0, "end": 1 },
        "y": { "start": 1, "end": 2 },
        "z": { "start": 2, "end": 3 },
        "roll": { "start": 3, "end": 4 },
        "pitch": { "start": 4, "end": 5 },
        "yaw": { "start": 5, "end": 6 },
        "pad": { "start": 6, "end": 7 },
        "gripper": { "start": 7, "end": 8 }
    },
    "action": {
        "x": { "start": 0, "end": 1 },
        "y": { "start": 1, "end": 2 },
        "z": { "start": 2, "end": 3 },
        "roll": { "start": 3, "end": 4 },
        "pitch": { "start": 4, "end": 5 },
        "yaw": { "start": 5, "end": 6 },
        "gripper": { "start": 6, "end": 7 }
    },
    "video": {
        "primary_image": { "original_key": "observation.images.image" },
        "wrist_image": { "original_key": "observation.images.wrist_image" }
    },
    "annotation": {
        "human.action.task_description": { "original_key": "task_index" }
    }
}
```

**字段说明**：
- `state` / `action`：每个子键的 `start`/`end` 是 parquet 中对应列（numpy 数组）的切片索引
- `video`：`original_key` 是 parquet 中视频路径列的列名（如 `observation.images.image`）
- `annotation`：语言指令的来源列，`task_index` 表示从 `tasks.jsonl` 中查表获取文本

### 3.3 预置数据集

项目支持的已知数据集（可直接用于训练，无需额外注册）：

| 数据集 | data_mix 名称 | 机器人类型 | 参考 modality.json |
|---|---|---|---|
| LIBERO | `libero_all` | `libero_franka` | `examples/LIBERO/modality.json` |
| Droid | `droid` | `libero_franka` | `examples/Droid/modality.json` |
| BridgeV2 | `bridge` | `oxe_bridge` | `examples/SimplerEnv/modality.json` |
| Fractal | `bridge_rt_1` | `oxe_rt1` | `examples/SimplerEnv/modality.json` |
| FR3 真实数据 | `fr3_realworld` | `fr3_real_world` | 自定义 |

---

## 4. 注册自定义数据集

如果你的机械臂不是上述预置类型，需要注册自定义配置。

### 4.1 添加 robot config 类

打开 `starVLA/dataloader/gr00t_lerobot/data_config.py`，添加一个 Config 类：

```python
class MyRobotConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    def __init__(self, observation_indices, action_indices):
        self.observation_indices = observation_indices
        self.action_indices = action_indices

    def modality_config(self):
        from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    def transform(self):
        from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
            StateActionToTensor,
            StateActionTransform,
        )
        transforms = [
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
```

然后在同文件末尾的 `ROBOT_TYPE_CONFIG_MAP` 中注册：

```python
ROBOT_TYPE_CONFIG_MAP = {
    # ... 已有的 ...
    "my_robot": MyRobotConfig,
}
```

> **注意**：`video_keys` / `state_keys` / `action_keys` 中的字段名（如 `"video.primary_image"`）必须与 `modality.json` 中定义的键名完全一致。

### 4.2 添加数据混合配置

打开 `starVLA/dataloader/gr00t_lerobot/mixtures.py`，添加一个新的条目：

```python
DATASET_NAMED_MIXTURES = {
    # ... 已有的 ...
    "my_data": [
        ("", 1.0, "my_robot"),
    ],
}
```

每个元组的结构：`(data_name, sampling_weight, robot_type)`
- **data_name**：数据子目录名。如果你的数据直接放在 `data_root_dir/` 下（即 `data_root_dir/meta/` 存在），写 `""`；如果放在 `data_root_dir/my_subdir/` 下，写 `"my_subdir"`
- **sampling_weight**：多个数据集混合时的采样权重
- **robot_type`**：上一步在 `ROBOT_TYPE_CONFIG_MAP` 中注册的 key 名

---

## 5. 配置文件详解

复制 `scripts/config/vlajepa_robot_ft.yaml` 并根据你的数据修改。完整配置说明如下：

```yaml
# 实验标识
run_id: my_robot_ft
run_root_dir: checkpoints
seed: 42
trackers:
  - json
is_debug: false

# =========== 模型框架 ===========
framework:
  name: VLA_JEPA
  qwenvl:
    base_vlm: /path/to/Qwen3-VL-2B-Instruct   # [必须] Qwen3-VL 模型路径
    attn_implementation: flash_attention_2
    vl_hidden_dim: 2048
  action_model:
    action_model_type: DiT-B
    action_dim: 7                               # [必须] 动作维度
    state_dim: 8                                # [必须] 状态维度
    future_action_window_size: 6                # 预测未来 N 步动作
    action_horizon: 7                           # chunk = past + 1 + future
    past_action_window_size: 0
    repeated_diffusion_steps: 8                 # 扩散 loss 重复次数
    noise_beta_alpha: 1.5                       # Beta 分布参数 α
    noise_beta_beta: 1.0                        # Beta 分布参数 β
    num_inference_timesteps: 4                  # 推理时 DDIM 步数
    num_target_vision_tokens: 32
    diffusion_model_cfg:
      cross_attention_dim: 2048
      dropout: 0.2
      final_dropout: true
      interleave_self_attention: true
      norm_type: ada_norm
      num_layers: 16
      output_dim: 1024
  vj2_model:
    base_encoder: /path/to/vjepa2-vitl-fpc64-256  # [必须] V-JEPA2 编码器路径
    depth: 12
    num_heads: 8
    special_action_token: "<|action_{}|>"
    num_action_tokens_per_timestep: 8
    embodied_action_token: "<|embodied_action|>"
    num_embodied_action_tokens_per_instruction: 32
    num_frames: 8                                 # 视频帧数
  reduce_in_full_precision: true

# =========== 数据集 ===========
datasets:
  vla_data:
    dataset_py: lerobot_datasets                 # 使用 LeRobot v2.1 格式
    data_root_dir: /path/to/your/dataset          # [必须] 数据根目录
    data_mix: my_data                             # [必须] mixtures.py 中注册的名字
    action_type: delta_qpos                       # delta_qpos 或 absolute
    CoT_prompt: "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."
    resolution_size: 224                          # VLM 输入图像分辨率
    per_device_batch_size: 32                     # 每 GPU batch size
    video_resolution_size: 256                    # 世界模型视频分辨率
    load_all_data_for_training: true
    with_state: true                              # 是否使用 proprioceptive state

# =========== 训练参数 ===========
trainer:
  epochs: 100
  max_train_steps: 30000                          # 总训练步数
  num_warmup_steps: 5000                          # LR warmup 步数
  save_interval: 10000                            # 保存间隔
  eval_interval: 100                              # 评估间隔
  learning_rate:
    base: 3.0e-05                                 # VLM backbone LR
    qwen_vl_interface: 1.0e-05                   # VLM 接口层 LR
    action_model: 1.0e-04                         # 动作头 LR
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: ''                              # 冻结模块（逗号分隔：qwen_vl, action_model, vj_encoder, vj_predictor）
  loss_scale:
    vla: 1.0
    vlm: 0.1
  max_grad_norm: 1.0
  gradient_clipping: 1.0
  gradient_accumulation_steps: 1
  logging_frequency: 10

  #pretrained_checkpoint: /path/to/checkpoint     # [可选] 加载预训练权重

  optimizer:
    name: AdamW
    betas: [0.9, 0.95]
    eps: 1.0e-08
    weight_decay: 1.0e-08

  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true

  # 断点续训参数
  is_resume: false
  resume_epoch: null
  resume_step: null
```

### 关键配置项说明

| 配置项 | 说明 | 推荐值 |
|---|---|---|
| `action_dim` | 动作空间维度，与你的数据一致 | 7 |
| `state_dim` | 状态空间维度，与你的数据一致 | 8 |
| `with_state` | 是否使用 proprioceptive state 作为动作头的输入 | `true`（推荐）|
| `action_type` | `delta_qpos`（增量）或 `absolute`（绝对位置） | `delta_qpos` |
| `freeze_modules` | 逗号分隔的模块列表，可冻结部分模块加速训练 | 可冻结 `qwen_vl` 或 `action_model` |
| `pretrained_checkpoint` | 第一阶段 Co-training 的权重路径（可选） | 如有则设置 |

**冻结模块的可选值**：
- `qwen_vl` — 冻结 VLM backbone（Qwen3-VL）
- `action_model` — 冻结动作头
- `vj_encoder` — 冻结 V-JEPA 编码器（训练中默认已冻结）
- `vj_predictor` — 冻结世界模型 predictor

---

## 6. 启动训练

### 4.1 直接使用脚本

编辑 `scripts/vlajepa_robot_ft.sh`，修改 `--config_yaml` 指向你的配置：

```bash
# scripts/vlajepa_robot_ft.sh

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft.yaml
```

然后运行：

```bash
bash scripts/vlajepa_robot_ft.sh
```

### 4.2 自定义命令行参数

也可以直接调用 `train_starvla.py`，用 `--config_yaml` 指定任意配置文件：

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml /path/to/your_config.yaml
```

### 4.3 单 GPU 调试

```bash
python ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft.yaml
```

> 注意：单 GPU 运行时需要去掉 `accelerate launch`，且 batch size 可能需要调小（如 `per_device_batch_size: 4`）。

### 4.4 训练输出

训练过程中的输出保存在 `checkpoints/<run_id>/` 目录下：

```
checkpoints/robot_ft/
├── config.yaml                  # 使用的配置
├── config.json                  # JSON 格式配置
├── steps_10000_pytorch_model.pt # 第 10000 步 checkpoint
├── steps_20000_pytorch_model.pt # 第 20000 步 checkpoint
├── steps_30000_pytorch_model.pt # 第 30000 步 checkpoint
├── dataset_statistics.json      # 数据集统计信息
├── summary.jsonl                # 训练日志
└── final_model/                 # 最终模型
    └── pytorch_model.pt
```

---

## 7. 断点续训

在 YAML 配置中设置：

```yaml
trainer:
  is_resume: true
  resume_epoch: null        # 或指定 epoch 号
  resume_step: 10000        # 指定从哪个 step 恢复
```

如果 `resume_step` 为 `null`，框架会自动从 `checkpoints/<run_id>/` 目录下最近的 checkpoint 恢复。

恢复时需要将 `run_id` 与之前训练时的值保持一致，以便找到 checkpoint 目录。

---

## 8. 加载第一阶段权重（可选）

第一阶段 Co-training 权重对于第二阶段是 **可选的**，不加载也不影响训练启动。加载关系如下：

```
Qwen3-VL (HF 官方)    ← 必需的，不加载 Co-training 时直接使用
V-JEPA2 (HF 官方)     ← 必需的，不加载 Co-training 时直接使用
       │
       ▼（可选）
Co-training checkpoint  ← 不加载时，框架会自动跳过，直接使用上述两个官方权重
       │
       ▼
Robot Fine-tuning       ← 第二阶段
```

如果要使用 Co-training 权重，在 YAML 中取消注释：

```yaml
trainer:
  pretrained_checkpoint: /path/to/Pretrain/checkpoints/steps_50000_pytorch_model.pt
```

框架会在 `prepare_training()` 阶段加载该权重，并跳过 backbone（Qwen3-VL、V-JEPA2）的原始预训练权重加载。

> 有无 Co-training 权重的区别：加载后相当于在更多数据上预热过，微调效果可能更好；跳过则直接从原始 backbone 开始训练，仍然有效。

权重可以从 Hugging Face 下载：https://huggingface.co/ginwind/VLA-JEPA （`Pretrain/` 目录）。

---

## 9. 常见问题

### Q1: 训练时出现 "No matching modality keys" 错误

**原因**：`data_config.py` 中定义的 `video_keys` / `state_keys` / `action_keys` 与 `modality.json` 中的键名不匹配。

**解决**：确保两者完全一致。例如 `modality.json` 中 video 的 key 是 `"primary_image"`，那么 data_config 中应该是 `"video.primary_image"`。

### Q2: 视频加载失败（decord / torchcodec / pyav）

**原因**：视频后端选择不当。

**解决**：在 `lerobot_datasets.py:49` 中修改 `video_backend` 参数，可选项：`"decord"`、`"torchcodec"`、`"opencv"`、`"torchvision_av"`、`"pyav"`。

### Q3: 如何调整 GPU 数量？

**解决**：修改 shell 脚本中的 `--num_processes` 参数，并相应调整 `per_device_batch_size` 使全局 batch size 不变。

```bash
accelerate launch --num_processes 4 ...  # 4 GPU
```

### Q4: GPU 内存不足（OOM）

**解决**：
- 调小 `per_device_batch_size`
- 开启 `enable_gradient_checkpointing: true`（默认已开启）
- 减少 `num_frames`（从 8 改为 4）
- 使用 ZeRO-3 替代 ZeRO-2（修改 `deepspeed_zero2.yaml`）

### Q5: 加载 checkpoint 时 key 不匹配

**原因**：模型结构或配置与 checkpoint 不完全一致（如 `action_dim` 不同）。

**解决**：在加载代码中添加 `strict=False`（需要修改 `train_starvla.py` 中的加载逻辑），或确保配置一致。

---

## 参考文件

| 文件 | 说明 |
|---|---|
| `scripts/vlajepa_robot_ft.sh` | 训练启动脚本 |
| `scripts/config/vlajepa_robot_ft.yaml` | 训练配置模板 |
| `starVLA/training/train_starvla.py` | 训练入口代码 |
| `starVLA/dataloader/lerobot_datasets.py` | 数据加载入口 |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | LeRobot v2.1 数据集实现 |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | 机器人配置注册 |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 数据混合注册 |
| `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py` | 机器人类型标签 |
| `examples/LIBERO/modality.json` | LIBERO 数据格式示例 |
| `examples/Droid/modality.json` | Droid 数据格式示例 |
