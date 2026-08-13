# VLA-JEPA 训练流程详解

## 1. 项目概述

VLA-JEPA 是一个增强型 VLA（Vision-Language-Action）模型，在标准的 VLA 框架（Qwen3-VL + DiT 动作头）基础上，引入 **V-JEPA2 世界模型** 进行 latent 空间视频预测，作为辅助监督信号。

### 模型架构

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
  │                       │                        │
  └───────────────────────┴────────────────────────┘
                              +
                           total_loss
```

### 核心组件

| 组件 | 来源 | 说明 |
|---|---|---|
| **Qwen3-VL-2B** | Qwen | VLM 骨干，编码图像+指令为 hidden states |
| **V-JEPA2 (vjepa2-vitl)** | Meta/FAIR | 视频编码器（训练时 frozen） |
| **V-JEPA Predictor** | 自定义 | Action-Conditioned Transformer，预测未来视频 patch embedding |
| **DiT-B Action Head** | NVIDIA GR00T N1.5 | Flow Matching 扩散策略，预测动作轨迹 |

---

## 2. 训练数据格式

### 2.1 机器人数据 — LeRobot v2.1 格式

**配置文件选择**：`datasets.vla_data.dataset_py: lerobot_datasets`

**目录结构**：

```
<data_root>/
├── meta/
│   ├── modality.json      # 列映射文件（state/action/video 的字段定义）
│   ├── episodes.jsonl     # episode 元数据（episode_index, length）
│   ├── tasks.jsonl        # 任务定义（task_index → 描述文本）
│   └── info.json          # 数据集信息（data_path, video_path, features）
├── data/
│   └── chunk_0/
│       └── episode_0.parquet  # 低维数据列（state, action, timestamp 等）
└── videos/
    └── chunk_0/
        └── episode_0_<camera>.mp4  # 视频文件
```

**modality.json 示例**（`examples/LIBERO/modality.json`）：

```json
{
    "state": {
        "x": { "start": 0, "end": 1 },
        "y": { "start": 1, "end": 2 },
        "gripper": { "start": 7, "end": 8 }
    },
    "action": {
        "x": { "start": 0, "end": 1 },
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

### 2.2 人类视频数据 — 视频文件夹 + CSV

**配置文件选择**：`datasets.video_data.dataset_py: video_datasets`

```
<video_dir>/
  ├── 0.webm
  ├── 1.webm
  └── ...
<text_file>  # 无表头 CSV，";" 分隔；第1列=索引，第2列=文本描述
```

### 2.3 VLM 数据 — LLaVA JSON/JSONL 格式

```json
{
    "images": "filename.jpg",
    "data_path": "/path/to/images/",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is in this image?"},
        {"from": "gpt", "value": "A cat sitting on a chair."}
    ]
}
```

### 2.4 可选：LeRobot v3 格式

配置 `dataset_py: lerobot_v3_datasets`，使用官方 `lerobot` pip 包加载数据。

---

## 3. 训练流程

### 3.1 两阶段策略

| 阶段 | 脚本 | 配置 | 入口文件 | 数据 |
|---|---|---|---|---|
| **Co-training** | `scripts/vlajepa_cotrain.sh` | `scripts/config/vlajepa_cotrain.yaml` | `train_vlajepa_cotrain.py` | VLA + 视频 |
| **Robot Fine-tuning** | `scripts/vlajepa_robot_ft.sh` | `scripts/config/vlajepa_robot_ft.yaml` | `train_starvla.py` | 仅 VLA |

两个阶段可独立运行。阶段二不需要阶段一的 checkpoint，可直接从 Qwen3-VL + V-JEPA2 原始预训练权重开始。

### 3.2 完整训练流程图

```
Config YAML
  │
  ├─ build_framework(cfg) ──→ VLA_JEPA 模型
  │     ├─ Qwen3-VL (base_vlm)
  │     ├─ 扩展 tokenizer（添加 action token）
  │     ├─ V-JEPA2 Encoder (frozen)
  │     ├─ V-JEPA Predictor
  │     └─ DiT Action Head (Flow Matching)
  │
  ├─ build_dataloader() × 2 (co-train) / × 1 (fine-tune)
  │     ├─ LeRobotMixtureDataset（采样自 data_mix 指定的多个数据集）
  │     └─ VideoFolderDataset（视频文件夹采样）
  │
  ├─ setup_optimizer_and_scheduler()
  │     ├─ 3 个参数组（base LR=3e-5, qwen LR=1e-5, action LR=1e-4）
  │     ├─ AdamW (betas=[0.9,0.95])
  │     └─ Cosine LR scheduler (warmup=5000, min_lr=1e-6)
  │
  ├─ prepare_training()
  │     ├─ set_seed()
  │     ├─ 可选：加载 pretrained_checkpoint
  │     ├─ 可选：冻结部分模块（freeze_modules）
  │     └─ accelerator.prepare()
  │
  └─ train() loop [while steps < max_train_steps]
        │
        ├── Co-training 每步有 2 个 phase：
        │
        │   Phase 1 — VLA Forward：
        │   │   1. QwenVL 编码图像+指令 → last_hidden
        │   │   2. 提取 action_tokens + embodied_action_tokens
        │   │   3. V-JEPA 编码视频（no_grad）→ predictor 预测未来 → wm_loss
        │   │   4. DiT Flow Matching → action_loss
        │   │   5. total_loss = action_loss + wm_loss × 0.1
        │   │   6. backward → optimizer.step()
        │   │
        │   Phase 2 — VLM Forward：
        │       1. 取纯视频 batch（无 action）
        │       2. QwenVL + V-JEPA → 仅计算 wm_loss
        │       3. backward → optimizer.step()
        │
        ├── Robot FT 只有 Phase 1（无视频数据）
        │
        ├── 每 eval_interval 步：评估 MAE/MSE
        └── 每 save_interval 步：保存 checkpoint
```

### 3.3 模型前向传播细节 (`VLA_JEPA.forward`)

```
输入 batch：
  image: [PIL.Image × 2]        # 双视角，resize 到 224
  video: [V, T, H, W, 3]        # V=2 视角, T=8 帧, resize 到 256
  lang: str                      # 语言指令
  action: [T-1, action_dim]      # 7 维动作（delta_qpos）
  state: [1, state_dim]          # 8 维状态

Step 1 - QwenVL 编码：
  qwen_inputs = process_images + tokenize(instruction)
  qwen_outputs = qwen_vl(**qwen_inputs, output_hidden_states=True)
  last_hidden = qwen_outputs.hidden_states[-1]    # [B, L, 2048]
  action_tokens = last_hidden[action_indices]      # [B, 8, 2048]
  embodied_action_tokens = last_hidden[embodied_indices]  # [B, 32, 2048]

Step 2 - V-JEPA 视频编码：
  input_videos = processor.preprocess(video)     # [B*V, T, C, H, W]
  with torch.no_grad():
      video_embeddings = encoder.get_vision_features(input_videos)

Step 3 - Predictor（世界模型预测）：
  input_states = video_embeddings[:, :(T-1)*D, :]  # 前 7 帧
  gt_states = video_embeddings[:, (T-1)*D:, :]      # 第 8 帧
  predicted_states = vj_predictor(input_states, action_tokens)
  wm_loss = L1_loss(predicted_states, gt_states)    # × 0.1

Step 4 - Flow Matching（动作预测）：
  actions_target = actions[:, -(F+1):, :]    # 7 步动作 chunk
  actions_repeated = repeat(actions_target, repeated_diffusion_steps)
  action_loss = action_model(embodied_action_repeated, actions_repeated, state)
```

### 3.4 Flow Matching 细节

```python
# 训练时
eps = N(0, 1)                          # 采样噪声
t = Beta(1.5, 1.0)                     # 采样时间步
t_prime = (s - t) / s                  # 缩放
noisy_traj = (1-t) * eps + t * action  # 加噪轨迹
velocity = action - eps                # 目标速度
pred_velocity = DiT(noisy_traj, t, conditioning)
action_loss = MSE(pred_velocity, velocity)

# 推理时（DDIM，4 步）
x = N(0, 1)                            # 从噪声开始
for t in reversed(timesteps):
    v = DiT(x, t, conditioning)
    x = x + v * dt                     # 欧拉积分
actions = x                            # 最终动作轨迹
```

---

## 4. 损失函数

| 损失 | 来源 | 公式 | 权重 |
|---|---|---|---|
| `action_loss` | Flow Matching | MSE(pred_velocity, velocity) | × 1.0 |
| `wm_loss` | V-JEPA Predictor | L1(pred_states, gt_states) | × 0.1 |

`total_loss = sum(loss * scale for loss, scale in losses)`

---

## 5. 优化器与学习率

| 参数组 | 学习率 | 覆盖模块 |
|---|---|---|
| `base` | 3e-5 | VLM backbone（Qwen3-VL） |
| `qwen_vl_interface` | 1e-5 | VLM 接口层（Qwen 的视觉投影层等） |
| `action_model` | 1e-4 | DiT 动作头 |

- **Optimizer**: AdamW (betas=[0.9, 0.95], eps=1e-8, weight_decay=1e-8)
- **Scheduler**: Cosine with min_lr (min_lr=1e-6, warmup_steps=5000)
- **Mixed Precision**: bf16 (DeepSpeed ZeRO-2)
- **Gradient Clipping**: max_grad_norm=1.0
- **Gradient Checkpointing**: 启用
- **冻结模块**: V-JEPA Encoder（`torch.no_grad()`），其他可选

---

## 6. 核心创新点

1. **世界模型增强 VLA**：在标准 VLA 基础上加入 V-JEPA 视频编码器和 Action-Conditioned Predictor，同时在 latent 空间预测未来视频帧作为辅助监督
2. **Action-Conditioned 视频预测**：VLM 输出的 `action_tokens` 作为 predictor 条件，将语言/视觉特征与动作信息融合到世界模型预测中
3. **双阶段联合训练 (Co-training)**：同时使用机器人数据（VLA loss）和纯视频数据（world model loss），利用无动作视频数据提升世界模型泛化能力
4. **Flow Matching 动作头**：来自 NVIDIA GR00T N1.5，比标准扩散更高效，推理仅需 4 步 DDIM

---

## 7. 关键配置参数

以 `scripts/config/vlajepa_cotrain.yaml` 为例：

```yaml
run_id: cotrain

framework:
  name: VLA_JEPA
  qwenvl:
    base_vlm: Qwen3-VL-2B-Instruct
    attn_implementation: flash_attention_2
  action_model:
    action_model_type: DiT-B
    action_dim: 7           # 动作空间维度
    state_dim: 8            # 状态空间维度
    future_action_window_size: 6  # 预测未来步数
    action_horizon: 7       # chunk = 1(past) + 6(future)
    repeated_diffusion_steps: 8   # loss 重复次数
    num_inference_timesteps: 4    # 推理步数
  vj2_model:
    base_encoder: vjepa2-vitl-fpc64-256
    num_frames: 8

datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: /path/to/data
    data_mix: droid           # 或 libero_all / bridge / ...
    per_device_batch_size: 16
    resolution_size: 224      # VLM 图像输入分辨率
    video_resolution_size: 256  # 世界模型视频分辨率
  video_data:
    dataset_py: video_datasets
    per_device_batch_size: 16

trainer:
  max_train_steps: 50000
  learning_rate:
    base: 3e-5
    qwen_vl_interface: 1e-5
    action_model: 1e-4
  loss_scale:
    vla: 1.0
    vlm: 0.1
  save_interval: 10000
  eval_interval: 100
  freeze_modules: ''          # 可选：冻结模块列表
  gradient_accumulation_steps: 1
```

---

## 8. 模型权重来源

| 权重 | 来源 | 说明 |
|---|---|---|
| Qwen3-VL-2B | https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct | VLM 骨干，需自行下载 |
| V-JEPA2 VIT-L | https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 | 世界模型编码器，需自行下载 |
| Co-training checkpoint | https://huggingface.co/ginwind/VLA-JEPA (Pretrain/) | 阶段一预训练权重（可选） |
| Fine-tuned checkpoints | 同上 (LIBERO/, SimplerEnv/, Real-world/) | 阶段二微调权重 |

---

## 9. 启动训练

```bash
# 阶段一：Co-training
bash scripts/vlajepa_cotrain.sh

# 阶段二：Robot Fine-tuning
bash scripts/vlajepa_robot_ft.sh
```

每个脚本使用 `accelerate launch` + DeepSpeed ZeRO-2，默认 8 GPU。

---

## 10. 核心文件索引

| 功能 | 文件路径 |
|---|---|
| 模型定义 | `starVLA/model/framework/VLA_JEPA.py` |
| VLM 接口 | `starVLA/model/modules/vlm/QWen3.py` |
| 动作头 (Flow Matching) | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |
| DiT 模型 | `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` |
| 世界模型 Predictor | `starVLA/model/modules/world_model/vj2_predictor.py` |
| 数据加载器入口 | `starVLA/dataloader/__init__.py` |
| 机器人数据集 | `starVLA/dataloader/lerobot_datasets.py` |
| LeRobot v2.1 实现 | `starVLA/dataloader/gr00t_lerobot/datasets.py` |
| 视频数据集 | `starVLA/dataloader/video_datasets.py` |
| 数据集混合配置 | `starVLA/dataloader/gr00t_lerobot/mixtures.py` |
| Co-train 训练入口 | `starVLA/training/train_vlajepa_cotrain.py` |
| 微调训练入口 | `starVLA/training/train_starvla.py` |
| 训练工具 (LR, optimizer) | `starVLA/training/trainer_utils/trainer_tools.py` |
| Co-train 配置 | `scripts/config/vlajepa_cotrain.yaml` |
| 微调配置 | `scripts/config/vlajepa_robot_ft.yaml` |
