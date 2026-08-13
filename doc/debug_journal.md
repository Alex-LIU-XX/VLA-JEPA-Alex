# VLA-JEPA 自定义数据集调试记录

本文档记录了将自定义 Piper 机器人数据集（LeRobot v3.0）适配到 VLA-JEPA 进行第二阶段训练的全过程，包括所有遇到的问题、排查思路和解决方案。

---

## 目录

1. [数据集格式检查](#1-数据集格式检查)
2. [数据集转换 v3.0 → v2.1](#2-数据集转换-v30--v21)
3. [注册机器人配置](#3-注册机器人配置)
4. [训练启动与问题排查](#4-训练启动与问题排查)
5. [最终成功运行](#5-最终成功运行)

---

## 1. 数据集格式检查

### 1.1 用户提供的数据

```
路径：/share/home/.../pick_place_block_all_0408_1_offset_state/
├── meta/
│   ├── info.json          # codebase_version: "v3.0"
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet   # 15026 帧，多 episode 混存
└── videos/
    ├── observation.images.image/
    │   └── chunk-000/
    │       └── file-000.mp4   # AV1 编码，83.7MB，含多 episode
    └── observation.images.wrist_image/
        └── chunk-000/
            └── file-000.mp4
```

### 1.2 发现的问题

| 检查项 | 实际值 | VLA-JEPA 期望值 | 结果 |
|---|---|---|---|
| 格式版本 | `v3.0` | `v2.1` | ❌ 不兼容 |
| 数据文件 | 多 episode 混合在同一个 parquet | 每 episode 独立 parquet | ❌ |
| 视频文件 | 多 episode 混合在同一个 mp4，timestamp 索引 | 每 episode 独立 mp4 | ❌ |
| Episode 元数据 | `meta/episodes/chunk-000/file-000.parquet` | `meta/episodes.jsonl` | ❌ |
| Task 定义 | `meta/tasks.parquet`（index 存文本） | `meta/tasks.jsonl` | ❌ |
| `modality.json` | 缺失 | **必需** | ❌ |
| 机器人类型 | `piper` | 需注册 | ❌ |

### 1.3 数据内容（合格项）

- ✅ 100 个 episodes，15026 帧
- ✅ 7 维 action（joint_1~6 + gripper）
- ✅ 7 维 state（joint_1~6 + gripper）
- ✅ 双视角视频（image + wrist_image）
- ✅ 43 个任务描述
- ✅ 视频帧率 10fps，分辨率 640×480
- ✅ 视频可正常解码（PyAV/libdav1d）

---

## 2. 数据集转换 v3.0 → v2.1

### 2.1 转换脚本

`scripts/convert_v3_to_v2_1.py`

核心逻辑：

```python
# 读取 v3.0 episode 元数据
ep_df = pd.read_parquet("meta/episodes/chunk-000/file-000.parquet")

for 每个 episode:
    # 从 v3.0 大数据文件中切片出当前 episode 的数据
    ep_data = df.iloc[data_from:data_to].copy()

    # 独立保存 parquet
    ep_data.to_parquet(f"data/chunk_{chunk:05d}/episode_{ep_idx:06d}.parquet")

    # 从 v3.0 视频中通过 timestamp 提取帧
    for camera in ['image', 'wrist_image']:
        frames = extract_frames(v3_video, timestamps + from_ts)
        write_video(frames, f"videos/{camera}/chunk_{chunk:05d}/episode_{ep_idx:06d}.mp4")
```

### 2.2 视频解码问题

**问题**：v3.0 视频使用 **AV1** 编码，系统自带的 ffmpeg 不支持解码。

```
ffprobe: codec_name=av1
ffmpeg version 4.3: 未编译 AV1 支持
```

**解决**：使用 Python 的 **PyAV** 库，它通过系统 libdav1d 解码 AV1：

```python
import av
container = av.open("video.mp4")
for frame in container.decode(video=0):
    img = frame.to_ndarray(format="rgb24")
```

### 2.3 Timestamp 偏移问题

**关键发现**：v3.0 的 `timestamp` 列是相对于 **每个 episode 起始** 的（0-based），但视频文件中的 timestamp 是相对于 **整个视频文件** 的。

```python
# v3.0 episodes 表中有 from_timestamp 字段
from_ts = ep_row["videos/{vk}/from_timestamp"]

# 从视频中提取帧时需要加上偏移量
absolute_timestamp = parquet_timestamp + from_ts
```

### 2.4 Task 解析问题

**问题**：`tasks.parquet` 的 index 是任务描述文本，`task_index` 列是数值索引。

```python
# 正确解析方式
task_df = pd.read_parquet("meta/tasks.parquet")
for task_text, row in task_df.iterrows():
    entry = {"task_index": int(row['task_index']), "task": str(task_text)}
```

---

## 3. 注册机器人配置

### 3.1 添加 DataConfig

**文件**：`starVLA/dataloader/gr00t_lerobot/data_config.py`

```python
class PiperDataConfig:
    video_keys = [
        "video.observation.images.image",
        "video.observation.images.wrist_image",
    ]
    state_keys = [
        "state.x", "state.y", "state.z",
        "state.roll", "state.pitch", "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z",
        "action.roll", "action.pitch", "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    # ... modality_config() 和 transform() 方法
```

**重要**：key 的名称（如 `x`, `y`, `z`, `gripper`）必须与 `modality.json` 中的 key 完全一致。

### 3.2 注册到 ROBOOT_TYPE_CONFIG_MAP

```python
ROBOT_TYPE_CONFIG_MAP = {
    ...
    "piper": PiperDataConfig,
}
```

### 3.3 添加混合配置

**文件**：`starVLA/dataloader/gr00t_lerobot/mixtures.py`

```python
DATASET_NAMED_MIXTURES = {
    ...
    "piper_pick_place": [
        ("", 1.0, "piper"),
    ],
}
```

- 第一个参数 `""` 表示数据直接放在 `data_root_dir/` 下（无子目录）
- 第二个参数 `1.0` 是采样权重
- 第三个参数 `"piper"` 对应 `ROBOT_TYPE_CONFIG_MAP` 中的 key

### 3.4 添加 Embodiment Tag

**文件**：`starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`

```python
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    ...
    "piper": EmbodimentTag.NEW_EMBODIMENT,
}
```

---

## 4. 训练启动与问题排查

### 4.1 问题 1：info.json 缺少 video 字段

**错误**：
```
ValueError: 'channel' is not in list  →  KeyError: 'info'
```

**原因**：转换脚本生成的 `info.json` 中，video feature 缺少 `"info"` 字段。

**解决**：在 `info.json` 的每个 video feature 中添加 `"info"`：

```json
"observation.images.image": {
    "dtype": "video",
    "shape": [3, 480, 640],
    "names": ["channels", "height", "width"],
    "info": {
        "video.height": 480,
        "video.width": 640,
        "video.fps": 10,
        "video.channels": 3
    }
}
```

### 4.2 问题 2：状态/动作 Key 命名不匹配

**错误**：
```
ValueError: No suitable position columns found.
Available columns: ['observation.state', 'action', 'timestamp', ...]
```

**原因**：`_get_position_and_gripper_values()` 方法硬编码查找 `x`, `y`, `z`, `gripper` 等 action key 名称，但原始 `modality.json` 中使用的是 `joint_1`~`joint_6`。

**解决**：将 `modality.json` 中的 key 名称改为 `x`, `y`, `z`, `roll`, `pitch`, `yaw`, `gripper`。

> ⚠️ **重要说明**：此修改**只改了切片标签名，不改实际数据内容**。
> 
> Piper 机械臂的控制量确实是 6 个关节角 + 夹爪开合（joint_1~6 + gripper），**不是**末端位姿（xyzrpy）。
> 但 `modality.json` 中的 `start/end` 索引定义的是 7 维向量中切哪一段：
> 
> ```
> 原始 parquet 列 "action" 存的 7 维向量：
> [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
>    ↑        ↑        ↑        ↑        ↑        ↑         ↑
>   start=0  start=1  start=2  start=3  start=4  start=5   start=6
>   end=1    end=2    end=3    end=4    end=5    end=6     end=7
> ```
> 
| modality.json 中的 key | start → end | 实际取到的数据 | 含义 |
|---|---|---|---|
| `x` | 0→1 | `joint_1` 的值 | 关节角 → 但代码硬编码搜 `action.x` |
| `y` | 1→2 | `joint_2` 的值 | 关节角 → 代码硬编码搜 `action.y` |
| `z` | 2→3 | `joint_3` 的值 | 关节角 → 代码硬编码搜 `action.z` |
| `roll` | 3→4 | `joint_4` 的值 | 关节角 → 代码硬编码搜 `action.roll` |
| `pitch` | 4→5 | `joint_5` 的值 | 关节角 → 代码硬编码搜 `action.pitch` |
| `yaw` | 5→6 | `joint_6` 的值 | 关节角 → 代码硬编码搜 `action.yaw` |
| `gripper` | 6→7 | `gripper` 的值 | 夹爪 → 代码硬编码搜 `action.gripper` |
>
> **命名为 `x, y, z...` 只是为了绕过** `_get_position_and_gripper_values()` 的硬编码检查（该函数用于检测暂停帧，需要找 `action.x`, `action.y`, `action.z`, `action.gripper` 这些 key 名）。实际喂给模型训练的 7 维数值仍然是正确的关节角数据，**不是** xyzrpy 末端位姿。
> 
> 如果希望保持原名 `joint_1~6`，则需要修改 `_get_position_and_gripper_values()` 方法（约 530-600 行），在 `position_candidates` 和 `gripper_candidates` 中添加 joint 类型的 key。但目前用 `x, y, z...` 是安全的。

同时同步修改 `data_config.py` 中的 `PiperDataConfig.state_keys` 和 `action_keys`。

### 4.3 问题 3：DeepSpeed 分布式初始化失败

**错误**：
```
ValueError: Default process group has not been initialized
```

**原因**：单 GPU 调试时直接运行 `python train_starvla.py`，但代码中 `dist.get_rank()` 在 `accelerator.prepare()` 之前被调用。

**解决**：使用 `accelerate launch` 启动，并指定 DeepSpeed config 文件：

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/piper_test.yaml
```

### 4.4 问题 4：CUDA Out of Memory

**错误**：
```
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 7.93 GiB. GPU has 23.52 GiB total.
```

**原因**：2.77B 参数模型 + Adam 优化器状态（fp32）需要远超 24GB 显存。

- 模型权重 (bf16)：~5.5 GB
- 梯度 (bf16)：~5.5 GB
- Adam states (fp32)：~22 GB（momentum + variance）
- 总计 >> 24 GB

**解决 1**：启用 ZeRO-2 + CPU Offload（将优化器状态放到 CPU 内存）：

```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

效果：GPU 显存从 >24GB 降至 **~5.7 GB**。

**解决 2**：安装 ninja 编译 DeepSpeed CPU Adam 算子：

```bash
pip install ninja
```

---

## 5. 最终成功运行

### 5.1 训练命令

```bash
cd /share/home/.../VLA-JEPA
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/piper_test.yaml
```

### 5.2 训练配置

```yaml
# scripts/config/piper_test.yaml（测试用，50 步）
datasets:
  vla_data:
    data_root_dir: /path/to/pick_place_block_all_0408_1_offset_state_v2_1_test_small
    data_mix: piper_pick_place
    per_device_batch_size: 2
trainer:
  max_train_steps: 50
  num_warmup_steps: 5
  enable_gradient_checkpointing: true
  enable_mixed_precision_training: true
```

### 5.3 训练输出

```
📊 model parameter statistics:
# Parameters (in millions): 2770.332 Total, 2770.332 Trainable

Step   5, Loss: {'action_loss': 1.786, 'wm_loss': 0.196, 'learning_rate': 1e-05}
Step  10, Loss: {'action_loss': 1.661, 'wm_loss': 0.192, 'mse_score': 0.116, 'mae_score': 0.912}
...
```

Loss 正常下降，模型可以成功训练。

---

## 6. 关键文件清单

| 文件 | 用途 |
|---|---|
| `scripts/convert_v3_to_v2_1.py` | v3.0 → v2.1 格式转换脚本 |
| `scripts/config/piper_test.yaml` | Piper 机器人测试训练配置 |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | 添加了 `PiperDataConfig` |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 添加了 `piper_pick_place` 条目 |
| `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py` | 添加了 `piper` → `NEW_EMBODIMENT` |
| `starVLA/config/deepseeds/ds_config_test.json` | 测试用 DeepSpeed 配置（CPU offload） |
| `starVLA/config/deepseeds/accelerate_test.yaml` | 测试用 Accelerate 配置（1 GPU） |

---

## 7. 经验总结

### 数据适配要点

1. **VLA-JEPA 只支持 LeRobot v2.1**，v3.0 需要先转换
2. v3.0→v2.1 转换的核心是**将多 episode 混合的 parquet/video 拆分成单 episode 文件**
3. v3.0 的 timestamp 是 episode 内 0-based，视频文件内需要加 `from_timestamp` 偏移
4. `modality.json` 的 state/action key 名称需要使用 `x, y, z, roll, pitch, yaw, gripper`（而不是 `joint_1` 等），否则 `_get_position_and_gripper_values` 会报错

### 训练环境要点

1. **单 GPU 无法容纳 2.77B 模型**：必须使用 DeepSpeed ZeRO-2 + CPU Offload，或至少 4-8 张 GPU
2. DeepSpeed CPU Adam 需要 `ninja` 编译 C++ 扩展
3. `accelerate launch` + DeepSpeed 是强制要求，不能直接 `python train.py`
4. AV1 编码视频需要 PyAV 解码（系统 ffmpeg 通常不支持 AV1）

### 注册新机器人要点

1. 三方注册：`data_config.py` → `mixtures.py` → `embodiment_tags.py`
2. DataConfig 的 key 名称必须与 `modality.json` 完全一致
3. `video_keys` 使用 `video.{modality.json中的key}` 格式（运行时自动去掉 `"video."` 前缀）
