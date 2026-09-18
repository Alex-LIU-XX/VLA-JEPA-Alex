# 01 · 训练前数据转换

把真机采集的 **LeRobot v3.0** 数据转成 VLA-JEPA 能吃的 **LeRobot v2.1** 格式，
并完成新机器人类型的注册。

> **本文档是训练的前置步骤。** 转换完成后请继续看 [`02_training.md`](./02_training.md)。

---

## 目录

1. [什么时候需要转换](#1-什么时候需要转换)
2. [格式对照](#2-格式对照)
3. [转换命令](#3-转换命令)
4. [转换后校验](#4-转换后校验)
5. [注册新机器人类型](#5-注册新机器人类型)
6. [常见问题](#6-常见问题)
7. [命令速查](#7-命令速查)

---

## 1. 什么时候需要转换

piper 这条线（训练与评测的全部现有配置）都走 `dataset_py: lerobot_datasets`，
**只吃 v2.1**。仓库里虽然另有 `lerobot_v3_datasets.py`，但它需要 `img_keys` /
`state_key` / `action_key` / `resize_size` 等额外配置项，目前没有任何 piper 配置接入它。

判断你的数据是哪个版本：

```bash
cat <dataset>/meta/info.json | head -3
# "codebase_version": "v3.0"   → 需要转换
# "codebase_version": "v2.1"   → 已经是目标格式，直接跳到第 5 节
```

或者看目录结构：

```bash
ls <dataset>/meta/
# v3.0:  episodes/  info.json  stats.json  tasks.parquet        ← 需要转换
# v2.1:  episodes.jsonl  info.json  modality.json  stats_gr00t.json  tasks.jsonl
```

---

## 2. 格式对照

| 项目 | v3.0（原始采集） | v2.1（VLA-JEPA 需要） |
|---|---|---|
| 版本号 | `codebase_version: "v3.0"` | `codebase_version: "v2.1"` |
| 低维数据 | **多 episode 混在** `data/chunk-000/file-000.parquet` | **每 episode 独立** `data/chunk_00000/episode_000000.parquet` |
| 视频 | **多 episode 混在** `videos/<key>/chunk-000/file-000.mp4` | **每 episode 独立** `videos/<key>/chunk_00000/episode_000000.mp4` |
| episode 元数据 | `meta/episodes/chunk-000/file-000.parquet` | `meta/episodes.jsonl` |
| 任务定义 | `meta/tasks.parquet` | `meta/tasks.jsonl` |
| 列映射 | **无** | `meta/modality.json`（**必需**） |
| 统计量 | `meta/stats.json` | `meta/stats_gr00t.json` |
| 机器人类型 | — | 需在 `ROBOT_TYPE_CONFIG_MAP` 注册 |

---

## 3. 转换命令

### 3.1 用哪个脚本

仓库里有两个转换脚本，**新数据一律用 `_aligned` 版本**：

| 脚本 | 说明 |
|---|---|
| `scripts/convert_v3_to_v2_1_aligned.py` | ✅ **推荐**。按帧号切片；修复多 parquet 文件的全局索引 bug；自动清理采样缓存 |
| `scripts/convert_v3_to_v2_1.py` | ⚠️ 旧版。对本项目数据会出错（见下方"旧脚本的两个 bug"） |

> **旧脚本的两个 bug**（[`archive/adjust_cup_real_world.md`](./archive/adjust_cup_real_world.md) 有完整记录）：
> 1. `dataset_from_index` / `dataset_to_index` 是**全局行号**，旧脚本直接拿它切当前文件，
>    导致第二个数据文件中的 episode 全部取空（实测只得到 5034/6837 行）；
> 2. 依赖浮点 PTS 精确匹配，遇到 AV1 重编码的视频会错帧。
>    `_aligned` 版本改为 `from_timestamp × fps` 定位起始帧 + `length` 切片。

### 3.2 标准命令

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA

python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /share/home/tm866052366100000/a926312360/LXX/project/Datasets/real_world/miku112/pick_open_place_0724_1_offset_state \
  --output /share/home/tm866052366100000/a926312360/LXX/project/Datasets/lerobot/miku112/pick_open_place_0724_1_offset_state_v2_1_full \
  --fps 10 \
  --chunk-size 100
```

**参数说明**：

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--input` | ✅ | — | v3.0 数据集路径 |
| `--output` | ✅ | — | 输出 v2.1 路径（不存在会自动创建） |
| `--fps` | | 10 | 视频帧率，**必须与采集时一致** |
| `--chunk-size` | | 100 | 每多少个 episode 分一个 chunk 目录 |

**耗时参考**：`adjust_cup_0409`（50 episodes / 6837 帧）约 **42 秒**。

### 3.3 本项目实际用过的命令

```bash
# adjust_cup_0409（真机，50 eps / 6837 帧）
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_converted \
  --output /share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_v2_1 \
  --fps 10 --chunk-size 100

# pick_open_place_0724（真机，48 eps / 12327 帧）
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /share/home/tm866052366100000/a926312360/LXX/project/Datasets/real_world/miku112/pick_open_place_0724_1_offset_state \
  --output /share/home/tm866052366100000/a926312360/LXX/project/Datasets/lerobot/miku112/pick_open_place_0724_1_offset_state_v2_1_full \
  --fps 10 --chunk-size 100
```

> 历史上 `Datasets/lerobot/miku112/` 下的三个 `_v2_1_full` 数据集
> （`pick_place_block_all_0408` / `pick_open_place_0724` / `pick_food_pot_0725`）
> 是用旧脚本 `convert_v3_to_v2_1.py` 转的，其 `tasks.jsonl` 被写成了数字占位符（见 §6.4）。

### 3.4 转换脚本会自动做什么

1. 按帧号把 v3.0 的大 parquet / mp4 **切成每 episode 独立的文件**；
2. 生成 `meta/modality.json`（piper 的 `x/y/z/roll/pitch/yaw/gripper` 切片定义）；
3. 生成带 `info` 块的 `meta/info.json`（**缺这个字段训练会报 `KeyError: 'info'`**）；
4. 写出真实的 `meta/tasks.jsonl`；
5. **清理 `meta/steps_*.pkl` 采样缓存**，避免其它数据集残留的索引串味。

---

## 4. 转换后校验

**转换完不要直接开训**，先跑完这一节的检查。

### 4.1 结构与计数

```bash
D=/share/home/tm866052366100000/a926312360/LXX/project/Datasets/lerobot/miku112/xxx_v2_1_full

# 必需文件是否齐全
ls $D/meta/
# 期望：episodes.jsonl  info.json  modality.json  stats_gr00t.json  tasks.jsonl

# 版本号
python -c "import json;print(json.load(open('$D/meta/info.json'))['codebase_version'])"   # 期望 v2.1

# episode / 帧数
python -c "
import json
i=json.load(open('$D/meta/info.json'))
print('episodes', i['total_episodes'], '| frames', i['total_frames'], '| tasks', i['total_tasks'])
"

# 每个 episode 的 parquet 行数是否等于 episode length
python -c "
import json, pandas as pd
from pathlib import Path
D=Path('$D')
eps=[json.loads(l) for l in open(D/'meta'/'episodes.jsonl')]
bad=[]
for e in eps:
    p=D/'data'/f\"chunk_{e['episode_index']//100:05d}\"/f\"episode_{e['episode_index']:06d}.parquet\"
    n=len(pd.read_parquet(p, columns=['action']))
    if n!=e['length']: bad.append((e['episode_index'], n, e['length']))
print('行数不匹配的 episode:', bad if bad else '无 ✅')
"
```

### 4.2 视频帧数

```bash
# 抽查：视频总帧数应等于 episode length
python -c "
import av
from pathlib import Path
p=Path('$D/videos/observation.images.image/chunk_00000/episode_000000.mp4')
c=av.open(str(p)); print('frames', sum(1 for _ in c.decode(video=0)))
"
```

### 4.3 数据能被加载器正常读出（最关键的检查）

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA
.venv/bin/python -c "
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg = OmegaConf.load('checkpoints/adjust_cup_10k/config.yaml')   # 借一份 piper 配置
d = cfg.datasets.vla_data
d.data_root_dir = '$D'
ds = get_vla_dataset(data_cfg=d, mode='val', action_horizon=7, video_horizon=8)
it = ds[0]
print('len        :', len(ds))
print('lang       :', repr(it['lang']))
print('action     :', it['action'].shape, it['action'].dtype)   # 期望 (7, 7) float16
print('state      :', it['state'].shape)                        # 期望 (1, 7)
print('n images   :', len(it['image']), it['image'][0].size)    # 期望 2 张 (224, 224)
print('video      :', it['video'].shape)                        # 期望 (2, 8, 256, 256, 3)
"
```

期望输出：

```
len        : 12327
lang       : '22'
action     : (7, 7) float16
state      : (1, 7)
n images   : 2 (224, 224)
video      : (2, 8, 256, 256, 3)
```

### 4.4 检查任务文本是不是数字占位符

```bash
head -3 $D/meta/tasks.jsonl
# {"task_index": 0, "task": "0"}          ← ⚠️ 数字占位符，语言条件失效
# {"task_index": 0, "task": "把杯子..."}   ← 正常
```

详见 §6.4。

---

## 5. 注册新机器人类型

**如果机器人是 `piper`，这一步已经做完了**（见 §5.4），可以跳过。
换别的机械臂才需要做。

需要改**三个文件**，缺一个都会报错：

### 5.1 `starVLA/dataloader/gr00t_lerobot/data_config.py`

新增一个 Config 类：

```python
class MyRobotConfig:
    video_keys = [
        "video.observation.images.image",
        "video.observation.images.wrist_image",
    ]
    state_keys = ["state.x", "state.y", "state.z", "state.roll",
                  "state.pitch", "state.yaw", "state.gripper"]
    action_keys = ["action.x", "action.y", "action.z", "action.roll",
                   "action.pitch", "action.yaw", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]

    def __init__(self, observation_indices, action_indices):
        self.observation_indices = observation_indices
        self.action_indices = action_indices

    def modality_config(self):
        video_modality = ModalityConfig(delta_indices=self.observation_indices,
                                        modality_keys=self.video_keys)
        state_modality = ModalityConfig(delta_indices=self.observation_indices,
                                        modality_keys=self.state_keys)
        action_modality = ModalityConfig(delta_indices=self.action_indices,
                                         modality_keys=self.action_keys)
        language_modality = ModalityConfig(delta_indices=self.observation_indices,
                                           modality_keys=self.language_keys)
        return {"video": video_modality, "state": state_modality,
                "action": action_modality, "language": language_modality}

    def transform(self):
        transforms = [
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "min_max" for k in self.action_keys},
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
```

**关键约束**：这里的字段名（`x` / `y` / `z` / …）必须与 `modality.json` 中的键名
**完全一致**，且 `video_keys` 要用 `video.{modality.json 中的 key}` 前缀格式。

然后在文件末尾注册：

```python
ROBOT_TYPE_CONFIG_MAP = {
    # ... 已有的 ...
    "my_robot": MyRobotConfig,
}
```

### 5.2 `starVLA/dataloader/gr00t_lerobot/mixtures.py`

```python
DATASET_NAMED_MIXTURES = {
    # ... 已有的 ...
    "my_data": [
        ("", 1.0, "my_robot"),
    ],
}
```

三元组含义：`(数据子目录名, 采样权重, robot_type)`

- **数据子目录名**：数据直接放在 `data_root_dir/` 下（即 `data_root_dir/meta/` 存在）时写 `""`；
  放在 `data_root_dir/my_subdir/` 下时写 `"my_subdir"`
- **robot_type**：5.1 中注册的 key

### 5.3 `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`

```python
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # ... 已有的 ...
    "my_robot": EmbodimentTag.NEW_EMBODIMENT,
}
```

### 5.4 piper 的注册结果（已完成，供参考）

| 文件 | 内容 |
|---|---|
| `data_config.py` | `class PiperDataConfig`（7 维 state / 7 维 action / 双视角） |
| `data_config.py` | `ROBOT_TYPE_CONFIG_MAP["piper"] = PiperDataConfig` |
| `mixtures.py` | `"piper_pick_place": [("", 1.0, "piper")]` |
| `embodiment_tags.py` | `"piper" → EmbodimentTag.NEW_EMBODIMENT` |

因为 `piper_pick_place` 的子目录名是**空串**，所以配置里的 `data_root_dir`
要**直接指向数据集目录本身**，不要再套一层。

---

## 6. 常见问题

### 6.1 视频解不开：AV1 编码

v3.0 采集的视频通常是 **AV1** 编码，系统自带的 ffmpeg 4.3 往往没编译 AV1 支持：

```
ffprobe: codec_name=av1
ffmpeg version 4.3: 未编译 AV1 支持
```

**解决**：转换脚本内部用 **PyAV**（走系统 libdav1d 解码），不需要动系统 ffmpeg。
只要确认 `av` 包可导入即可：

```bash
python -c "import av; print(av.__version__)"
```

### 6.2 Timestamp 偏移错位

v3.0 的 `timestamp` 列是**相对每个 episode 起始**（0-based），
而视频文件里的 timestamp 是**相对整个文件**：

```python
absolute_timestamp = parquet_timestamp + ep_row["videos/<key>/from_timestamp"]
```

`_aligned` 脚本已按 `from_timestamp × fps` 定位起始帧，无需手工处理。

### 6.3 `ValueError: No suitable position columns found`

```
ValueError: No suitable position columns found.
Available columns: ['observation.state', 'action', 'timestamp', ...]
```

**原因**：`_get_position_and_gripper_values()`（用于检测暂停帧）硬编码查找
`action.x` / `action.y` / `action.z` / `action.gripper`，而你的 `modality.json`
用的是 `joint_1`~`joint_6`。

**解决**：把 `modality.json` 的 key 改名为 `x, y, z, roll, pitch, yaw, gripper`。

> ⚠️ **这只是改切片标签名，不改数据内容。**
> Piper 的控制量确实是 6 关节角 + 夹爪，不是末端位姿 xyzrpy。
> `start/end` 索引才决定切哪一段：
>
> | modality.json key | start→end | 实际取到的数据 |
> |---|---|---|
> | `x` | 0→1 | `joint_1` |
> | `y` | 1→2 | `joint_2` |
> | `z` | 2→3 | `joint_3` |
> | `roll` | 3→4 | `joint_4` |
> | `pitch` | 4→5 | `joint_5` |
> | `yaw` | 5→6 | `joint_6` |
> | `gripper` | 6→7 | `gripper` |
>
> 命名成 `x/y/z...` 只是**为了绕过那个硬编码检查**，喂给模型的 7 维数值仍是正确的关节角。

### 6.4 ⚠️ 任务文本被写成数字占位符

**这是本项目当前数据的一个真实缺陷。** 转换后的 `tasks.jsonl` 长这样：

```json
{"task_index": 0, "task": "0"}
{"task_index": 1, "task": "1"}
```

也就是说模型训练时看到的"语言指令"就是 `"0"` / `"22"` 这种数字，
**语言条件实际是失效的**。

**影响**：开环测试结论只覆盖「视觉 + 状态 → 动作」，语言通道的贡献无法评估，
也不排除指令噪声拉低了整体表现。

**修法**：转换时从 v3.0 的 `meta/tasks.parquet` 读出真实文本写回 `tasks.jsonl`。
注意 `tasks.parquet` 的 **index 是任务描述文本**、`task_index` 列才是数值索引：

```python
task_df = pd.read_parquet("meta/tasks.parquet")
for task_text, row in task_df.iterrows():
    print(row["task_index"], "→", str(task_text))
```

### 6.5 复制数据集后采样索引串味

`meta/steps_2d5a34b904d2.pkl` 是采样索引缓存，**文件名是硬编码的**
（`datasets.py` 里写死了 `steps_332420bad1ab.pkl` / `steps_2d5a34b904d2.pkl`）。

复制数据集时**必须删掉它**，否则会加载到别的数据集的索引：

```bash
rm -f <dataset>/meta/steps_*.pkl
```

`_aligned` 转换脚本会自动清理。

### 6.6 `ValueError: 'channel' is not in list` → `KeyError: 'info'`

**原因**：`info.json` 的 video feature 缺少 `"info"` 块。

**解决**：确保每个 video feature 都带 `info`：

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

---

## 7. 命令速查

```bash
# ---- 1. 判断版本 ----
python -c "import json;print(json.load(open('DATASET/meta/info.json'))['codebase_version'])"

# ---- 2. 转换（推荐 aligned 版本）----
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  <v3.0 数据路径> \
  --output <v2.1 输出路径> \
  --fps 10 --chunk-size 100

# ---- 3. 校验 ----
ls <v2.1>/meta/                                  # 应有 modality.json / stats_gr00t.json
rm -f <v2.1>/meta/steps_*.pkl                    # 清采样缓存
head -3 <v2.1>/meta/tasks.jsonl                  # 检查任务文本是否数字占位符

# ---- 4. 加载器自检 ----
.venv/bin/python -c "
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg = OmegaConf.load('checkpoints/adjust_cup_10k/config.yaml')
d = cfg.datasets.vla_data; d.data_root_dir = '<v2.1 输出路径>'
ds = get_vla_dataset(data_cfg=d, mode='val', action_horizon=7, video_horizon=8)
print(len(ds), ds[0]['action'].shape, repr(ds[0]['lang']))
"
```

**下一步** → 修改训练配置并启动训练，见 [`02_training.md`](./02_training.md)。

---

## 相关文档

| 文档 | 说明 |
|---|---|
| [`02_training.md`](./02_training.md) | 训练启动命令说明 |
| [`03_openloop_testing.md`](./03_openloop_testing.md) | 训练完成后的开环测试 |
| [`archive/debug_journal.md`](./archive/debug_journal.md) | 完整的 Piper 适配调试记录（含所有原始报错） |
| [`archive/adjust_cup_real_world.md`](./archive/adjust_cup_real_world.md) | adjust_cup 真机数据接入全过程与冒烟测试报告 |
