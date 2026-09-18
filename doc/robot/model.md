# VLA-JEPA 模型文档（载入 / 输入 / 数据 / 输出）

> **定位**：只讲**模型自身**的契约——怎么载入、吃什么、数据怎么定义、吐什么。
> "怎么把它跑起来、接到仿真或机器人上"属于部署内容，见 [deployment.md](./deployment.md)；
> 真机 ZMQ server 的实现计划见 [piper_server_plan.md](./piper_server_plan.md)。
>
> 判定标记：`[已确认]` = 有源码/脚本原文可查（给出 `文件:行`）；`[建议]` = 工程做法，不是仓库既有实现；
> `[待确认]` = 仓库内无证据，需要实测。**本文所有路径、命令、字段名都来自真实文件。**
>
> **路径缩写**：下文中的 `data_config.py` / `datasets.py` / `mixtures.py` / `schema.py` 均指
> `starVLA/dataloader/gr00t_lerobot/` 下的同名文件；`state_action.py` 指该目录 `transform/state_action.py`；
> `VLA_JEPA.py` 指 `starVLA/model/framework/VLA_JEPA.py`；`piper_test.yaml` 等指 `scripts/config/` 下的配置。

---

## 0. 速览

| 事项 | 结论 |
|---|---|
| 载入 | `baseframework.from_pretrained("<run_dir>/final_model/pytorch_model.pt")`，要求该 `.pt` 上一级目录同时有 `config.yaml` 与 `dataset_statistics.json` |
| 输入 | `predict_action(batch_images=[[头图, 腕图]], instructions=[指令], state=[(1,7) 数组])` |
| 输出 | `{"normalized_actions": [B, future_action_window_size+1, action_dim]}` —— **归一化动作**，必须反归一化后才能下发 |
| 推理链路 | Qwen3-VL（取 `<|embodied_action|>` token 的隐状态）→ flow-matching DiT 动作头；**V-JEPA2 世界模型只在训练期使用** |
| 语言条件 | 必须复现所用数据集 `meta/tasks.jsonl` 的原文；当前 `adjust_cup_0409_1_offset_state_v2_1` 使用自然语言任务文本，不能套用旧的数字占位符结论 |

---

## 1. 模型结构与推理链路

framework 名 **`VLA_JEPA`**（`starVLA/model/framework/VLA_JEPA.py:34`），三个子模块：

| 组件 | 作用 | 代码位置 |
|---|---|---|
| Qwen3-VL-2B / Qwen2.5-VL | 视觉-语言主干，融合图像 + 语言指令 + 动作 token | `starVLA/model/modules/vlm/` |
| V-JEPA2 世界模型 | 由动作 token 预测未来帧隐表示（`wm_loss`），**仅训练期** | `starVLA/model/modules/world_model/` |
| Flow-matching DiT 动作头 | 从噪声迭代采样动作 chunk | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |

**推理路径只有两段** `[已确认]`（`VLA_JEPA.py:277-337` 全文无 `vj_encoder`/`vj_predictor` 调用）：

```
PIL 图像（多视角） + 指令
  → Qwen3-VL 前向（chat 模板 + 特殊动作 token）
  → 取 <|embodied_action|> token 位置的隐状态  →  action_model.predict_action(...)
  → flow-matching 欧拉积分 num_inference_timesteps 步
  → normalized_actions [B, future_action_window_size+1, action_dim]
```

但**世界模型的权重与权重路径仍然必须存在**：
`VLA_JEPA.__init__` 会构造 `vj_encoder`/`vj_processor`/`vj_predictor`，其中
`AutoModel.from_pretrained(config.framework.vj2_model.base_encoder)` 在构造期执行，
且 `from_pretrained` 是 `strict=True` 加载，state_dict 必须含 `vj_*` 权重。`[已确认]`
（`VLA_JEPA.py:81-98`、`base_framework.py:96-110`）

---

## 2. 模型载入

### 2.1 权重目录必须长这样

```
checkpoints/<run_id>/
├── config.yaml                 # 训练配置快照（rank0 写出）
├── config.json                 # 同一份配置的 json 版
├── dataset_statistics.json     # 动作/状态统计量（反归一化用，rank0 写出）
├── checkpoints/
│   └── steps_<N>_pytorch_model.pt
└── final_model/
    └── pytorch_model.pt
```

`[已确认]` 写出位置：`starVLA/training/train_starvla.py:79-80`（config.yaml/json）、
`:231-234`（`checkpoints/steps_<N>_pytorch_model.pt`）、`:505-508`（`final_model/pytorch_model.pt`）、
`starVLA/dataloader/__init__.py:64-67`（`dataset_statistics.json`）。

**关键约束**：`read_mode_config()` 用 `checkpoint_pt.parents[1]` 当 run 目录，所以 `.pt` 必须在
run 目录下**恰好两层**：`<run_dir>/<任意子目录>/xxx.pt`。`[已确认]`
（`starVLA/model/framework/share_tools.py:263-273`，等价实现见 `starVLA/model/tools.py:156-195`）

### 2.2 载入链路（`from_pretrained` 做了什么）

```
baseframework.from_pretrained(ckpt_path)
  1. read_mode_config(ckpt)       → (config.yaml 内容, dataset_statistics.json 内容)
  2. dict_to_namespace(cfg)        → OmegaConf DictConfig
  3. cfg.trainer.pretrained_checkpoint = None
  4. build_framework(cfg)          → 按 cfg.framework.name 从 FRAMEWORK_REGISTRY 取类并实例化
  5. model.norm_stats = norm_stats → 供反归一化
  6. torch.load(ckpt, map_location="cpu") + load_state_dict(strict=True)
```

`[已确认]`（`base_framework.py:56-114`；工厂 `starVLA/model/framework/__init__.py:35-61`；
注册表 `starVLA/model/tools.py:118-143`）

- **strict=True**：state_dict 键不匹配直接抛 `RuntimeError`（`base_framework.py:98-110`）。
- 返回的模型仍**在 CPU**，且**不会自动 `.eval()`**；设备放置与 eval 模式由调用方负责
  （`base_framework.py:77`、`deployment/model_server/server_policy.py:23-27`）。
- `**kwargs` 形参存在但**实际被忽略**（传参行被注释掉，`base_framework.py:89`）——
  不要指望 `from_pretrained(..., 覆盖参数)`。
- 开环评估脚本走另一条路（`build_framework` + `load_state_dict(strict=False)` + 关键键校验），
  因为它直接吃 `--config_yaml`，见 `scripts/eval_openloop.py:80-98`。

### 2.3 config.yaml 里与推理强相关的字段

以真机 Piper 配置 `scripts/config/piper_test.yaml` 为例：

| 字段 | 示例值 | 含义 |
|---|---|---|
| `framework.name` | `VLA_JEPA` | 决定载入哪个 framework 类（注册表名） |
| `framework.qwenvl.base_vlm` | 本地 Qwen3-VL 路径 | **必须在运行机器上存在**（构造期加载） |
| `framework.vj2_model.base_encoder` | 本地 V-JEPA2 路径 | 同理必须在（虽然推理不用它前向） |
| `framework.action_model.action_dim` | `7` | 动作维度（Piper：6 关节 + 夹爪） |
| `framework.action_model.state_dim` | `7`（Piper）/ `8`（LIBERO） | **必须与数据集 state 维度一致** |
| `framework.action_model.action_horizon` | `7` | 只用于 token 扩展与 dataloader 切片 |
| `framework.action_model.future_action_window_size` | `6` | **chunk 长度 = 它 + 1** |
| `framework.action_model.num_inference_timesteps` | `4` | flow-matching 去噪步数（真正的"采样步数"开关） |
| `datasets.vla_data.resolution_size` | `224` | 训练图像边长（调用方负责把图缩到同尺寸） |
| `datasets.vla_data.video_resolution_size` | `256` | 只影响世界模型输入，推理不需要 |
| `datasets.vla_data.with_state` | `true` | 是否使用本体状态输入 |
| `datasets.vla_data.CoT_prompt` | `"Your task is {instruction}..."` | 推理 prompt 模板（§3.3） |
| `datasets.vla_data.action_type` | `absolute`（Piper 全部任务） | 动作语义：绝对关节角 |

`[已确认]`（`scripts/config/piper_test.yaml:7-62`、`scripts/config/iclr_*.yaml:19-20,56`）

两条容易踩的细节 `[已确认]`：

- **chunk 长度的真正来源**：动作头里 `self.action_horizon = config.future_action_window_size + 1`
  （`GR00T_ActionHeader.py:233`）；yaml 的 `action_model.action_horizon` **不参与动作头形状**，
  只被 `max_action_tokens = action_horizon * 4`（`VLA_JEPA.py:66`）与 dataloader 使用
  （`starVLA/dataloader/__init__.py:52`）。
- **运行时覆盖**：`framework.action_model.diffusion_model_cfg.cross_attention_dim` 会被 VLM 的
  `hidden_size` 覆盖（`VLA_JEPA.py:73`），yaml 里的值只是占位。

### 2.4 部署机器的硬性要求

| 要求 | 原因 | 证据 |
|---|---|---|
| **必须有 CUDA GPU** | VLM 载入硬编码 `device_map="cuda"` | `starVLA/model/modules/vlm/QWen3.py:58-63` |
| **必须装 flash-attn** | 硬编码 `attn_implementation="flash_attention_2"` | 同上（`pip install flash-attn --no-build-isolation`，`README.md:70`） |
| `transformers==4.57.0` | 只有该版本提供 `Qwen3VLForConditionalGeneration` | `requirements.txt:1` |
| VLM 以 **bf16** 载入 | 硬编码 `dtype=torch.bfloat16` | `QWen3.py:58-63` |
| 世界模型路径存在 | 构造期 `from_pretrained(base_encoder)` | `VLA_JEPA.py:81-82` |

配置里另有三个字段**写了但代码从不读取**（改它们无效）：
`framework.qwenvl.attn_implementation`、`framework.qwenvl.vl_hidden_dim`、
`framework.reduce_in_full_precision`。`[已确认]`
（`grep -rn "vl_hidden_dim\|reduce_in_full_precision" --include=*.py starVLA/` 零命中）

### 2.5 起推理服务（websocket，最快的端到端验证路径）

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt \
  --port 10093 --use_bf16 --cuda 0
```

参数表 `[已确认]`（`deployment/model_server/server_policy.py:44-50`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt_path` | `Qwen/Qwen2.5-VL-3B-Instruct` | **必须显式传**，否则会当成 HF 模型名去载入并失败 |
| `--port` | `10093` | 监听端口；host 固定 `0.0.0.0` |
| `--use_bf16` | False | 打开后 `vla.to(torch.bfloat16)` |
| `--cuda` | `0` | `cuda:<n>` |

服务端行为：`from_pretrained(ckpt)` →（可选 bf16）→ `.to(device).eval()` →
`WebsocketPolicyServer(policy, host="0.0.0.0", port, metadata={"env": "simpler_env"})` → `serve_forever()`。`[已确认]`
（`server_policy.py:19-41`）

请求/响应格式与客户端实现见 [deployment.md](./deployment.md) §1.1、§2。

### 2.6 本地直接载入做一次推理

```python
import numpy as np
from PIL import Image
from starVLA.model.framework.base_framework import baseframework

ckpt = "checkpoints/<run_id>/final_model/pytorch_model.pt"
model = baseframework.from_pretrained(ckpt).cuda().eval()   # 内部已按 config.yaml 建好子模块

# 本地直调时传 PIL.Image（尺寸按训练 resolution_size，通常是 224）
img  = Image.open("head.png").convert("RGB").resize((224, 224))
wimg = Image.open("wrist.png").convert("RGB").resize((224, 224))

out = model.predict_action(
    batch_images=[[img, wimg]],          # [B][views] —— 视角顺序 = 训练时 video keys 顺序
    instructions=["pick up the block"],  # [B] str
    state=[np.zeros((1, 7), dtype=np.float32)],  # 可选，shape 必须是 (1, state_dim)
)
norm_actions = out["normalized_actions"]   # [B, future_action_window_size+1, action_dim]
```

**numpy 与 PIL 的区别**：本地直调必须自己保证是 PIL；只有走 websocket/ZMQ 协议时才可以传
`np.ndarray (H,W,3) uint8`，因为服务端会先调 `to_pil_preserve` 转换。`[已确认]`
（`deployment/model_server/tools/image_tools.py:61-119`）

`[已确认]`：签名（`VLA_JEPA.py:277-284`）、返回键（`VLA_JEPA.py:337`）、fake sample 写法（`:362-381`）。

---

## 3. 模型输入格式

### 3.1 推理接口签名

```python
@torch.inference_mode()
def predict_action(
    self,
    batch_images: List[List[Image.Image]],   # 外层 batch，内层多视角 PIL.Image
    instructions: List[str],                 # 每个样本一条任务指令
    state: Optional[np.ndarray] = None,      # 实际传 list，元素形状 (1, state_dim)
    **kwargs: str,                           # 多余参数被吞掉（见 §6 坑位 1）
) -> dict:
    # {"normalized_actions": np.ndarray[B, future_action_window_size+1, action_dim],
    #  "embodied_action_tokens": np.ndarray[B, 32, H]}
```

`[已确认]`（`VLA_JEPA.py:277-337`）

### 3.2 图像

| 项目 | 要求 | 证据 |
|---|---|---|
| 类型 | 协议传入 `np.ndarray (H,W,3) uint8`（服务端转 PIL）；本地直调传 PIL | `image_tools.py:61-119` |
| 结构 | `batch_images = [[view1, view2, ...]]`，内层是**多视角**，外层是 batch | `model2simpler_interface.py:134-135` |
| 视角顺序 | 训练时 `modality.json` 里 `video` 键的注册顺序。Piper = `[observation.images.image, observation.images.wrist_image]`（头相机在前、腕相机在后） | `data_config.py:787-790`、`eval_libero.py:189` |
| 尺寸 | 调用方 resize 到 224×224；框架内**不会**自动缩放（除非 config 有 `datasets.vla_data.image_size`，现有 Piper/LIBERO 配置都没有） | `VLA_JEPA.py:305-307` |
| 数值归一化 | 仓库内**没有** mean/std 常量、没有 `/255`、没有 ToTensor；归一化由 HF processor 内部完成 | `QWen3.py:146-153`、`VLA_JEPA.py:219-222` |
| 增强 | 生效的 robot 配置里没有任何图像增强/crop，训练=推理 | `data_config.py:839-855` |
| 特例 | LIBERO 环境图像需 **180° 旋转**（`[::-1, ::-1]`）以对齐训练预处理；真机不需要 | `eval_libero.py:159-163` |

### 3.3 语言指令与 prompt 模板

- 调用方传 `instructions=[<任务文本>]`，服务端不做额外处理。
- 模型内部把图像 + 指令拼成 Qwen3-VL 的 chat 消息（`{"type":"image"}` × N + `{"type":"text"}`），
  再 `processor.apply_chat_template(..., add_generation_prompt=True)`。`[已确认]`（`QWen3.py:107-153`）
- 文本模板取 `config.datasets.vla_data.CoT_prompt`，把 `{instruction}` 替换成任务文本，再把
  `{actions}` / `{e_actions}` 替换成动作 token 串（`predict_action` 传入 `prompt_replace_dict`）。`[已确认]`
  （`VLA_JEPA.py:310-313`、`QWen3.py:119-132`、`piper_test.yaml:57`）
- ⚠️ **指令必须按数据集逐任务复现**。当前 `adjust_cup_0409_1_offset_state_v2_1/meta/tasks.jsonl:1`
  的 `task_index=0` 原文是 `Put the cup the right way up on the table.`；不能把其他数据集的数字占位符
  或自然语言结论套到当前 checkpoint。`[已确认]`
  （当前数据集 `meta/tasks.jsonl`、`starVLA/dataloader/gr00t_lerobot/datasets.py`）

### 3.4 本体状态 `state`

- 形状：`state=[s]`，其中 `s` 形状 `(1, state_dim)`；多个样本时 `list` 长度 = batch。模型侧
  `torch.from_numpy(np.array(state))` → `[B, 1, state_dim]`，再由 `state_encoder`（MLP，`input_dim=state_dim`）编码。`[已确认]`
  （`VLA_JEPA.py:331-334`、`GR00T_ActionHeader.py:236-240`、`eval_openloop.py:175-184`）
- `state_dim` 必须等于 `config.framework.action_model.state_dim`：Piper `7`、LIBERO `8`（多一个 `pad`）。`[已确认]`
  （`piper_test.yaml:19-20`、`vlajepa_robot_ft.yaml:19-20`、`data_config.py:463-472,791-799`）
- **Piper 的 state 不归一化**：`PiperDataConfig` 的 transform 只对 action 做归一化，state 原样送入；
  Bridge/RT-1 才对 state 用 `q99`/`binary`。所以真机必须送**原始单位**的 7 维状态。`[已确认]`
  （`data_config.py:839-855` vs `:219-246`；另见 `scripts/analyze_openloop.py:43-45`）
- **语义**：Piper 的 7 维 = **6 关节角 + 夹爪**；`modality.json` 里的 `x/y/z/roll/pitch/yaw` 只是标签名，
  不是末端位姿。`[已确认]`（`doc/01_data_conversion.md:369-383`）
- 若 `state_dim` 为空/0，动作头不构造 `state_encoder`，`state=` 入参会被忽略。`[已确认]`（`GR00T_ActionHeader.py:236-240`）

### 3.5 推理期不需要的东西

- 世界模型前向（V-JEPA2 编码器 / predictor）**完全不参与推理**，`video` / `video_resolution_size`
  也只服务训练。`[已确认]`（`VLA_JEPA.py:277-337` vs `:215-247`）
- 但如 §1 所述，`base_encoder` 路径与 `vj_*` 权重**仍然必须存在**（构造 + strict 加载）。

---

## 4. 数据定义

### 4.1 数据集必须存在的东西

加载器只读 **LeRobot v2.1**（`starVLA/dataloader/gr00t_lerobot/datasets.py:58-64`）：

```
<dataset_root>/<data_name>/
├── meta/
│   ├── info.json          # data_path / video_path / chunks_size / fps / features
│   ├── episodes.jsonl
│   ├── tasks.jsonl        # 任务文本表（annotation 键查这里）
│   ├── modality.json      # ★ 字段切片定义（见 4.2）
│   └── stats_gr00t.json   # 归一化统计（缺失时自动计算并回写）
├── data/chunk_XXXXX/episode_XXXXXX.parquet
└── videos/<video_key>/chunk_XXXXX/episode_XXXXXX.mp4
```

Piper 数据（v3.0 → v2.1）由 `scripts/convert_v3_to_v2_1_aligned.py` 生成，其中 `meta/modality.json`
是脚本产出的（`:242-251`），键名固定为 `POSE_KEYS = ["x","y","z","roll","pitch","yaw","gripper"]`（`:32`）。

### 4.2 `meta/modality.json` 的 schema

顶层 4 个键（`annotation` 可选，其余必填）：`state` / `action` / `video` / `annotation`。`[已确认]`
（`starVLA/dataloader/gr00t_lerobot/schema.py:101-119`）

| 段落 | 每项字段 | 含义 |
|---|---|---|
| `state.*` / `action.*` | `start`, `end`（**必填**） | 在拼接后的向量里的切片区间，宽度 = `end - start` |
| | `rotation_type`, `absolute`(默认 true), `dtype`(默认 float64), `range`, `original_key` | 可选；仓库内 4 个示例都只用 `start`/`end` |
| `video.*` | `original_key` | 对应 LeRobot 的 `observation.images.*` 列 |
| `annotation.*` | `original_key` | 任务文本来源，示例统一为 `task_index` |

真实示例（LIBERO，`examples/LIBERO/modality.json:1-77` 节选）：

```json
{
  "state":  { "x": {"start":0,"end":1}, "...": {}, "pad": {"start":6,"end":7}, "gripper": {"start":7,"end":8} },
  "action": { "x": {"start":0,"end":1}, "...": {}, "gripper": {"start":6,"end":7} },
  "video":  { "primary_image": {"original_key":"observation.images.image"},
              "wrist_image":   {"original_key":"observation.images.wrist_image"} },
  "annotation": { "human.action.task_description": {"original_key": "task_index"} }
}
```

Bridge/SimplerEnv 示例见 `examples/SimplerEnv/train_files/bridge_modality.json:1-74`（单视角 `image_0`）。

### 4.3 `robot_type` → 实际送入模型的键名与维度

`robot_type` 在 `starVLA/dataloader/gr00t_lerobot/mixtures.py` 的 `DATASET_NAMED_MIXTURES` 随 `data_mix`
指定（如 `"piper_pick_place": [("", 1.0, "piper")]`，`:72-73`），它决定字段顺序与归一化方式
（`data_config.py:858-867`）：

| robot_type | state 维（顺序） | action 维 | 归一化模式 | 证据 |
|---|---|---|---|---|
| `piper` | 7：joint1..6, gripper（标签名 x,y,z,roll,pitch,yaw,gripper） | 7 | action 全部 `min_max`；**state 不归一化** | `data_config.py:786-855` |
| `libero_franka` | 8：x,y,z,roll,pitch,yaw,**pad**,gripper | 7 | action x/y/z/roll/pitch/yaw `min_max`（gripper 未列入） | `data_config.py:457-534` |
| `oxe_bridge` | 8：…,pad,gripper | 7 | 位姿 `q99`，gripper `binary` | `data_config.py:148-246` |
| `oxe_rt1` | 8：x,y,z,rx,ry,rz,rw,gripper | 7 | 位姿 `q99`，gripper `binary` | `data_config.py:266-380` |
| `droid_franka` | 8 | 7 | `min_max` + 旋转 `axis_angle` | `data_config.py:537-628` |

数据集样本（`DataLoader` 的一个元素，`collate_fn` 原样返回 list，**无 batch 维**）`[已确认]`
（`datasets.py:1596-1624`、`lerobot_datasets.py:10-11`）：

| 键 | 形状 | dtype | 备注 |
|---|---|---|---|
| `image` | `list[PIL.Image]`，长度 = 视角数（Piper/LIBERO 2） | PIL RGB，`resolution_size` 见方 | 每视角取视频第 0 帧 |
| `video` | `(V, T, video_resolution_size, video_resolution_size, 3)`，T=`num_frames`(8) | uint8 | 仅训练期世界模型用 |
| `lang` | `str` | str | 来自 `annotation` |
| `action` | `(action_horizon, action_dim)` = `(7, 7)` | float16（已归一化） | 训练标签 |
| `state` | `(1, state_dim)` = `(1,7)` | float16 | `with_state: true` 时才有 |

### 4.4 归一化统计量：`dataset_statistics.json`

- 结构：`{<unnorm_key>: {"action": {"mean","std","max","min","q01","q99","mask"}, "state": {...}, ...}}`。`[已确认]`
  （`datasets.py:1287-1294`、`:1960-1975`、`starVLA/dataloader/__init__.py:15-34`）
- `mask`：与 action 维度等长；**gripper 维为 `False`**，其余为 `True`（`datasets.py:1314-1346`）。
- `unnorm_key`：`dataset_statistics.json` 的顶层键。单数据集时可留空自动推断；多数据集（co-train 权重）
  **必须显式指定**。`[已确认]`（`base_framework.py:116-158`、`model2libero_interface.py:203-221`）
- 训练侧归一化实现：`min_max` → `2*(x-min)/(max-min)-1`；`q99` → `2*(x-q01)/(q99-q01)-1`；
  `binary` → `x>0.5`；均 clamp 到 `[-1,1]`。`[已确认]`（`transform/state_action.py:98-212`）

---

## 5. 模型输出格式

### 5.1 `predict_action` 返回

```python
{
  "normalized_actions": np.ndarray,      # [B, future_action_window_size+1, action_dim]，归一化空间，≈[-1,1]
  "embodied_action_tokens": np.ndarray,  # [B, 32, hidden]（Qwen 隐状态，调试用）
}
```

`[已确认]`（`VLA_JEPA.py:336-337`）动作头内部是真值 flow-matching 欧拉积分
（`actions = randn(...); for t in range(num_inference_timesteps): actions += dt * v`），
随机初值意味着**同一输入两次调用结果不同**。`[已确认]`
（`GR00T_ActionHeader.py:320-369`；复现性说明见 `scripts/eval_openloop.py:66-75`）

### 5.2 反归一化（调用方必做）

两条公式，取决于 `dataset_statistics.json` 里有哪些键：

```python
# A. q01/q99 形式（框架自带，见 base_framework.unnormalize_actions）
mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
hi, lo = np.array(stats["q99"]), np.array(stats["q01"])
a = np.clip(normalized, -1, 1)
a[:, 6] = np.where(a[:, 6] < 0.5, 0, 1)          # 第 6 维（gripper）二值化
actions = np.where(mask, 0.5 * (a + 1) * (hi - lo) + lo, a)

# B. min/max 形式
mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
hi, lo = np.array(stats["max"]), np.array(stats["min"])
...  # 其余同上
```

`[已确认]`：A 见 `starVLA/model/framework/base_framework.py:174-205`、`model2simpler_interface.py:205-217`；
B 见 `examples/LIBERO/model2libero_interface.py:135-147`；等价写法 `(x+1)/2*(hi-lo)+lo` 见
`scripts/eval_openloop.py:161-162`。

⚠️ 两条公式**不可混用**：Piper/LIBERO 训练归一化是 `min_max`，Bridge/RT-1 是 `q99`。
**先确认权重是哪套统计量，再选公式。**

### 5.3 gripper（第 6 维）是被硬编码特殊处理的

两份客户端都无条件执行 `a[:, 6] = np.where(a[:, 6] < 0.5, 0, 1)`，把该维压成 0/1；
`dataset_statistics.json` 里 `mask` 对 gripper 维也是 `False`
（`generate_action_mask_for_used_keys`：命中 "gripper" 的维 mask=False）。
对 Piper 这种"gripper 也用 `min_max` 归一化"的配置，部署侧输出的 gripper 语义与训练标签不完全一致
（训练标签连续，部署被二值化）。**真机接入前必须确认 Piper 夹爪期望什么量纲。** `[已确认]`（代码事实）
+ `[待确认]`（夹爪真实控制语义）
（`model2libero_interface.py:140`、`model2simpler_interface.py:210`、`base_framework.py:198`、
`datasets.py:1314-1346`）

当前 `adjust_cup_0409_1_offset_state_v2_1` 的 parquet 还显示：原始 gripper action 范围为
`[0, 1.4228571653]`，共有约 696 个离散采样值。由于 checkpoint 的 `mask[6]=False`，默认二值化
路径实际下发 `0/1`；`scripts/piper_zmq_replay.py --no-binarize-gripper` 才会按 Piper 的
`min/max` 统计量还原连续物理值。`[已确认]`

### 5.4 动作 chunk 语义

- chunk 长度 = `future_action_window_size + 1`（Piper 配置 = 7），对应**未来 7 帧**的绝对/增量动作。
- **动作语义由训练配置决定**：Piper 的 8 个 ICLR 任务全部 `action_type: absolute`（绝对关节角 + 夹爪）；
  LIBERO/co-train 用 `delta_qpos`。下发前必须与执行侧对齐（真机侧见
  [deployment.md](./deployment.md) §4.2）。`[已确认]`（`scripts/config/iclr_*.yaml:56`、`piper_test.yaml:56`）
- 执行策略（每 N 步推理一次 / 每步都推理 / 是否做 ensemble）属部署侧行为，见
  [deployment.md](./deployment.md) §1.1、§3.4。

---

## 6. 模型侧坑位速查

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | 调 `predict_action(..., num_ddim_steps=N)` 完全没用 | `predict_action` 用 `**kwargs` 吞掉未知参数 | 改 `model.action_model.num_inference_timesteps`；`eval_openloop.py:92-96` 已这么做 |
| 2 | 反归一化后动作尺度明显不对 | 权重统计量是 `min/max`，代码却按 `q01/q99`（或反之） | 先 `cat <run_dir>/dataset_statistics.json` 确认键名，再选 §5.2 的 A/B 公式 |
| 3 | `Missing 'config.yaml' for run_dir=...` | `.pt` 不在 run 目录下两层，或缺 config.yaml | 保持 `run_dir/checkpoints/xxx.pt` 或 `run_dir/final_model/pytorch_model.pt` 布局 |
| 4 | 载入时报找不到 `base_vlm` / `base_encoder` | config.yaml 里是训练机绝对路径 | 在运行机器上改这两个字段（`README.md:142-145`） |
| 5 | `state` 形状报错 | 传成 `(7,)` 而非 `(1,7)`，或 `state_dim` 与数据不符 | 传 `state=[s.reshape(1, D)]`；核对 `action_model.state_dim` |
| 6 | 两次推理结果不同 | 动作头从 `randn` 起采样 | 属正常；需要复现就固定随机种子（`eval_openloop.py:66-75`） |
| 7 | README 里的 `framework.qwenvl.basevlm` 改了没生效 | 真实键名是 `base_vlm` | 以 `config.yaml` / `vlm/__init__.py:4` 为准 |
| 8 | `model.get_action_stats("xxx")` 抛 `AttributeError: type object ... has no attribute 'norm_stats'` | 该方法写成 `@classmethod` 却访问实例属性 `self.norm_stats`（`base_framework.py:146-158,227-236`） | 改用 `read_mode_config(ckpt)` + 静态方法 `baseframework.unnormalize_actions(...)` |
| 9 | 报 `NotImplementedError: Framework QwenFM/QwenJEVLA is not implemented` | 两份配置写了未注册的框架名：`starVLA/config/training/starvla_cotrain_oxe.yaml:10`（`QwenFM`）、`examples/SimplerEnv/train_files/vlajepa_ft.yaml:8`（`QwenJEVLA`）；已注册名只有 `InternVLA-M1 / Qwen-Dual / QwenFast / QwenGR00T / QwenOFT / QwenPI / VLA_JEPA` | 把 `framework.name` 改成 `VLA_JEPA` |
| 10 | 导入 starVLA 时看到 `AttributeError: 'PureOverwatch' object has no attribute 'log'`，掩盖真实错误 | `framework/__init__.py:32-33` 的兜底 except 调用了不存在的 `logger.log()` | 真实原因通常是依赖不全（如 transformers 版本过低缺 `Qwen3VLForConditionalGeneration`）；按 `requirements.txt` 装齐 |
| 11 | 夹爪输出只有 0/1，但训练标签是连续值 | 反归一化硬编码二值化第 6 维，且 `mask` 对 gripper 维为 `False` | 见 §5.3；真机接入前确认夹爪量纲 |

配置里"写了但没人读"的字段（改了不会有任何效果）：
`framework.qwenvl.attn_implementation`、`framework.qwenvl.vl_hidden_dim`、
`framework.reduce_in_full_precision`。`[已确认]`

---

## 7. 模型侧关键文件索引

| 用途 | 文件 |
|---|---|
| 载入框架（from_pretrained / 反归一化） | `starVLA/model/framework/base_framework.py` |
| run 目录解析（config + stats） | `starVLA/model/framework/share_tools.py:203-290`、`starVLA/model/tools.py:156-195` |
| 模型注册表与工厂 | `starVLA/model/framework/__init__.py`、`starVLA/model/tools.py:118-143` |
| VLA-JEPA 模型与推理 | `starVLA/model/framework/VLA_JEPA.py` |
| 动作头（flow matching） | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |
| VLM 输入构造 | `starVLA/model/modules/vlm/QWen3.py`、`starVLA/model/modules/vlm/__init__.py` |
| 世界模型（仅训练期） | `starVLA/model/modules/world_model/` |
| 数据管线 / modality / 统计量 | `starVLA/dataloader/gr00t_lerobot/{datasets.py,data_config.py,mixtures.py,schema.py}` |
| 归一化实现 | `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` |
| 逐任务训练配置（含 action_type/维度） | `scripts/config/*.yaml` |
| 数据处理与语义说明 | `doc/01_data_conversion.md`（§6.3 关节角语义、§6.4 指令占位符） |

---

## 附：相关文档

| 文档 | 内容 |
|---|---|
| [deployment.md](./deployment.md) | 把模型跑起来：websocket 服务、仿真评测（LIBERO / LIBERO-Plus / SimplerEnv）、真机 Piper 部署 |
| [piper_server_plan.md](./piper_server_plan.md) | 面向控制端 `vla_infer` 的 ZMQ 协议理解与 VLA-JEPA server 开发计划 |
| `doc/01_data_conversion.md` | v3 → v2.1 数据转换与 robot_type 注册 |
| `doc/02_training.md` | 训练启动与排错 |
| `doc/03_openloop_testing.md` | 开环测试与指标解读 |
