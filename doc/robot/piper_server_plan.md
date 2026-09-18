# Piper 控制端通信协议理解 + VLA-JEPA Server 开发计划

> 阶段：**第二步（协议实现 + 真实权重离线验证）**。协议样例、真实权重离线回放已完成；
> 真机控制端联调、动作安全验证和任务评测仍未完成。
> 本文持续记录两件事：
> ① 把 `vla_infer`（机器人控制端）的 client-server 协议从两侧代码里读清楚；
> ② 给出并验证面向该协议的 VLA-JEPA server 实施方案与验收标准。
>
> 证据来源（全部只读核对）：
> - 服务端侧：`~/repo/Double_Piper_Teleop/vla_infer/src/zmq/`、`src/inference/server.py`、`src/models/base.py`、`src/models/dream_adapter_model.py`、`example/dream-adapter/dream-adapter_server.py`
> - 客户端侧：`example/dream-adapter/dream-adapter-piper_client.py`、`src/inference/piper_client.py`、`src/robots/piper_single.py`、`src/process/utils.py`
> - 模型侧（本仓库）：`deployment/model_server/`、`starVLA/model/framework/VLA_JEPA.py`、`scripts/config/iclr_*.yaml`、`doc/01_data_conversion.md`
>
> 标记约定：`[已确认]` 有代码原文（给出 `文件:行`）；`[建议]` 工程做法；`[待确认]` 仓库内无证据，需实测或向控制端/数据侧确认。

---

## 0. 结论速览（一段话）

控制端 `vla_infer` 的协议是 **ZMQ REQ/REP + msgpack(+msgpack_numpy) + 按 key 名启发式 JPEG 压缩图像** 的同步请求-响应：
客户端每个控制周期发一次 `{state, image, wrist_image, cmd}`，同步等回 `{"action": (T, D)}`，
超时（默认 2000 ms）即抛 `TimeoutError` 并按 `stop_on_timeout` 停机。
**服务端只需实现"收观测 → 返回动作块"这一个契约**，模型如何加载/推理由服务端内部决定——
因此 VLA-JEPA 的 server 应当写在**本仓库**（模型与依赖都在这边），只依赖 `pyzmq/msgpack/msgpack-numpy/Pillow`，
并且**必须逐字节匹配它的线格式**（注意本仓库现有的 `deployment/model_server/tools/msgpack_numpy.py` 是另一种不兼容格式，见 §4-G8）。

---

## 1. 协议理解（服务端视角）

### 1.1 传输层：ZMQ REQ/REP，一问一答

| 项 | 事实 | 证据 |
|---|---|---|
| 模式 | `zmq.REP`（服务端）/ `zmq.REQ`（客户端），严格同步的一问一答 | `src/zmq/zmq_server.py:28`、`src/zmq/zmq_client.py:21` |
| 地址 | `tcp://{ip}:{port}`；库默认 `127.0.0.1:5555`；示例 server 默认 `ip="0.0.0.0", port=5555` | `zmq_server.py:26-30`、`example/dream-adapter/dream-adapter_server.py:46-47` |
| 关闭语义 | 两端都设 `LINGER=0`，`close()` 立即销毁 socket，不等残留消息 | `zmq_server.py:29`、`zmq_client.py:24` |
| 超时 | 客户端设 `RCVTIMEO=timeout_ms`（默认 2000）；超时抛 `zmq.error.Again` → 记日志 "ZMQ Timeout! Emergency stop required." → `raise TimeoutError` | `zmq_client.py:19,26,54-60` |
| 并发 | 单 socket 串行；无 request id、无心跳、无重试、无鉴权 | 通篇无相关字段（`protocol.py` 仅打包/解包） |
| 服务端循环 | `while True: recv → model.predict → send`，`KeyboardInterrupt`/异常时 `close()` | `src/inference/server.py:35-52` |

`[已确认]` 直接后果：
1. **服务端必须"永不抛异常"**——一旦 `predict()` 抛异常，REP socket 不再回包，客户端只能在 2 s 后超时急停（`server.py:44-46` 会 `raise` 并结束进程）。`[建议]` 在 server 内层兜底，异常时返回一个安全的"保持当前位姿"块或明确错误，并在外层用 supervisor 重启。
2. **单客户端**：多进程同时连会互相抢消息（REP 无路由）。当前架构一台机器人一个 server。`[已确认]`

### 1.2 序列化层：msgpack + 图像启发式压缩

打包（**客户端与服务端共用同一套逻辑**，`protocol.pack_payload`）：

```python
# src/zmq/protocol.py:49-69
for key, value in payload.items():
    if isinstance(value, np.ndarray):
        if value.ndim == 3 and value.dtype == np.uint8:      # ← 启发式：3D uint8 一律当图像
            processed_payload[key] = cls.encode_image(value, quality=jpeg_quality)   # JPEG bytes
        elif value.dtype == np.float64:
            processed_payload[key] = value.astype(np.float32)                        # 降精度
        else:
            processed_payload[key] = value
    else:
        processed_payload[key] = value                        # str/bool/int 原样
return msgpack.packb(processed_payload, use_bin_type=True)
```

解包：

```python
# src/zmq/protocol.py:72-85
unpacked = msgpack.unpackb(payload_bytes, raw=False)
for key, value in unpacked.items():
    if isinstance(value, bytes) and ('img' in key.lower() or 'image' in key.lower()):
        unpacked[key] = cls.decode_image(value)               # ← 只看 key 名，不看内容
return unpacked
```

要点与陷阱 `[已确认]`：

| # | 事实 | 影响 |
|---|---|---|
| a | `msgpack_numpy.patch()` 在模块 import 时全局生效（`protocol.py:20`），因此非图像的 ndarray 走 **msgpack-numpy 的 `b'nd'` 扩展格式**（dict：`nd/type/kind/shape/data`） | 服务端必须使用**真正的 `msgpack_numpy` 包**，不能用手写扩展（本仓库现有那个是 `__ndarray__`，不兼容，见 §4-G8） |
| b | 图像判定是 **"3 维 + uint8"**，与 key 名无关；解包时才按 key 名（含 `img`/`image`）还原 | 服务端返回值里若放 3D uint8 且 key 不含 image，客户端收到的是 **原始 JPEG bytes**，不是数组 |
| c | `float64` 一律降为 `float32` | `state` 到服务端是 float32；服务端回包也应给 float32（避免半精度/整型歧义） |
| d | 打包用 `use_bin_type=True`、解包用 `raw=False` | str 键/值保持 str；`bytes` 才是 JPEG |
| e | 无 schema、无版本号、无 request id | 键名就是唯一契约，改动即破坏兼容 |
| f | `encode_image` 用 `Image.fromarray` | 输入必须是 HWC、uint8、且能构造 PIL（非连续/负 stride 需先 `copy()`，客户端已保证） |

### 1.3 服务端运行时组成

```
example/dream-adapter/dream-adapter_server.py      # 入口：draccus 配置 → 建组件 → start()
  ├─ VlaZmqServer(ip, port, jpeg_quality)          # 传输层（REP）
  ├─ <SomeModel>(...)                              # 模型适配层，实现 BaseVLAModel
  └─ ModelZmqInferenceServer(model, zmq_server)    # 循环：recv → model.predict → send
```

- `ModelZmqInferenceServer.predict()` 要求 `model.predict(request)` **必须返回 dict**，否则 `TypeError`。`[已确认]`（`src/inference/server.py:20-25`）
- `ModelZmqInferenceServer.start()` 是无休止循环，仅在 `KeyboardInterrupt` 时优雅退出，其它异常向上抛。`[已确认]`（`:35-48`）

### 1.4 模型侧接口契约（`BaseVLAModel`）

```python
# src/models/base.py:15-40
class BaseVLAModel(ABC):
    def __init__(self): self.load_model()                      # 构造即加载
    @abstractmethod
    def load_model(self) -> None: ...                          # 建议末尾做一次 dummy forward 预热
    @abstractmethod
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]: ...   # 必须含 "action"
```

所有现有适配器（dream_adapter / vla_adapter / OpenVLA_OFT / smolvla）读取的观测键**完全一致** `[已确认]`：

| 请求键 | 类型/形状 | 必需性 | 说明 |
|---|---|---|---|
| `image` | `np.ndarray (H,W,3) uint8` | 必需 | 主（头部）相机，客户端已 resize 到 224×224 |
| `wrist_image` | `np.ndarray (H,W,3) uint8` | 必需 | 腕部相机 |
| `state` | `np.ndarray (7,) float32` | 必需 | 6 关节 + 夹爪（见 §3.4 语义核对） |
| `cmd` | `str` | 必需（缺省用 server 的 `default_instruction`） | 任务指令 |
| `return_action_chunk` | `bool` | 可选（`True`=返回 chunk） | 仅部分适配器支持；VLA-JEPA 天然返回 chunk，建议同样支持该开关 |

响应：`{"action": np.ndarray (T, D) float32}`，`D = 7`、`T` = chunk 长度。`[已确认]`（`src/models/dream_adapter_model.py:238-250`）

适配器内部还有两个值得照抄的健壮性处理：
- `_validate_rgb_image`：要求 `ndim==3 and shape[-1]==3`，**负 stride 时 `copy()`**；`[已确认]`（`:115-123`）
- `_validate_state`：`np.asarray(..., float32).reshape(-1)` 后**严格校验长度 == proprio_dim**；`[已确认]`（`:125-130`）
- `_to_action_array`：把 `(T,1,D)` 压成 `(T,D)`、`(D,)` 升成 `(1,D)`，最终必须是 2D。`[已确认]`（`:132-141`）

### 1.5 文档与代码的冲突（以代码为准）

`vla_infer/README.md:62-67` 写请求字段是 `instruction`，但**两侧代码用的都是 `cmd`**
（`dream-adapter-piper_client.py:156`、`dream_adapter_model.py:203`、`smolvla_model.py:284`）。
⇒ 我们的 server 必须读 `cmd`；为兼容起见可同时接受 `instruction` 作为 fallback。`[已确认]` + `[建议]`

---

## 2. 协议理解（机器人客户端视角，用于核对）

### 2.1 一个控制周期的完整链路

```
PiperSingleRobot.get_observation()            # cam_head/cam_wrist/state(7)/joint(6)/qpos(6)/gripper(1)
  → PiperVLAClient.get_observation()          # 组 state（按 state_type） + 图像 adaptive_resize(224) + check_uint8_rgb
  → get_response()                            # 追加 cmd，VlaZmqClient.get_response → 阻塞等回包，取 ["action"]
  → execute()                                 # 后处理链：delta→absolute / 平滑 / 夹爪变换 / 插值
  → PiperSingleRobot.apply_action()           # {"arm":{"left_arm":{"joint":[6],"gripper":float}}} → robot.move
  → sleep(control_interval_s)                 # 默认 0.04 s
```

`[已确认]`（`example/dream-adapter/dream-adapter-piper_client.py:240-250,252-291`、`src/robots/piper_single.py:227-239`）

### 2.2 观测构造细节

```python
# example/dream-adapter/dream-adapter-piper_client.py:107-142
if cfg.state_type == "qpos":        # 默认
    state = concat(raw["qpos"](6), [raw["gripper"]](1))       # ← qpos 字段，非 joint
elif cfg.state_type == "joint":
    state = raw["state"](7)                                    # = joint(6) + gripper(1)
obs = {"state": state, "image": cam_head, "wrist_image": cam_wrist}
obs["image"]        = check_uint8_rgb(adaptive_resize_image(obs["image"]))        # 224×224 letterbox
obs["wrist_image"]  = check_uint8_rgb(adaptive_resize_image(obs["wrist_image"]))
if cfg.action_type == "joint" and cfg.state_type == "qpos":
    self.obs["joint_state"] = raw["state"](7)                  # 供 delta→absolute 用
```

- `adaptive_resize_image` = **保持长宽比 + 白边（pad_value=255）缩放到 224×224**（`ImageOps.pad` 风格，bilinear）。`[已确认]`（`src/process/utils.py:214-227,183-211`）
- 底部注释掉的 `ensure_hwc3_uint8_image` 说明这条"图像规范化"路径历史上改过，最终生效的是 `check_uint8_rgb(adaptive_resize_image(...))`。`[已确认]`（`...piper_client.py:123-128`）
- **观测里没有 `prev_action`**（类 docstring 提到 optional，但本示例不发送）。`[已确认]`

### 2.3 响应消费与后处理链（**决定 server 该输出什么动作空间**）

`execute()` 的默认配置（`enable_binary_gripper=False`，`state_type="qpos"`，`action_type="joint"`，`absolute_action=True`）：

```python
# ...piper_client.py:179-190
if not enable_binary_gripper:
    if state_type == "qpos":                     # ← 默认走这条
        abs_action = delta_action_chunk_to_absolute(self.obs["joint_state"], action)   # cumsum + 当前关节
    elif state_type == "joint":
        abs_action = action if absolute_action else delta_action_chunk_to_absolute(self.obs["state"], action)
    smooth_action = smooth_action_chunk(abs_action, ...) if use_smoothing else abs_action
```

关键事实 `[已确认]`：

| 配置 | 客户端对 `action` 的解释 | 证据 |
|---|---|---|
| `state_type="qpos"`（**默认**） | **增量**：`absolute = joint_state + cumsum(action)`，与 `absolute_action` 无关 | `:180-181` |
| `state_type="joint"` + `absolute_action=True` | **绝对**：原样下发 | `:183-184` |
| `state_type="joint"` + `absolute_action=False` | 增量：`state + cumsum(action)` | `:185-186` |
| `enable_binary_gripper=True` | 前 6 维仍做 delta→absolute，第 7 维按阈值映射到 `gripper_open_value/closed_value` | `:194-199,169-176` |

其它后处理默认值：`use_smoothing=False`（开启后：滑动平均窗 3 + EMA 0.35 + 可选角加速度/加加速度限制）、
`enable_action_interpolation=False`（开启后可插值到 `interpolation_target_steps`）、
**`enable_gripper_transform=True`（默认开启！）**：`smooth_action[:, -1] -= 0.3 * (action[:, -1] < 0.55)`。`[已确认]`（`:187-212`、`src/process/utils.py:534-553`）

执行阶段：`execute_steps = min(max(1, execute_chunk_steps), T)`，逐帧 `apply_action` 并 `sleep(0.04)`；
`apply_action` 只取第 0 行、要求 `dim >= 7`，把 `[0:6]` 当关节角、`[6]` 当夹爪，直接**绝对位置**下发。`[已确认]`（`...piper_client.py:222-238`、`piper_single.py:197-213,227-239`）

### 2.4 两侧一致性核对表

| 项 | 客户端发送/期望 | 服务端应满足 | 状态 |
|---|---|---|---|
| 传输 | `zmq.REQ` → `tcp://ip:port`，默认 5555 | `zmq.REP` bind 同端口 | `[已确认]` 一致 |
| 序列化 | msgpack + `msgpack_numpy.patch()` + JPEG(3D uint8) | 用真 `msgpack_numpy`，同 JPEG 规则 | `[已确认]` 需自建（本仓库现有实现不兼容） |
| 图像键 | `image` / `wrist_image`（HWC3 uint8，224×224 letterbox） | 原样转 PIL 送模型 | `[已确认]` 键名一致 |
| 状态键 | `state`（7 维 float32） | 转 `(1,7)` 送 `predict_action` | 语义待核对，见 §4-G2 |
| 指令键 | `cmd`（README 写 `instruction`） | 读 `cmd`，兼容 `instruction` | `[已确认]` 以 `cmd` 为准 |
| 响应键 | `["action"]`，`(T,7)` float32 | `{"action": chunk}` | `[已确认]` 一致 |
| 动作语义 | 默认按**增量**解释 | 需与训练动作空间匹配 | **不一致风险**，见 §4-G3 |
| chunk 长度 | `execute_chunk_steps=8`，取 `min(8, T)` | VLA-JEPA 输出 T=7（`future_action_window_size+1`） | 可运行但需显式对齐，见 §4-G5 |
| 夹爪 | 默认 `enable_gripper_transform` 会再减 0.3 | 我们输出 0/1 | **风险**，见 §4-G4 |
| 时延 | 2000 ms 超时 → 急停 | 单周期推理必须 < 2 s（含首帧预热） | 见 §4-G7 |

---

## 3. VLA-JEPA 侧现状（作为映射依据）

| 项 | 事实 | 证据 |
|---|---|---|
| 现有 server | websocket + msgpack（`__ndarray__` 格式），与本协议**不兼容** | `deployment/model_server/server_policy.py`、`tools/msgpack_numpy.py:43-45` |
| 推理接口 | `predict_action(batch_images=List[List[PIL]], instructions=List[str], state=可选 (B,1,D))` → `{"normalized_actions": (B,T,D)}` | `starVLA/model/framework/VLA_JEPA.py:277-337` |
| 输出 | **归一化**动作，必须用 `dataset_statistics.json` 反归一化 | `base_framework.py:174-205`、`examples/*/model2*_interface.py` |
| Piper 训练配置 | `action_dim=7, state_dim=7, future_action_window_size=6, action_horizon=7, num_inference_timesteps=4`，`action_type: absolute` | `scripts/config/piper_test.yaml:19-23,56`、`scripts/config/iclr_*.yaml:19-20,56` |
| Piper 数据语义 | **6 关节角 + 夹爪**；`modality.json` 里的 `x/y/z/roll/pitch/yaw` 只是标签名，不是末端位姿 | `doc/01_data_conversion.md:369-383` |
| 语言条件 | 当前 `adjust_cup` 数据 `meta/tasks.jsonl:1` 是自然语言原文；部署必须传与 checkpoint 对应的 task 文本 | 当前数据集 `meta/tasks.jsonl` |
| 服务器依赖 | `pyzmq`、`msgpack`、`msgpack-numpy`、`Pillow` 已在项目 `pyproject.toml` 中声明；机器人端仍需使用自己的 `vla_infer` 环境 | `pyproject.toml`、`examples/real-robot/wire.py` |

---

## 4. 差异与风险清单（Gap List）

### G1 语言指令：必须复现当前 checkpoint 的任务文本 `[已确认]`
当前 `adjust_cup_0409_1_offset_state_v2_1/meta/tasks.jsonl:1` 的 `task_index=0` 原文是
`Put the cup the right way up on the table.`，不是数字占位符；本次离线回放直接使用样本中的 `lang`。
⇒ 真机部署必须把对应 `task_index` 的原文作为 `cmd` 发送；换数据集或 checkpoint 时不得沿用本例文本。
**验证方法**：读取 `<dataset>/meta/tasks.jsonl`，把对应 `task_index` 的 `task` 原样填进客户端 `task_instruction`。

### G2 状态语义：训练=关节角，客户端默认发的是 `qpos` `[已确认]+[待确认]`
- 训练：`observation.state[:6]` = 6 关节角（`doc/01_data_conversion.md:369-383`），`state_dim=7`。
- 客户端默认 `state_type="qpos"` 发送 `raw["qpos"](6) + gripper`，而 `piper_single.py` 的文档里
  `qpos` 与 `joint` 是**两个不同字段**（`:107-113` 注释把 `qpos` 写作"关节速度"，顶层注释里 `qpos` 又像标量）。
⇒ **必须把客户端切到 `state_type="joint"`**（发 `joint(6)+gripper`），否则喂给模型的第 4-7 维"关节角"是错的东西。`[建议]`
**待确认**：`offset_state` 数据集名暗示 state 做过**偏移/去零**处理（仓库内无文档说明）。
若训练数据是"相对某次初始位姿的偏移量"，则部署时必须复现同一偏移，否则 state 分布不一致。
**验证方法**：读一条 episode 的 parquet，打印 `observation.state` 的首尾若干行，与真机 `get_observation()["joint"]` 对比量级与数值。

### G3 动作空间：训练 `absolute`，客户端默认按 `delta` 解释 `[已确认]`
- 所有 Piper 任务配置都是 `action_type: absolute`（`scripts/config/iclr_*.yaml:56`），且模型输出反归一化后即
  **绝对关节角 + gripper**。
- 客户端默认 `state_type="qpos"` 时会 `absolute = joint_state + cumsum(action)`，把绝对量当增量累加 → 动作会被**不断叠加**，非常危险。
⇒ **必须把客户端配成 `state_type="joint"` + `absolute_action=True`**（此时原样下发）。
`[建议]` 另一种做法是 server 端把绝对动作转成增量（`delta = absolute - current_state`，利用请求里已有的 `state`），
但那样依赖 client 的 cumsum 与 state 时序严格一致，**不推荐**；若确有需要，务必加一层坐标/单位核对。

### G4 夹爪：客户端默认会"再关一点" `[已确认]`
server 默认反归一化后第 7 维是 `0/1`（客户端/框架硬编码二值化，见 [`model.md`](./model.md) §5.3）；
对当前 Piper 数据集，server 使用 `--no-binarize-gripper` 时会按 `min/max` 统计量输出连续物理值；
该路径已用 50 episode × 2 窗口离线回放验证，归一化 MAE=`0.0381`，gripper 维 MAE=`0.00153`。
而客户端默认 `enable_gripper_transform=True`，对 `< 0.55` 的值再减 `0.3` → `0 → -0.3`，可能被下位机解释为非法或过度闭合。
⇒ **必须二选一**：`enable_gripper_transform=False`（二值路径直接下发 0/1 或连续路径原样下发），或 `enable_binary_gripper=True` +
`gripper_open_value/closed_value`（映射到真机标定值，并走 `:194-199` 分支）。`[建议]`
**待确认**：Piper 夹爪指令的合法区间与"张开/闭合"方向（`piper_single.apply_action` 只把 float 透传给 `move`，未做标定）。

### G5 chunk 长度与执行步数不匹配 `[已确认]`
VLA-JEPA 输出 `T=7`；客户端 `execute_chunk_steps=8`，实际执行 `min(8,7)=7` 步。
⇒ 可运行，但建议显式设 `execute_chunk_steps: 7`，避免"以为执行了 8 步"的误判；同时确认 server 是否需要补零到 8（**不要**，补零会把关节零位当目标）。

### G6 图像预处理方式不同 `[待确认]`
训练侧是 `Image.fromarray(frame).resize((224,224))`（**直接拉伸**，`starVLA/dataloader/gr00t_lerobot/datasets.py:1605`），
客户端是 `adaptive_resize_image`（**保持长宽比 + 白边** letterbox，`src/process/utils.py:214-227`）。
⇒ 同一张图两种预处理后的像素分布不同；再加上 JPEG quality=80 的有损压缩（训练侧是 mp4/h264 解码，同样有损）。
`[建议]` 先用两种方式各跑离线回放对比动作差异；若差异明显，在客户端把 resize 换成直接拉伸（或让 server 侧统一做 letterbox 的反变换）。

### G7 时延预算 `[待确认]`
客户端 2000 ms 超时即急停。VLA-JEPA 单次推理 = 1 次 Qwen3-VL 前向 + 4 步 flow-matching + JPEG 编解码 + 网络。
⇒ `[建议]` ① server 载入后**必须预热**（`BaseVLAModel.load_model` 的注释也这么建议）；② 实测 P95 时延，
若接近 2 s 则调大 `timeout_ms` 或降 `num_inference_timesteps`；③ 记录每周期时延。

### G8 依赖与线格式不兼容（最容易踩的坑）`[已确认]`
- 本仓库 `deployment/model_server/tools/msgpack_numpy.py` 用的是 `b"__ndarray__"` dict 格式（`:43-45`），
  而 `vla_infer` 用的是真 `msgpack_numpy` 的 `b"nd"` 扩展（`msgpack_numpy-0.4.8`，键为 `nd/type/kind/shape/data`）。
  **两种格式互不识别** → 直接复用本仓库那个模块会得到 `state` 解不出来或整体解析失败。
- 依赖版本冲突：`vla_infer` 要求 `numpy>=2.0.0,<2.3.0`（`vla_infer/pyproject.toml:12`），本仓库钉 `numpy==1.26.4`
  （`requirements.txt:25`，README 又写 `1.24.4`）。
⇒ `[建议]` **不要把 `vla_infer` 装进 VLA 环境**；server 只依赖 `pyzmq / msgpack / msgpack-numpy / Pillow`（+ 已有 torch 系）。
互操作测试时用两个环境：server 在 VLA 环境，客户端校验脚本在机器人环境（正好也是真机拓扑）。

### G9 响应里不要塞非 image 的 3D uint8 数组 `[已确认]`
按 §1.2(b)，这种值到客户端会变成 bytes。若以后要回传重建图像，键名必须含 `image`/`img`。

### G10 无 request id / 无心跳 / 无鉴权 `[已确认]`
出问题只能靠客户端超时发现。`[建议]` server 侧自己打 step 计数与 latency 日志；如需更高的可观测性，
可在**不破坏现有键**的前提下额外返回 `debug_*` 字段（客户端只读 `action`，多余键被忽略）。

### G11 `state_type="qpos"` 且 `action_type != "joint"` 时 `joint_state` 缺失 `[已确认]`
`joint_state` 只在 `action_type == "joint" and state_type == "qpos"` 时保存；否则
`self.obs.get("joint_state", zeros(7))` 会退化成**全零基准**做 cumsum。`[建议]` 保持 `action_type: joint` 不动。

---

## 5. 开发计划：面向该协议的 VLA-JEPA Server

### 5.1 目标与非目标

**目标**：在 VLA-JEPA-Alex 仓库内提供一个 ZMQ REP server，使 `vla_infer` 的**现有** Piper 客户端
（不改客户端代码，只改配置）能直接连上并拿到可执行的动作块。

**非目标（本期不做）**：多客户端并发、鉴权、加密、模型热切换、TensorRT/量化加速、把 `vla_infer` 并入本仓库。

### 5.2 放置位置与目录结构

✅ **已按本节落地**（目录名改为 `examples/real-robot/`，其余一一对应；见
[`examples/real-robot/README.md`](../examples/real-robot/README.md)）：

```
examples/real-robot/                 # 规划时写作 deployment/piper_zmq/，实际落在 examples/ 下
├── wire.py              # 与 vla_infer.src.zmq.protocol 线格式一致的编解码（使用真 msgpack_numpy）
├── transport.py         # ZmqRepServer：REP 绑定 / LINGER=0 / jpeg_quality / 单请求处理 / 优雅关闭
├── policy.py            # VLAJepaPiperPolicy + DryRunPolicy：加载 + 预热 + obs→predict_action + 反归一化
├── piper_zmq_server.py  # 入口：argparse 配置 + 主循环 + 异常兜底（规划中的 server.py）
├── selftest_client.py   # 协议自检客户端（规划中的 piper_zmq_selftest.py，改为 REQ 客户端形态）
└── README.md            # 用法、参数表、已验证/未验证清单
```

离线回放入口位于 `scripts/piper_zmq_replay.py`：它不打开 ZMQ 端口，使用真实 checkpoint
复现客户端的 letterbox + JPEG/msgpack 观测边界，再调用同一个 `VLAJepaPiperPolicy`。

`[建议]` 不放进 `deployment/model_server/`（那是 websocket 协议族），避免两套协议混住。

### 5.3 模块设计

**`wire.py`**（约 60 行）
- `import msgpack, msgpack_numpy as mnp; mnp.patch()`（与对端一致）
- `pack_payload(dict, jpeg_quality=80) -> bytes` / `unpack_payload(bytes) -> dict`，
  **逐条复刻** §1.2 的三条规则（3D-uint8 → JPEG；float64 → float32；解包按 key 名还原图像）
- `[建议]` 单独写一个 round-trip 测试，并用 `vla_infer` 的 `VLAProtocol` 交叉验证（同一份 bytes 两边都能解）

**`transport.py`**（约 70 行）
- `ZmqRepServer(ip, port, jpeg_quality)`：`zmq.REP`、`LINGER=0`、`bind`
- `serve_forever(handler)`：`recv → handler(dict) → send`；**handler 异常不得冒泡**：
  记日志 + 回一个"安全块"（见 §5.5）或上一次的动作块，保证 REP 状态机不断
- `close()`；`[建议]` 支持 `Ctrl-C` 优雅退出（与 `ModelZmqInferenceServer` 行为对齐）

**`policy.py`**（核心，约 150 行）
```python
class VLAJepaPiperPolicy:
    def __init__(self, ckpt_path, unnorm_key=None, device="cuda:0", use_bf16=False,
                 num_inference_timesteps=None, chunk_steps=None, action_space="absolute"): ...
    def load(self):      # baseframework.from_pretrained + .to(device).eval() + 读取 dataset_statistics.json
    def warmup(self):    # 用一张 224×224 全零图 + 空指令跑一次 predict_action（避免首帧超时）
    def predict_request(self, obs: dict) -> dict:   # 协议层入口
```
`predict_request` 的映射（严格照 §1.4 的键契约）：

| 请求键 | 处理 | 送给模型的参数 |
|---|---|---|
| `image`, `wrist_image` | 校验 HWC3 uint8、负 stride 复制；`Image.fromarray` 转 PIL | `batch_images=[[head, wrist]]`（**顺序固定：头在前、腕在后**） |
| `cmd`（fallback `instruction`） | 取 str，缺省用 server 的 `default_instruction` | `instructions=[cmd]` |
| `state` | `float32`、`reshape(-1)`、长度必须 == `config.framework.action_model.state_dim` | `state=[state.reshape(1, -1)]`（形状 `(1,1,7)`） |
| `return_action_chunk` | 可选；`False` 时返回 `(1,7)` | — |

后处理：`normalized_actions[0]` → 反归一化（`min/max` 或 `q01/q99`，按 `dataset_statistics.json` 实际键名，见 [`model.md`](./model.md) §5.2）
→ `astype(np.float32)` → `np.ascontiguousarray` → 截断到 `chunk_steps`（默认取 `future_action_window_size+1`）
→ `{"action": chunk}`。

**`server.py`**（入口）
- 参数（`[建议]`）：`--ckpt_path`（必填）、`--host 0.0.0.0`、`--port 5555`、`--jpeg_quality 80`、
  `--unnorm_key`、`--cuda 0`、`--use_bf16`、`--num_inference_timesteps`、`--chunk_steps`、
  `--action_space {absolute,delta}`、`--default_instruction`、`--log_level`
- 启动顺序：解析参数 → `policy.load()` → `policy.warmup()` → `ZmqRepServer.serve_forever(policy.predict_request)`
- 日志：每次请求打印 `step / cmd / state 摘要 / latency_ms / action 首末行`（便于复盘）

### 5.4 与客户端的联合配置（真机启动配置）

✅ **已定默认（项目约定）**：`state_type="joint"` + `action_type="joint"` + `absolute_action=True`
（即**绝对关节角**语义）。下表其余项为在此默认之上仍需确认/覆盖的字段：

| 字段 | 客户端出厂默认 | 本项目取值 | 原因 |
|---|---|---|---|
| `server_ip` / `port` | `127.0.0.1` / `5555` | 指向 GPU 机器 / server 端口 | — |
| `state_type` | `"qpos"` | **`"joint"`（已定）** | G2：发 `joint(6)+gripper`，与训练 state 语义一致 |
| `action_type` | `"joint"` | `"joint"`（已定） | G11：保证 `joint_state` 被保存 |
| `absolute_action` | `True` | **`True`（已定，即 absolute 动作）** | G3：与全部 Piper 训练配置 `action_type: absolute` 对齐 |
| `task_instruction` | 自然语言 | **训练时该任务的 `tasks.jsonl` 文本**（当前 adjust_cup 为 task_index=0 的自然语言原文） | G1 |
| `enable_binary_gripper` | `False` | `False`（或按夹爪标定改为 `True` + 标定值） | G4 |
| `enable_gripper_transform` | `True` | **`False`** | G4 |
| `execute_chunk_steps` | `8` | **`7`**（= 模型 chunk 长度） | G5 |
| `timeout_ms` | `2000` | 先实测时延再定，必要时调大 | G7 |
| `use_smoothing` | `False` | 先 `False`；抖动明显再开（注意会改变动作分布） | — |
| `stop_on_timeout` | `True` | 保持 `True`（安全） | — |

### 5.5 安全与异常策略

1. **超时即停**：维持客户端既有行为（`TimeoutError` → `stop_on_timeout` 停机）。
2. **server 异常兜底**：handler 内 `try/except` 捕获一切异常，记完整 traceback，并回**上一个成功的动作块**
   （若无则回与当前 state 相同的"保持位姿"块，即 7 维 = 当前 state），避免 REP 卡死导致 2 s 急停。
   `[建议]` 该行为要打显式告警日志，便于区分"正常保持"与"降级"。
3. **启动自检**：`load()` 校验 ckpt 目录三件套（`config.yaml` + `dataset_statistics.json`）与 `state_dim`
   是否等于 7；不匹配直接拒绝启动（比上机后出错安全）。
4. **首帧预热**：`warmup()` 必须成功，否则不启动服务。
5. **动作合法性钳制**（可选，默认关闭）：`[待确认]` Piper 关节角的软限位从哪来（`my_robot` 侧未在 vla_infer 暴露），
   确认后再加逐维 clamp 与跳变保护。

### 5.6 测试计划（由轻到重）

| 级别 | 内容 | 通过标准 | 状态 |
|---|---|---|---|
| T1 单元 | `wire.py` 与 `vla_infer.src.zmq.protocol` 交叉 round-trip（同一 bytes 双向解包一致）；含 3D uint8 图像、float32/float64 状态、str/bool | 逐字段 `np.allclose` / 类型一致 | ✅ **已通过**（实测：控制端 pack → 本 server → 控制端 unpack，以及反向，均得 `(7,7) float32`） |
| T2 互通 | 用**机器人环境的 `VlaZmqClient`** 连我们的 server（假模型，返回固定 chunk） | 客户端拿到 `(7,7) float32`，无异常/无超时 | 🟡 **协议层已通过**：用 `selftest_client.py`（等价 REQ 收发）+ `--dry-run` 假策略验证；真 `VlaZmqClient` 需在机器人环境复跑 |
| T3 模型 | 真 ckpt 载入 + 单帧真实观测推理 | shape/dtype 正确；latency 记录；反归一化后数值落在训练 `q01/q99` 范围内 | ✅ **首轮通过**：`iclr_adjust_cup` 权重已载入并预热，8 个窗口延迟约 100–123 ms，P95 约 123 ms；实际 server 单请求客户端总延迟 193 ms |
| T4 离线回放 | 从数据集取 N 个窗口，构造**与真机同路径**的请求（图像 → letterbox 224 → JPEG80 → 解包），比较 server 返回与 GT chunk（复用 `scripts/analyze_openloop.py` 的指标思路） | MAE/R² 与直接用 `eval_openloop.py`（无 JPEG/letterbox）的结果同量级；差异可解释 | ✅ **已完成**：连续 gripper 路径 `--no-binarize-gripper` 已完成 50 episode × 2 窗口（100 窗口）回放；归一化 MAE=`0.0381`，gripper 维=`0.00153`，P95=`100.2 ms`。结果见 `eval_openloop/iclr_adjust_cup_piper_zmq_replay_continuous_gripper_50ep_2windows/replay.json`；默认二值路径仍保留为对照 |
| T4-SC 开环 | server + client 按数据集 episode 逐帧通信，保存完整 action chunk、step-0 对齐结果和轨迹图 | 连续请求无超时；每个回包为 `(7,7) float32`；JSON/NPZ/PNG 完整 | ✅ **已完成**：`adjust_cup` episode 0 共 124 帧，124 次请求全部成功；P95=`196.5 ms`，step-0 归一化 MAE=`0.0692`，结果见 `eval_openloop/iclr_adjust_cup_piper_zmq_openloop_ep0/` |
| T5 降级 | 构造坏 payload（缺 key、错 dtype、图像 4 通道）、模型内部抛异常 | client 不挂死；server 日志清晰；超时路径可复现 | 🟡 **坏 payload 已通过**（缺图像 → 记 `ValueError` + 回退保持位姿，server 不退出）；模型异常路径待做 |
| T6 真机 dry-run | 机械臂悬空或垫高，`max_steps` 小（如 20），低速、有人守急停 | 无异常动作方向；每周期时延稳定；执行 7 步/块 | ⏳ 待做 |
| T7 任务评测 | 每个任务固定初始位姿跑 N 次 | 成功率、平均完成时间、失败模式记录（写入 `doc/reports/`） | ⏳ 待做 |

### 5.7 里程碑

| 里程碑 | 交付物 | 出口条件 |
|---|---|---|
| **M1 协议互通** | `wire.py` + `transport.py` + 假模型 server | T1/T2 通过（不改 `vla_infer` 一行代码） |
| **M2 模型接入** | `policy.py` + `server.py`（真 ckpt） | T3 首轮通过；本机预热后稳态 P95 约 123 ms，低于 1 s |
| **M3 离线回放** | `scripts/piper_zmq_replay.py` + `replay.json` | T4 已完成；已记录 JPEG/bf16/夹爪二值化与连续反归一化路径的差异，真机动作语义仍需确认 |
| **M4 真机 dry-run** | 真机运行记录（时延/动作曲线） | T5/T6 通过；客户端配置按 §5.4 落地 |
| **M5 任务评测** | `doc/reports/piper_<task>_zmq.md` | T7 完成，成功率与失败模式可解释 |

### 5.8 验收标准（可测）

1. **零客户端改动**：`vla_infer` 侧只改 `InferenceConfig` 字段即可联通（`git diff` 仅配置/示例脚本）。
2. **契约正确**：请求 `{image, wrist_image, state, cmd}` → 响应 `{"action": (7,7) float32}`，
   连续 100 次请求无异常、无超时、无 REP 卡死。
3. **数值正确**：同一帧观测、固定随机种子下，server 返回的动作与本地 `predict_action` 直调反归一化结果一致（`np.allclose`，容差按 float32）。
4. **时延达标**：稳态 P95 < 1 s（超时预算 2 s 的一半）。
5. **可观测**：日志含 step、cmd、state 摘要、latency、action 首末行；异常有 traceback 与降级标记。
6. **文档同步**：[`deployment.md`](./deployment.md) §4（真机部署）与本文的实现状态保持一致（把"计划"标为"已实现/已验证"）。

### 5.9 需要确认的开放问题（阻塞项优先级从高到低）

| # | 问题 | 影响 | 怎么确认 |
|---|---|---|---|
| Q1 | `offset_state` 数据集里 `observation.state` 是否做过偏移（相对初始位姿）？ | G2，直接决定 state 输入是否正确 | 读一条 episode parquet 打印 state 首尾；与真机 `joint` 对比 |
| Q2 | Piper 夹爪指令的合法区间与方向（数据集原始 action 已确认范围为 `[0,1.4228571653]`，但真机标定/方向仍未知） | G4，夹爪可能不动或过闭 | 查 `my_robot` 驱动/标定，或单关节空载试打 |
| Q3 | 真机部署用哪个权重？（8 个 ICLR 任务各自一个 ckpt） | 决定 `task_instruction` 与 ckpt 路径 | 明确任务清单与权重路径 |
| Q4 | 当前 `adjust_cup` 的训练指令原文 | 已确认 task_index=0 为 `Put the cup the right way up on the table.`；换数据集时仍需复核 | 当前数据集 `meta/tasks.jsonl:1` |
| Q5 | GPU 与机器人是否同机？网络时延预算多少？ | 决定是否需要进一步优化时延 | 现场网络实测 |
| Q6 | 关节角软限位/最大速度约束 | §5.5 的钳制层是否必须 | 查 `my_robot` 配置 |

---

## 6. 附录

### 6.1 协议字段速查（照此实现即可互通）

**请求（client → server）**
```python
{
  "image":       np.ndarray(H, W, 3) uint8,   # JPEG 编码传输
  "wrist_image": np.ndarray(H, W, 3) uint8,   # JPEG 编码传输
  "state":       np.ndarray(7,)     float32,  # msgpack-numpy
  "cmd":         str,                         # 指令（README 写 instruction，代码用 cmd）
  # 可选
  "return_action_chunk": bool,
}
```
**响应（server → client）**
```python
{"action": np.ndarray(T, 7) float32}          # T=7（VLA-JEPA），客户端只读这一个键
```

### 6.2 关键代码行索引（控制端）

| 主题 | 位置 |
|---|---|
| 协议编解码 | `vla_infer/src/zmq/protocol.py:20,28-46,49-69,72-85` |
| REP server | `vla_infer/src/zmq/zmq_server.py:26-55` |
| REQ client（超时/急停） | `vla_infer/src/zmq/zmq_client.py:19-31,33-60` |
| 服务循环 | `vla_infer/src/inference/server.py:20-52` |
| 模型基类契约 | `vla_infer/src/models/base.py:15-40` |
| 参考适配器（观测校验/返回） | `vla_infer/src/models/dream_adapter_model.py:51-58,115-141,200-250` |
| 示例 server 入口 | `vla_infer/example/dream-adapter/dream-adapter_server.py:18-85` |
| 示例 Piper 客户端（含全部配置默认值） | `vla_infer/example/dream-adapter/dream-adapter-piper_client.py:29-66,107-142,144-162,164-238,252-291` |
| 机器人观测/执行 | `vla_infer/src/robots/piper_single.py:36-53,107-126,172-195,197-213,227-239` |
| 图像/动作后处理 | `vla_infer/src/process/utils.py:35-40,214-227,534-553,556-574` |
| 文档协议说明（与代码有出入） | `vla_infer/README.md:60-71` |

### 6.3 本仓库（VLA-JEPA）相关位置

| 主题 | 位置 |
|---|---|
| 现有 websocket server（另一套协议） | `deployment/model_server/server_policy.py` |
| **不兼容**的 msgpack 实现 | `deployment/model_server/tools/msgpack_numpy.py:43-45` |
| 推理接口与返回 | `starVLA/model/framework/VLA_JEPA.py:277-337` |
| 反归一化 | `starVLA/model/framework/base_framework.py:174-205` |
| Piper 任务配置（8 个 ICLR 任务） | `scripts/config/iclr_*.yaml` |
| 部署文档（真机 + 仿真） | `doc/robot/deployment.md`；模型契约见 `doc/robot/model.md` |
| 服务端样例实现 | `examples/real-robot/`（`wire.py`/`transport.py`/`policy.py`/`piper_zmq_server.py`/`selftest_client.py`） |
| 控制端示例（独立仓库） | `~/repo/Double_Piper_Teleop/vla_infer/example/vlajepa/vlajepa_piper_client.py` |
| Piper 关节角语义 / 语言条件缺陷 | `doc/01_data_conversion.md:369-383,385-401` |

### 6.4 与 `doc/robot/` 其他文档的关系

- [`model.md`](./model.md)：**模型契约**（载入、输入、数据、输出、反归一化）。本文复用它的结论，不重复展开。
- [`deployment.md`](./deployment.md)：**部署落地**——已有 websocket 链路（仿真评测）与真机 Piper 的默认配置、
  检查清单。本文是它在"真机 server 实现"方向上的**展开与前置设计**：协议逐条核对、风险清单（G1–G11）、
  模块设计与验收标准。
- [`README.md`](./README.md)：本目录索引。
