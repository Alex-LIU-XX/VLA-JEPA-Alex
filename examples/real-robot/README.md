# 真机推理服务样例（Piper 控制端 `vla_infer` 协议）

本目录是**面向真机控制端协议**的 VLA-JEPA server 样例代码：控制端（机器人 PC）用现成的
`vla_infer` 客户端连上来，服务端（GPU 机器，VLA 环境）返回**绝对关节角 + 夹爪**的动作块。

协议设计与风险清单见 [`doc/robot/piper_server_plan.md`](../../doc/robot/piper_server_plan.md)，
真机默认配置与上机清单见 [`doc/robot/deployment.md`](../../doc/robot/deployment.md) §4。

## 文件

| 文件 | 作用 |
|---|---|
| `wire.py` | 线格式编解码：msgpack + `msgpack_numpy` + 3D-uint8→JPEG 启发式；与 `vla_infer/src/zmq/protocol.py` 逐条对齐 |
| `transport.py` | `zmq.REP` 服务循环：`LINGER=0`、坏帧隔离 + error/安全兜底回包 |
| `policy.py` | 策略层：观测校验/转换 → `predict_action` → 反归一化 → `{"action": (T,7) float32}`；含 `DryRunPolicy` |
| `piper_zmq_server.py` | 入口（argparse + 主循环 + 逐请求日志） |
| `selftest_client.py` | 协议自检客户端（按控制端 `VlaZmqClient` 的收发方式），不需要机械臂 |

## 快速开始

```bash
cd <VLA-JEPA-Alex 仓库根>

# ① 一切从"协议自检"开始：不载入模型、不需要 GPU
python examples/real-robot/piper_zmq_server.py --dry-run --host 127.0.0.1 --port 15555 --max-requests 4 &
python examples/real-robot/selftest_client.py --host 127.0.0.1 --port 15555 --steps 3 --send-bad-payload

# ② 换成真权重（VLA 环境；权重目录需含 config.yaml 与 dataset_statistics.json）
python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt \
    --port 5555 --use-bf16 \
     --default-instruction "Put the cup the right way up on the table."  # ← 当前数据 task_index=0 原文
```

使用真实数据集样本测试完整通信链路（另开终端先启动上面的 server）：

```bash
.venv/bin/python scripts/piper_zmq_dataset_client_test.py \
    --config_yaml checkpoints/iclr_adjust_cup/config.yaml \
    --output_dir eval_openloop/iclr_adjust_cup_piper_zmq_dataset_client \
    --host 127.0.0.1 --port 5555 \
    --windows-per-episode 2 --num-episodes 1
```

该脚本只读取数据集并发送 observation，不连接机械臂；它复用
`selftest_client.py` 的 `msgpack_numpy + JPEG + ZMQ REQ/REP` 封装方式。

逐帧执行一个离线 episode 的 server-client 开环回放并绘图：

```bash
# server 终端：保持运行直到 client 完成
python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
    --host 127.0.0.1 --port 15560 --use-bf16 --no-binarize-gripper \
    --default-instruction "Put the cup the right way up on the table."

# client 终端：--max-steps 0 表示完整 episode
.venv/bin/python scripts/piper_zmq_openloop_client.py \
    --config_yaml checkpoints/iclr_adjust_cup/config.yaml \
    --output_dir eval_openloop/iclr_adjust_cup_piper_zmq_openloop_ep0 \
    --host 127.0.0.1 --port 15560 \
    --num-episodes 1 --stride 1 --max-steps 0 \
    --server-gripper-mode continuous
```

输出包含 `openloop_result.json`、`predictions.npz`、`plot_manifest.json`，以及
`trajectory_fit.png`、`trajectory_zoom.png`、`trajectory_2d.png`、
`error_over_time.png`、`gripper_timeline.png`。该回放使用数据集 observation，
不向真实机械臂发送动作。

控制端（机器人 PC）：

```bash
cd ~/repo/Double_Piper_Teleop
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip <GPU_IP> --port 5555 --max_steps 20
# 该示例已内置本项目默认：state_type=joint / absolute_action=True / enable_gripper_transform=False /
# execute_chunk_steps=7（见 doc/robot/deployment.md §4.2）
```

## 参数说明（服务端）

> ⚠️ **控制端示例用的是 draccus，布尔参数必须带值**（写 `--dry_run true`、`--use_bf16 true`），
> 不能写成裸 flag；而本目录的 server 用 argparse，`--dry-run` / `--use-bf16` 就是裸 flag。
> 两边写法不同，照抄命令时留意。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt_path` | 无（`--dry-run` 时可省） | run 目录下的 `.pt`；同目录必须有 `config.yaml` + `dataset_statistics.json` |
| `--port` / `--host` | `5555` / `0.0.0.0` | 必须与控制端 `port` 一致 |
| `--jpeg-quality` | `80` | 回包图像压缩质量（与客户端同值即可） |
| `--cuda` / `--use-bf16` | `0` / 关 | 设备与精度（VLM 本身已硬编码 bf16，见 `doc/robot/model.md` §2.4） |
| `--num-inference-timesteps` | `0`（用权重配置，Piper 为 4） | flow-matching 去噪步数——**唯一有效的步数开关** |
| `--chunk-steps` | `0`（= `future_action_window_size+1`，Piper 为 7） | 返回动作块长度；控制端 `execute_chunk_steps` 应对齐 |
| `--unnorm-key` | 空（单数据集自动推断） | `dataset_statistics.json` 顶层键；多数据集权重必须显式给 |
| `--default-instruction` | 空 | 请求没有 `cmd` 时使用；建议填该任务训练时的原文 |
| `--no-binarize-gripper` | 关（即默认二值化） | 关闭第 7 维 0/1 二值化，并按当前 Piper 数据集 `min/max` 输出连续物理值；真机前仍需确认驱动标定 |
| `--dry-run` | 关 | 不载入模型，返回保持位姿的动作块（协议自检） |
| `--max-requests` | `0`（常驻） | 处理 N 个请求后退出（自动化测试用） |

## 协议要点（与 `vla_infer` 对齐，实现时不要"顺手优化"）

1. **msgpack 必须用真 `msgpack_numpy`**（`b'nd'` 扩展格式）。本仓库 `deployment/model_server/tools/msgpack_numpy.py`
   用的是 `b'__ndarray__'`，**不兼容**，别混用。
2. **3 维 uint8 数组一律当图像压成 JPEG**（与 key 名无关，打包侧）；
   **解包侧只按 key 名含 `img`/`image` 还原**——所以不要在响应里放 key 名不含 image 的 3D uint8 数组。
3. `float64` 一律降为 float32。
4. 请求键：`image` / `wrist_image` / `state`(7,) / `cmd`（文档曾写 `instruction`，代码用 `cmd`，本实现两者都收）。
5. 响应键：`{"action": (T,7) float32}`；控制端只读这一个键。
6. 服务端会隔离坏帧和 handler 异常，优先用当前有限 state 返回保持位姿；没有可信 state 时返回
   `{"error": ...}`，**绝不伪造零位或复用上一块绝对动作**。控制端必须把 error 当作失败并急停。
7. 默认 `--host 0.0.0.0` 是为跨机器部署保留的便利配置；生产环境必须绑定可信网卡并用防火墙限制端口。
   当前协议没有认证、客户端租约或请求 ID，不支持把服务暴露到不受信任网络。

## 已验证 / 未验证

### 真实权重离线回放（不连接机器人）

使用 `scripts/piper_zmq_replay.py` 可验证真实 checkpoint 的载入、预热、客户端图像预处理、
JPEG/msgpack 往返、反归一化动作和推理延迟：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/piper_zmq_replay.py \
    --config_yaml checkpoints/iclr_adjust_cup/config.yaml \
    --checkpoint checkpoints/iclr_adjust_cup/final_model/pytorch_model.pt \
    --output_dir eval_openloop/iclr_adjust_cup_piper_zmq_replay \
    --windows_per_episode 8 --num_episodes 1 --cuda 0 --use-bf16
```

该脚本只调用本地 policy，不打开 ZMQ 端口；ZMQ 线格式本身由下方 dry-run 自检覆盖。

**已验证**（本机实测，命令与输出见下）：

- `python3 examples/real-robot/selftest_client.py --steps 3 --send-bad-payload`
  → 坏 payload 时 server 返回 error（客户端不超时且 REP 状态机保持可用），随后 3 次正常请求均返回
  `(7,7) float32`，稳态时延 1 ms（dry-run，不含模型推理）。
- **与控制端原生 `protocol.py` 双向互通**：用 `vla_infer/src/zmq/protocol.py` 的
  `VLAProtocol.pack_payload` 打包 → 本 server 解包并回包 → 用 `VLAProtocol.unpack_payload` 解包，
  得到 `(7,7) float32`；反向（本仓库打包 → 控制端解包）同样通过。
  ⚠️ 注意：`vla_infer` 要求 `numpy>=2.0`，本仓库钉 `1.26.4`——交叉验证时用两个环境即可（正好也是真机拓扑）。

**已验证**（本机实测，`iclr_adjust_cup`，8 个窗口）：

- 真实权重载入、预热与离线回放；8 个窗口延迟约 100–123 ms，P95 约 123 ms；
- JPEG80/msgpack 往返后动作输出为有限的 `(7,7) float32`。

**未验证**（仍需目标机器或真机）：

- 与真实 `PiperSingleRobot` + 相机 + `VlaZmqClient` 的端到端联调；
- 动作语义（绝对关节角）与真机方向/限位的正确性——**必须先悬空/低速 dry-run**（见
  `doc/robot/deployment.md` §4.4）。

## 与计划文档的对应关系

本目录实现了 `doc/robot/piper_server_plan.md` §5.2 规划的模块（`wire` / `transport` / `policy` / `server`），
以及 §5.6 测试计划中的 T1（协议 round-trip、含控制端原生实现交叉验证）与 T2（假模型/互通）。
T3（真实权重）与 T4（离线回放）已在本机 `iclr_adjust_cup` checkpoint 上完成首轮验证；
T6（真机 dry-run）与 T7（任务评测）仍需在目标机器上继续。
