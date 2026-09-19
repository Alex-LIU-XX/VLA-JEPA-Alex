# 真机推理服务启动指南

## 前提条件

1. **uv 环境已安装**（见 `doc/04_environment_uv.md`）
2. **权重已下载**到仓库 `weights/` 目录：
   - `weights/Qwen3-VL-2B-Instruct/model.safetensors`（~4.0 GB）
   - `weights/vjepa2-vitl-fpc64-256/model.safetensors`（~1.3 GB）
3. **checkpoint 结构正确**：每个 run_dir 下必须有 `config.yaml` + `dataset_statistics.json` + `checkpoints/*.pt`
4. **权重无 NaN/Inf**：`iclr_pick_banana_newtable` 的 `steps_10000` 权重含 NaN/Inf（bf16 训练溢出），
   必须改用 `steps_20000`。其余 7 个任务的 `steps_10000` 均已验证无 NaN/Inf。

### checkpoint 目录结构

```
checkpoints/<run_id>/
├── config.yaml                        ← 必须在 run_dir 根目录
├── dataset_statistics.json            ← 必须在 run_dir 根目录
└── checkpoints/
    ├── config.json
    ├── dataset_statistics.json
    └── steps_10000_pytorch_model.pt   ← 权重文件
```

## 快速启动

### 1. 协议自检（不载入模型，不需要 GPU）

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex

# 终端 1：启动 dry-run server
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --dry-run --port 5555

# 终端 2：运行自检客户端
.venv/bin/python examples/real-robot/selftest_client.py \
    --port 5555 --steps 3 --send-bad-payload
```

### 2. 真实权重启动

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex

# 示例：iclr_adjust_cup（steps_10000 已验证无 NaN/Inf）
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_adjust_cup/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 \
    --use-bf16
```

### 3. 逐任务启动命令

所有命令基于仓库根目录 `/home/charles/workspaces/VLA-JEPA-Alex`，默认端口 `5555`。

#### iclr_adjust_cup

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_adjust_cup/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_open_cabinet

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_open_cabinet/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_pick_banana_newtable

> ⚠️ `steps_10000` 权重含 NaN/Inf（bf16 溢出），**必须用 `steps_20000`**。

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_pick_banana_newtable/checkpoints/steps_20000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_pick_banana_pot

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_pick_banana_pot/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_pick_block

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_pick_block/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_pick_eggplant_cluttered

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_pick_eggplant_cluttered/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_pick_eggplant_drawer

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_pick_eggplant_drawer/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

#### iclr_sponge_wipe

```bash
cd /home/charles/workspaces/VLA-JEPA-Alex && \
.venv/bin/python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/iclr_sponge_wipe/checkpoints/steps_10000_pytorch_model.pt \
    --port 5555 --use-bf16
```

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt_path` | 无 | checkpoint `.pt` 路径（`--dry-run` 时可省） |
| `--port` | `5555` | ZMQ 端口 |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--use-bf16` | 关 | 半精度推理 |
| `--cuda` | `0` | GPU 序号 |
| `--num-inference-timesteps` | `0`（用权重配置） | flow-matching 去噪步数 |
| `--chunk-steps` | `0`（= `future_action_window_size+1`） | 返回动作块长度 |
| `--default-instruction` | 空 | 请求无 `cmd` 时的默认指令 |
| `--binarize-gripper` | 关（默认不二值化） | 开启第 7 维 0/1 二值化；默认输出连续物理值 |
| `--dry-run` | 关 | 不载入模型，协议自检用 |
| `--max-requests` | `0`（常驻） | 处理 N 个请求后退出 |

## 启动日志示例

正常启动应看到：

```
model loaded in 28.4s | device=cuda:0 bf16=True | state_dim=7 chunk_steps=7 num_inference_timesteps=4
warmup done in 1.4s
client must be configured with state_type=joint + absolute_action=True (absolute joint targets)
ZMQ REP server ready on tcp://0.0.0.0:5555
```

## 控制端对接（Client 启动）

控制端（机器人 PC）运行 `Double_Piper_Teleop/vla_infer/example/vlajepa/vlajepa_piper_client.py`。

### 前提条件

1. 安装 `dream-adapter` conda 环境
2. 确认 server 已启动并输出 `ZMQ REP server ready`

### 通用启动模板

```bash
conda activate dream-adapter
cd ~/workspaces/Double_Piper_Teleop

python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "<INSTRUCTION>" \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

### 8 任务启动命令

> ⚠️ 指令来源：`adjust_cup` 为训练数据 `meta/tasks.jsonl` 确认原文；其余从数据集名推断，
> 首次使用前请在远程机器 `meta/tasks.jsonl` 核对。

| # | run_id | 指令 |
|---|---|---|
| 1 | `iclr_adjust_cup` | `Put the cup the right way up on the table.` |
| 2 | `iclr_open_cabinet` | `Open the cabinet door.` |
| 3 | `iclr_pick_banana_newtable` | `Pick up the banana from the table.` |
| 4 | `iclr_pick_banana_pot` | `Pick up the banana from the pot.` |
| 5 | `iclr_pick_block` | `Pick up the block.` |
| 6 | `iclr_pick_eggplant_cluttered` | `Pick up the eggplant from the cluttered area.` |
| 7 | `iclr_pick_eggplant_drawer` | `Pick up the eggplant from the drawer.` |
| 8 | `iclr_sponge_wipe` | `Wipe the table with the sponge.` |

#### 1. iclr_adjust_cup

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Put the cup the right way up on the table." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 2. iclr_open_cabinet

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Open the cabinet door." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 3. iclr_pick_banana_newtable

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Pick up the banana from the table." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 4. iclr_pick_banana_pot

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Pick up the banana from the pot." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 5. iclr_pick_block

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Pick up the block." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 6. iclr_pick_eggplant_cluttered

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Pick up the eggplant from the cluttered area." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 7. iclr_pick_eggplant_drawer

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Pick up the eggplant from the drawer." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

#### 8. iclr_sponge_wipe

```bash
conda activate dream-adapter && cd ~/workspaces/Double_Piper_Teleop && \
python vla_infer/example/vlajepa/vlajepa_piper_client.py \
    --server_ip localhost \
    --port 5555 \
    --task_instruction "Wipe the table with the sponge." \
    --timeout_ms 5000 \
    --execute_chunk_steps 7 \
    --state_type joint \
    --action_type joint \
    --absolute_action True \
    --control_interval_s 0.04 \
    --enable_binary_gripper False \
    --binary_gripper_threshold 0.4 \
    --gripper_open_value 0.5 \
    --gripper_closed_value 0.2 \
    --enable_gripper_transform False \
    --gripper_transform_threshold 0.55 \
    --gripper_transform_delta 0.3 \
    --use_smoothing False \
    --enable_action_interpolation False \
    --interpolation_method linear \
    --interpolation_target_steps 0 \
    --show_output_track False \
    --log_level INFO
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server_ip` | `127.0.0.1` | GPU 服务器 IP |
| `--port` | `5555` | ZMQ 端口，必须与 server 一致 |
| `--task_instruction` | 无 | 任务指令，**必须与 checkpoint 训练原文一致** |
| `--timeout_ms` | `5000` | 单次推理超时（ms），超时=急停 |
| `--execute_chunk_steps` | `7` | 每块执行步数，= 模型 chunk 长度 |
| `--state_type` | `joint` | **勿改**，必须为 `joint` |
| `--action_type` | `joint` | 动作类型 |
| `--absolute_action` | `True` | **勿改**，必须为 `True` |
| `--control_interval_s` | `0.04` | 控制间隔（秒） |
| `--enable_binary_gripper` | `False` | 二值夹爪 |
| `--binary_gripper_threshold` | `0.4` | 二值夹爪阈值 |
| `--gripper_open_value` | `0.5` | 夹爪打开值 |
| `--gripper_closed_value` | `0.2` | 夹爪关闭值 |
| `--enable_gripper_transform` | `False` | 夹爪变换 |
| `--gripper_transform_threshold` | `0.55` | 夹爪变换阈值 |
| `--gripper_transform_delta` | `0.3` | 夹爪变换增量 |
| `--use_smoothing` | `False` | 动作平滑 |
| `--enable_action_interpolation` | `False` | 动作插值 |
| `--interpolation_method` | `linear` | 插值方法 |
| `--interpolation_target_steps` | `0` | 插值目标步数 |
| `--show_output_track` | `False` | 显示输出轨迹 |
| `--log_level` | `INFO` | 日志级别 |

### 配置核对表

| 项 | server | client | 说明 |
|---|---|---|---|
| 端口 | `--port 5555` | `--port 5555` | 必须一致 |
| state | 要求 7 维 | `state_type=joint` → joint(6)+gripper(1) | **不能**用 `qpos` |
| 动作 | 输出绝对关节角 `(7,7)` | `absolute_action=True` → 原样下发 | **不能**做 cumsum |
| 指令 | `--default-instruction` | `--task_instruction` | 两者都应等于训练任务原文 |
| chunk | `chunk_steps=7` | `execute_chunk_steps=7` | 执行 7 步/块 |

### 测试产出

运行测试验证预处理和 obs 捕获：

```bash
conda activate dream-adapter
cd ~/workspaces/Double_Piper_Teleop
python -m unittest tests.test_vlajepa_piper_client -v
```

产出保存在 `tests/image/vlajepa_test/`：

| 文件 | 说明 |
|---|---|
| `cam_head.png` / `cam_wrist.png` | 双相机观测图像 |
| `obs_state.json` | 7 维状态（6 关节 + 1 夹爪） |
| `preprocess_*.png` | 预处理输入/输出对比 |
| `run_once_result.json` | 单次运行结果 |

## 故障排查

| 问题 | 原因 | 解决 |
|---|---|---|
| `Missing config.yaml` | run_dir 下缺少 config.yaml | 参照 checkpoint 结构将 config.json 转为 config.yaml 放到 run_dir |
| `HFValidationError: Repo id must be in the form` | `base_vlm` 路径不存在 | 确认 `weights/Qwen3-VL-2B-Instruct/` 存在 |
| `Connection reset by peer` | 代理不稳定 | 重试 `uv sync` 或检查代理配置 |
| warmup 超时 | CUDA 初始化慢 | 首次启动等待预热完成，后续请求正常 |
| warmup 报 `normalized actions contain NaN/Inf` | checkpoint 权重含 NaN/Inf（bf16 训练溢出） | 用干净的 checkpoint 重试；检查命令：`.venv/bin/python -c "import torch; ckpt=torch.load('<path>.pt',map_location='cpu'); print([k for k,v in ckpt.items() if isinstance(v,torch.Tensor) and (torch.isnan(v).any() or torch.isinf(v).any())])"` |
