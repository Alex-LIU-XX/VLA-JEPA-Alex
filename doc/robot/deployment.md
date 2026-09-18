# VLA-JEPA 部署文档（真机 Piper + 仿真）

> **定位**：讲怎么把权重跑起来并接到目标上——两条链路（仿真 websocket / 真机 ZMQ）、命令、脚本、配置、排查。
> **模型自身**的契约（载入细节、输入输出形状、数据定义、反归一化公式）见 [model.md](./model.md)，
> 本文不重复展开，只在需要时给一句结论 + 链接。
> 真机 ZMQ server 的完整协议理解与实现计划见 [piper_server_plan.md](./piper_server_plan.md)。
>
> 判定标记：`[已确认]` = 有源码/脚本原文（给出 `文件:行`）；`[建议]` = 工程做法；`[待确认]` = 需实测。
>
> **路径缩写**：`model2libero_interface.py` / `eval_libero.py` 在 `examples/LIBERO/` 下；
> `model2simpler_interface.py` / `start_simpler_env.py` 在 `examples/SimplerEnv/eval_files/` 下；
> `.../piper_client.py` 指 `~/repo/Double_Piper_Teleop/vla_infer/example/dream-adapter/dream-adapter-piper_client.py`；
> `data_config.py` 指 `starVLA/dataloader/gr00t_lerobot/data_config.py`。

---

## 0. TL;DR

```bash
cd <VLA-JEPA-Alex 仓库根目录>

# ① 起 websocket 推理服务（仿真用；真机 ZMQ 见 §1.2）
python deployment/model_server/server_policy.py \
  --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt \
  --port 10093 --use_bf16 --cuda 0

# ② 自检（协议握手 + metadata）
python deployment/model_server/debug_server_policy.py --host 127.0.0.1 --port 10093 --test init

# ③ 仿真评测（改完脚本里的路径变量后）
bash ./examples/LIBERO/eval_libero.sh
bash examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh

# ④ 真机（Piper）：控制端配置见 §4.2，server 实现见 piper_server_plan.md
```

三条铁律：
1. `.pt` 上一级目录必须同时有 `config.yaml` 与 `dataset_statistics.json`（否则载入即失败，见 [model.md](./model.md) §2.1）。
2. 客户端拉到的是**归一化动作**，必须反归一化再下发（[model.md](./model.md) §5.2）。
3. 仿真客户端与 server 通常跑在**不同 python 环境**，`PYTHONPATH` 要同时含两个仓库根。

---

## 1. 两条部署链路

### 1.1 仿真 / 通用：websocket policy server（本仓库已有）

```
┌──────────────────────────┐        websocket + msgpack        ┌────────────────────────────┐
│  仿真环境 / 客户端        │  ───────────────────────────────►  │  policy server (GPU)       │
│  (各自 conda 环境)        │   {"payload": obs, "type":"infer"} │  deployment/model_server/  │
│                          │  ◄───────────────────────────────  │  server_policy.py          │
│  反归一化 + 动作执行       │   {"status","data":{"normalized_  │  └─ baseframework          │
│                          │       actions": ndarray}}          │     .from_pretrained(ckpt) │
└──────────────────────────┘                                    └────────────────────────────┘
```

- 传输：`websockets`，`compression=None, max_size=None`，连接后首帧发 server `metadata`。`[已确认]`
  （`deployment/model_server/tools/websocket_policy_server.py:39-58`）
- 序列化：msgpack + numpy 扩展（本仓库实现用 `__ndarray__` 标记；**与真机 ZMQ 链路不兼容**，见 §1.3）。`[已确认]`
  （`deployment/model_server/tools/msgpack_numpy.py:21-56`）
- 请求：`{"payload": obs, "type": "infer"}`；`obs` 的键就是 `predict_action` 的参数名
  （`batch_images` / `instructions` / 可选 `state`）。`[已确认]`（`websocket_policy_server.py:106-108`）
- 响应：`{"status","ok","type","request_id","data": {"normalized_actions": ndarray}}`；失败响应**没有 `data` 键**，
  而仓库里的客户端不检查 `status`，会直接抛 `KeyError: 'data'`。`[已确认]`
  （`websocket_policy_server.py:113-130`、`model2libero_interface.py:121`）
- 启动命令与参数表：见 [model.md](./model.md) §2.5。

### 1.2 真机：ZMQ REQ/REP（控制端 `vla_infer`，需自建 server）

真机控制端是**独立仓库** `~/repo/Double_Piper_Teleop`（`vla_infer` 模块），协议与上面**完全不同**：
`zmq.REQ/REP` + `msgpack_numpy.patch()` + 按 key 名启发式 JPEG 压缩；客户端每周期同步等回包，
超时（默认 2000 ms）即按 `stop_on_timeout` 停机。`[已确认]`

| 项 | 事实 |
|---|---|
| 地址 | `tcp://{ip}:{port}`，默认 `5555`；服务端 `bind`，客户端 `connect` |
| 请求键 | `image`（HWC3 uint8，已 resize 224）、`wrist_image`、`state`(7,)、`cmd` |
| 响应键 | `{"action": (T,7) float32}` —— 客户端只读这一个键 |
| 编码规则 | 3 维 uint8 数组一律压成 JPEG（与 key 名无关）；解包时**只按 key 名**（含 `img`/`image`）还原图像；float64 → float32 |
| 服务端契约 | `recv → model.predict(obs) → send`，模型必须返回 dict；单客户端串行；无 request id / 心跳 / 鉴权 |

完整协议逐条核对、与服务端的差异、以及 VLA-JEPA server 的目录/模块/测试/里程碑设计，全部在
[**piper_server_plan.md**](./piper_server_plan.md)（§1 服务端视角、§2 客户端视角、§4 风险、§5 计划）。

### 1.3 两套协议对照（别混用）

| 维度 | websocket（仿真/通用） | ZMQ（真机 Piper） |
|---|---|---|
| 传输 | `websockets` + msgpack | `pyzmq` REQ/REP + msgpack |
| numpy 编码 | `__ndarray__` dict（本仓库实现） | 真 `msgpack_numpy` 的 `b'nd'` 扩展 |
| 图像 | 协议内传 ndarray（服务端转 PIL） | 客户端先 JPEG 压缩（quality 80） |
| 请求键 | `batch_images`/`instructions`/`state` | `image`/`wrist_image`/`state`/`cmd` |
| 响应键 | `data.normalized_actions` | `action` |
| 现有 server | ✅ `deployment/model_server/server_policy.py` | ✅ **样例已实现**：`examples/real-robot/piper_zmq_server.py`（见其 [README](../examples/real-robot/README.md)） |

⇒ **两套 msgpack 实现互不识别**：直接把本仓库的 `tools/msgpack_numpy.py` 用到 ZMQ 链路上会解析失败。`[已确认]`

---

## 2. 端到端最小验证（不动仿真、不动真机）

```bash
# 终端 1：起服务
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt --port 10093 --use_bf16

# 终端 2：连通性自检（协议握手 + metadata）
python deployment/model_server/debug_server_policy.py --host 127.0.0.1 --port 10093 --test init
```

`[已确认]`（`deployment/model_server/README.md:1-22`、`debug_server_policy.py:25`）：
该脚本依赖 `sys.path[0]` 才能 import `tools.*`，所以**必须从仓库根、以脚本路径运行**。

判断标准：客户端打印出 server metadata（`{"env": "simpler_env"}`，硬编码值，跑 LIBERO 时也是它）。

⚠️ **不要依赖该脚本的 `--test infer` 做推理验证**：它硬编码读取 `assets/table.jpeg`，而 `assets/` 下只有
`VLA-JEPA.png`，在干净 clone 上该分支必然抛 `FileNotFoundError` 并被自身 `try/except` 吞掉
（日志 "Infer error (this still proves transport OK)"），**实际没有发出 infer 请求**。`[已确认]`
（`debug_server_policy.py:69-86`）

要做真正的推理冒烟测试：用 [model.md](./model.md) §2.6 的本地直调片段，或自己按 §1.1 的请求格式写 20 行客户端。
一次成功的 infer 响应应满足 `status == "ok"` 且 `data.normalized_actions.shape == (1, 7, 7)`（Piper 配置）。`[已确认]`

补充：`client.init_device("cuda")` 实际只发 `{"device": ..., "type": "ping"}`，服务端只回 ping、
**不会切换任何设备**。`[已确认]`（`websocket_policy_client.py:61-68`、`websocket_policy_server.py:92-93`）

---

## 3. 仿真部署

### 3.1 公共依赖与环境变量

| 项 | 值 / 来源 |
|---|---|
| 额外 pip 包 | `tyro matplotlib mediapy websockets msgpack`，并把 numpy 降到 `1.24.4`（`README.md:131-135`） |
| 服务端 python | VLA 环境（含 `websockets`、`msgpack`、`flash_attention_2`） |
| 客户端 python | 各仿真自己的环境（`sim_python`） |
| `PYTHONPATH` | 必须同时包含**仿真仓库根**与**本仓库根**（客户端要 import `deployment.*`、`starVLA.*`）——见 `eval_libero.sh:6-7` |

⚠️ 依赖版本有**两处冲突**，装环境时按实际情况选 `[已确认]`：
`README.md:134` 让装 `numpy==1.24.4`，而 `requirements.txt:25` 固定 `numpy==1.26.4`；
`requirements.txt:12` 装的是 `websocket-client`（另一个库，**不是** `websockets`），
所以 `websockets`、`msgpack`、`tyro` 都得按 `README.md:133` 手动补装。

### 3.2 LIBERO（4 个 task suite 并行，每 suite 一个 server）

```bash
# 1) 改 examples/LIBERO/eval_libero.sh 顶部：LIBERO_HOME / sim_python / your_ckpt
# 2) 一把跑完（脚本内部为每个 suite 起 server + client：base_port=15083，实际用 15084~15087，GPU 1~4）
bash ./examples/LIBERO/eval_libero.sh
```

脚本事实 `[已确认]`（`examples/LIBERO/eval_libero.sh:1-52`）：

- `export LIBERO_HOME=...`、`export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}` 与 `$(pwd)`
- `sim_python=/path/to/LIBERO/env/bin/python`
- server：`python ./deployment/model_server/server_policy.py --ckpt_path ${your_ckpt} --port ${port} --use_bf16 --cuda ${index} &`
- client：`${sim_python} ./examples/LIBERO/eval_libero.py --args.pretrained-path ${your_ckpt} --args.host 127.0.0.1 --args.port ${port} --args.task-suite-name <suite> --args.num-trials-per-task 50 --args.video-out-path <dir> --args.with_state true`
- 任务套件：`libero_10 / libero_goal / libero_object / libero_spatial`，每个 50 trials

客户端要点：`M1Inference` 默认 `port=10095`、`image_size=[224,224]`，`unnorm_key` 默认空
（单数据集自动推断）；每 `action_chunk_size` 步推理一次，gripper 从 `0/1` 二值化为 `±1`。`[已确认]`
（`model2libero_interface.py:19-33,116-133`、`eval_libero.py:25-29`）

两个容易被误导的点 `[已确认]`：

- **LIBERO 不跑 ensemble**：`AdaptiveEnsembler` 被构造和 reset，但 `step()` 从未调用 `ensemble_action`，
  实际行为是"每 7 步推理一次、开环执行整个 chunk"（`model2libero_interface.py:55-58,72-73,116-125`）。
- **脚本里的 `unnorm_key="franka"` 是死变量**（`eval_libero.sh:17` 定义后从未传给 python），
  所以走的是 `unnorm_key=None` → `_check_unnorm_key` 要求该权重的 `dataset_statistics.json`
  **只有一个顶层键**，否则 assert 失败；多数据集 co-train 权重需自己传 `unnorm_key`。`[已确认]`
  （`model2libero_interface.py:203-221`）
- 图像链路：LIBERO 环境以 **256** 渲染，客户端再 `cv.resize` 到 224。`[已确认]`（`eval_libero.py:24`）

### 3.3 LIBERO-Plus

```bash
bash ./examples/LIBERO-Plus/eval_libero_plus.sh
```

7 个扰动维度并行（`base_port=14082`，实际用 14083~14089；`num_trials_per_task=1`，`task_suite_name=libero_mix`，
`--args.category_value "<扰动名>"`）。需先把 `examples/LIBERO-Plus/libero_plus_init.py` 里的
`task_classification.json` 路径改成自己的，并按 README 替换 LIBERO-Plus 的 `benchmark/__init__.py`。`[已确认]`
（`examples/LIBERO-Plus/eval_libero_plus.sh:1-48`、`README.md:153-168`）

### 3.4 SimplerEnv

```bash
# 一把跑完 5 个任务面（内部为每个 run 起 server + 客户端 sim）
bash examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh

# 统计成功率（跑完后用生成的视频算）
bash ./examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh \
     <pick_coke_can|move_near|drawer|long_horizon_apple_in_drawer|bridge_put_on|all> \
     <model_path> [log_dir_root]
```

- `batch_evaluate.sh` 先设 `sim_python` 与 `SimplerEnv_PATH`，再依次调
  `star_bridge.sh` / `star_drawer_*` / `star_move_near_*` / `star_pick_coke_can_*` / `star_put_in_drawer_*`。`[已确认]`
  （`batch_evaluate.sh:1-19`）
- 任务面与 robot：`bridge_put_on` → WidowX；其余四个 → Google Robot（`README.md:186-193`）。
- 单机手工跑（最小可复现）：先 `run_policy_server.sh`（默认 `port=6680`, `gpu_id=0`，`--use_bf16`），再
  ```bash
  python examples/SimplerEnv/eval_files/start_simpler_env.py \
    --ckpt-path <model.pt> --port 6680 --policy-setup widowx_bridge \
    --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
    --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
    --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
    --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
    --obj-variation-mode episode --obj-episode-range 0 24 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1
  ```
  `[已确认]`（`examples/SimplerEnv/eval_files/start_simpler_env.sh:1-75`）
- SimplerEnv 客户端默认 `port=10093`、`policy_setup` 默认 `google_robot`，通过 `--policy-setup` 切换；
  连服务端时 **host 用默认值 `0.0.0.0`**（`start_simpler_env.py:30-36` 未透传 `args.host`）——
  实际能连通，但建议显式传 `127.0.0.1` 以免误解。`[已确认]` + `[建议]`

**各任务面参数一览**（照抄自 `star_*.sh`）`[已确认]`：

| 任务面（`calc_success_rate.sh` 的 task） | env / robot | control-freq / sim-freq / max-episode-steps | 脚本 |
|---|---|---|---|
| `bridge_put_on` | `PutCarrotOnPlateInScene-v0` 等 3+1 个，`widowx` / `widowx_sink_camera_setup` | `5 / 500 / 120` | `star_bridge.sh:13,152-154` |
| `pick_coke_can` | `GraspSingleOpenedCokeCanInScene-v0`，`google_robot_static`（3 种摆放 × 4 种 urdf） | `3 / 513 / 80` | `star_pick_coke_can_visual_matching.sh:106,127-134` |
| `move_near` | `MoveNearGoogleBakedTexInScene-v0`，`google_robot_static` | `3 / 513 / 80` | `star_move_near_visual_matching.sh:97,123` |
| `drawer` | 6 个 drawer env（`dummy_drawer` 场景，带 raytracing 额外参数） | `3 / 513 / 113` | `star_drawer_visual_matching.sh:102-120,136` |
| `long_horizon_apple_in_drawer` | 同上 drawer 系列（`model_ids=baked_apple_v2`） | `3 / 513 / 200` | `star_put_in_drawer_visual_matching.sh:127,159` |

与 LIBERO 的关键区别：**SimplerEnv 客户端每步都请求一次推理**（没有 chunk 缓存），
并用 `AdaptiveEnsembler` 做历史动作加权平均（widowx 窗口 7 / google_robot 窗口 2，alpha=0.1）。`[已确认]`
（`model2simpler_interface.py:129-154`、`:42-61`、`adaptive_ensemble.py:10-44`）

### 3.5 部署脚本索引与改造清单（照 examples/ 实测脚本整理）

**所有脚本都假定 CWD = 仓库根目录**（用 `$(pwd)` 拼 `PYTHONPATH`、用 `deployment/...` 相对路径），
否则 server 起不来、客户端 import 不到模块。`[已确认]`

| 脚本 | 作用 | 必须改的变量（行号） | 产物 / 日志 |
|---|---|---|---|
| `examples/LIBERO/eval_libero.sh` | 4 个 LIBERO suite 各占一卡并行（`base_port=15083` → 端口 15084~15087，`--cuda 1~4`） | `LIBERO_HOME`(4)、`sim_python`(9)、`your_ckpt`(11) | `results/<suite>/<ckpt_tag>/eval.log` + 评测视频；`logs/<时间戳>/` |
| `examples/LIBERO-Plus/eval_libero_plus.sh` | 7 个扰动维度并行（`base_port=14082` → 14083~14089，每任务 1 trial） | 同上 3 个 + `libero_plus_init.py:121` 的 `task_classification.json` 路径 | `results/plus_libero_mix/<扰动>/<ckpt_tag>/` |
| `examples/SimplerEnv/eval_files/run_policy_server.sh` | 单机手动起 1 个 server（`port=6680`、`gpu_id=0`） | `your_ckpt`(11)、`port`(6)、`gpu_id`(7) | `<ckpt_dir>/output_server/<ckpt>_policy_server_<port>.log` |
| `examples/SimplerEnv/eval_files/start_simpler_env.sh` | 旧版单机脚本：3 个 bridge 任务 + 1 个 sink 任务 | `SimplerEnv_PATH`(6)、`MODEL_PATH`(11)、`port`(13) | `<ckpt_dir>/output_eval/*.log` |
| `.../auto_eval_scripts/batch_evaluate.sh` | 一把跑 5 个任务面（依次调 5 个 `star_*.sh`） | `sim_python`(8)、`SimplerEnv_PATH`(9)、`MODEL_PATH`(12) | 见下面各 `star_*.sh` 的目录 |
| `.../star_bridge.sh` | WidowX bridge 3+1 任务，8 卡轮转，`base_port=6680`，`sleep 10` 等服务起来 | 接受第一个参数 `<ckpt>`；或由 `batch_evaluate.sh` 传入 | `<ckpt_dir>/widowx_robot_eval/` |
| `.../star_pick_coke_can_visual_matching.sh` | pick coke can：3 种摆放 × 4 种 urdf = 12 run | 同上（⚠️ 脚本内变量名有 bug，见 §5-4） | `<ckpt_dir>/google_robot_eval/` |
| `.../star_move_near_visual_matching.sh` | move near：1 任务 × 4 urdf | 同上（⚠️ 同 bug） | 同上 |
| `.../star_drawer_visual_matching.sh` | 6 个 drawer 任务 × 4 urdf，带 `--enable-raytracing` 等 `EXTRA_ARGS`（拼写正确） | 同上 | 同上 |
| `.../star_put_in_drawer_visual_matching.sh` | long-horizon 放苹果进抽屉（⚠️ 同 bug） | 同上 | 同上 |
| `.../calc_success_rate.sh` | 用输出视频统计成功率 | `<task> <model_path> [log_dir_root]` | 打印成功率（内部调 `calc_metrics_evaluation_videos.py`） |
| `.../check_ports_range.sh` | 检查端口是否空闲 | ⚠️ 检查的是 **5400-5500**，与实际 `base_port=6680` 不匹配 | — |
| `scripts/piper_zmq_dataset_client_test.py` | 用真实数据集样本验证 Piper ZMQ server 的 observation/action 通信 | `--config_yaml`、`--host`、`--port` | `test_result.json` |
| `scripts/piper_zmq_openloop_client.py` | 逐帧 server-client 开环回放并绘制动作轨迹 | `--config_yaml`、`--host`、`--port`、episode 参数 | `openloop_result.json`、`predictions.npz`、PNG 轨迹图 |

**脚本里的隐含约定（照抄时最容易漏）** `[已确认]`：

- `sim_python` 必须是**仿真环境自己的 python**（`/…/LIBERO/env/bin/python`、`/…/SimplerEnv/env/bin/python`）；
  而 server 用**本仓库 VLA 环境的 python**（脚本里写的是 `python` / `star_vla_python=python`）
  ⇒ 运行前要么 `conda activate VLA_JEPA`，要么把脚本里的 `python` 换成 `.venv/bin/python`。
- `PYTHONPATH` 必须同时含**仿真仓库根**与**本仓库根**（LIBERO 脚本已显式设置；
  SimplerEnv 的接口文件要 `import deployment.*`，同样需要本仓库根）。
- 起服务的脚本会先 `sleep 10~20 s` 再拉客户端；`star_bridge.sh:44-55` 还会用 `lsof | xargs kill -9` 抢占端口，
  其余脚本**不会**（端口被占时表现为客户端连不上/超时）。
- 渲染与并行相关环境变量：`SVULKAN2_CPU_COPY=1`、`SVULKAN2_DISABLE_DENOISER=1`（SimplerEnv）、
  `DISPLAY=""`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`（`start_simpler_env.py:24-26`）、
  `TOKENIZERS_PARALLELISM=false`、`PYTHONDONTWRITEBYTECODE=1`（LIBERO）。
- 批量脚本会自己 `mkdir -p` 输出/日志目录、后台并行、结束时 `kill` 掉所有 server（`stop_all_services`）。
- `MODEL_PATH`（或 `your_ckpt`）与 `TSET_NUM`（每任务跑几遍）可用环境变量覆盖，便于批量改权重。

**改造清单（换成自己的权重，最小改动）** `[建议]`：

1. **LIBERO**：改 `eval_libero.sh` 的 3 行 —— `LIBERO_HOME` / `sim_python` / `your_ckpt`，然后 `bash` 执行。
2. **LIBERO-Plus**：在 ① 基础上，把 `examples/LIBERO-Plus/libero_plus_init.py` 覆盖到 LIBERO-Plus 安装目录的
   `libero/libero/benchmark/__init__.py`，并改其中 `task_classification.json` 的绝对路径。
3. **SimplerEnv**：`export sim_python=… SimplerEnv_PATH=…` + `MODEL_PATH=<ckpt>`（或直接改 `batch_evaluate.sh`），
   跑完用 `calc_success_rate.sh <task> <ckpt> <log_dir>` 统计；`bridge_put_on` 的输出目录是
   `widowx_robot_eval`，**必须显式传第三个参数**，否则会去默认的 `google_robot_eval` 找视频。
4. **卡数不足**：脚本用 `run_count % NUM_GPUS` 轮转分配 GPU，缩小 `CUDA_VISIBLE_DEVICES`
   或裁剪 `items` / `urdf_version_arr` 数组即可，只是更慢。

---

## 4. 真机部署（Piper）

### 4.1 现状：本仓库不含真机控制端

- 本仓库提供**模型服务**与**数据/训练/开环评估**；真机控制端是**独立仓库**
  `~/repo/Double_Piper_Teleop`（`vla_infer` 模块）。`[已确认]`（本仓库 `deployment/` 下无真机控制代码，
  只有 `deployment/readme-deployment.md` 的 legacy 笔记，提到的 `real_deployment.model_controller_del_ee` 不存在）
- 真机 ZMQ server 已完成协议样例、dry-run 自检，以及 `iclr_adjust_cup` 真实权重的本机离线回放；
  真实机器人相机/控制端联调和动作安全性仍未完成。`[已确认]`（`examples/real-robot/`、
  `scripts/piper_zmq_replay.py`、`eval_openloop/iclr_adjust_cup_piper_zmq_replay_continuous_gripper_50ep_2windows/replay.json`）
- 控制端的模型接口与传输与本仓库不同：

  | 维度 | 本仓库 server | 控制端 `vla_infer` |
  |---|---|---|
  | 传输 | websocket + msgpack | ZMQ（图像 JPEG 压缩） |
  | 观测键 | `batch_images` / `instructions` / `state` | `image` / `wrist_image` / `state` / `cmd` |
  | 模型接口 | `predict_action(...) -> {"normalized_actions": ...}` | `BaseVLAModel.predict(obs) -> {"action": (chunk, dim)}` |
  | 已有适配 | — | smolvla / vla_adapter / OpenVLA_OFT / dream_adapter |

  `[已确认]`（`vla_infer/README.md:1-20`、`src/models/base.py:15-40`、`src/inference/piper_client.py:93-121`）

  ⇒ **缺口**：控制端**不区分模型**（只管发观测、执行动作），缺的是 VLA-JEPA 这一侧的 server；
  实施方案见 [`piper_server_plan.md`](./piper_server_plan.md)。`[已确认]`

### 4.2 真机部署默认配置（本项目约定）

控制端 `vla_infer` 的 `InferenceConfig`（`example/dream-adapter/dream-adapter-piper_client.py:29-66`）
在**本项目里统一按下面这套取值使用**：

| 字段 | 取值 | 理由 |
|---|---|---|
| `state_type` | **`"joint"`** | 发送 `joint(6) + gripper(1)`，与训练数据 state 语义一致（[model.md](./model.md) §3.4、`doc/01_data_conversion.md:369-383`） |
| `action_type` | `"joint"` | 保持默认；它决定 `joint_state` 会被保存（`...piper_client.py:139-140`） |
| `absolute_action` | **`True`**（动作语义 = **absolute**） | 服务端返回的 7 维就是"绝对关节角 + 夹爪"，客户端原样下发；与全部 Piper 训练配置 `action_type: absolute` 对齐（`scripts/config/iclr_*.yaml:56`） |
| `enable_gripper_transform` | **`False`** | 关掉客户端"再下压 0.3"的逻辑，避免改变 server 输出的二值或连续夹爪值（默认 `True`，`...piper_client.py:64-66,201-203`） |
| `execute_chunk_steps` | **`7`** | = 模型 chunk 长度（`future_action_window_size + 1`），避免"以为执行了 8 步" |
| `timeout_ms` | `2000`（实测后再调） | 超时即走急停路径 |
| `stop_on_timeout` | `True` | 安全默认 |
| `task_instruction` | 该任务在训练数据 `meta/tasks.jsonl` 里的**原文** | 当前 `adjust_cup` 数据集应传 `Put the cup the right way up on the table.`；换数据集必须重新核对 |

> ⚠️ **反向提醒**：客户端**出厂默认**是 `state_type="qpos"` + `enable_gripper_transform=True`，
> 这套默认下会把 `qpos` 当 state、把绝对动作当**增量**逐帧累加、并再次下压夹爪——三处都会出错。
> 启动前务必按上表覆盖。`[已确认]`（`...piper_client.py:48-49,64,179-186`）

启动命令（以 dream-adapter 客户端为例；VLA-JEPA 复用同一个客户端，只需换配置）：

```bash
cd ~/repo/Double_Piper_Teleop        # CWD 决定 vla_infer 的包路径
python vla_infer/example/dream-adapter/dream-adapter-piper_client.py \
  --server_ip <GPU_IP> --port 5555 \
  --task_instruction "Put the cup the right way up on the table." \
  --state_type joint \
  --absolute_action true \
  --enable_gripper_transform false \
  --execute_chunk_steps 7 \
  --max_steps 20                     # 首次上机先用小步数
```

`[建议]`（draccus 支持用 `--字段 值` 覆盖 dataclass 字段；首次上机请保留旁观者与急停按钮）。
服务端 = `examples/real-robot/piper_zmq_server.py`（[样例 README](../examples/real-robot/README.md)、设计见
[`piper_server_plan.md`](./piper_server_plan.md)）：

```bash
cd <VLA-JEPA-Alex 仓库根>
python examples/real-robot/piper_zmq_server.py \
    --ckpt_path checkpoints/<run_id>/final_model/pytorch_model.pt \
    --port 5555 --use-bf16 \
    --default-instruction "Put the cup the right way up on the table."  # ← 当前数据 task_index=0 原文
```

当前 server 默认保留框架兼容的 gripper 二值化；离线回放验证的连续 gripper 路径需额外加
`--no-binarize-gripper`。真机启动前必须依据 Piper 驱动标定结果二选一，不能仅凭离线数据范围决定。

控制端（机器人 PC）用配套示例客户端一键连接（默认值即 §4.2 的约定）：

```bash
cd ~/repo/Double_Piper_Teleop
python vla_infer/example/vlajepa/vlajepa_piper_client.py --server_ip <GPU_IP> --port 5555 --max_steps 20
```

### 4.3 对接方案（三种形态，推荐第一种）

> 📌 **推荐：ZMQ 原生 server**，协议细节、映射表、配置与验收标准都在
> [`piper_server_plan.md`](./piper_server_plan.md)。下表只做结论性对比。`[已确认]` + `[建议]`

| 形态 | 做法 | 取舍 |
|---|---|---|
| **① ZMQ 原生 server（推荐，样例已就绪）** | `examples/real-robot/`（`wire.py`/`transport.py`/`policy.py`/`piper_zmq_server.py`）直接实现控制端的 REQ/REP + msgpack + JPEG 协议 | 控制端**零代码改动**（只改配置）；server 只需 `pyzmq/msgpack/msgpack-numpy`，不引入 numpy 2.x 冲突 |
| ② 控制端加转发适配器 | 控制端新增 `BaseVLAModel` 适配器，转发到本仓库现有 websocket server | 需要在控制端维护协议转换；链路多一跳；仅在必须复用 websocket server 时选 |
| ③ 控制端进程内载入 | 控制端环境直接 `from starVLA... import baseframework` | 控制端需装 Qwen3-VL / flash-attn / V-JEPA2 权重，环境最重，只适合单机调试 |

三种形态的**观测/动作映射相同**（对应关系已核实）：

1. `obs["image"]`、`obs["wrist_image"]`（HWC uint8）→ `batch_images=[[头, 腕]]`（顺序固定：头在前、腕在后，
   与 `PiperDataConfig.video_keys` 一致）
2. `obs["cmd"]` → `instructions=[cmd]`
3. `obs["state"]`（7 维，**不要归一化**）→ `state=[state.reshape(1, 7)]`
4. 模型输出 `normalized_actions[0]` → 按 `dataset_statistics.json` 反归一化（[model.md](./model.md) §5.2）
   → `{"action": chunk}`（`(T,7) float32`）

`[已确认]`（`starVLA/model/framework/VLA_JEPA.py:277-337`、`data_config.py:787-790`）

### 4.4 上机前检查清单

`[建议]`（基于本仓库已验证事实，作为真机接入的自检项）：

1. **动作空间已对齐**：默认配置是 `state_type="joint"` + `absolute_action=True`（§4.2），与 Piper 全部
   `action_type: absolute` 的训练配置一致；若换用 `delta_qpos` 训练的权重，必须同步改成增量语义
   （`absolute_action=False`，且注意该开关只在 `state_type="joint"` 分支生效）。
2. **state 语义正确**：7 维 = 6 关节角 + 夹爪，**不是末端位姿**（`x/y/z/roll/pitch/yaw` 只是标签名）；
   `state_type="joint"` 才会发关节角。上机前把客户端打印的 `state` 与数据集里同场景的
   `observation.state` 比一比量级。`[已确认]`
   （`doc/01_data_conversion.md:369-383`、`vla_infer/src/robots/piper_single.py:107-126,172-195`）
3. 腕相机图像是否与训练时的 `observation.images.wrist_image` 同视角、同朝向（是否有镜像/旋转）？
   注意客户端用 letterbox（保持长宽比 + 白边）缩放到 224，而训练侧是直接拉伸（差异见
   [piper_server_plan.md](./piper_server_plan.md) §4-G6）。
4. 控制频率与 chunk 执行策略：一次推理执行 7 步（`execute_chunk_steps=7`），`control_interval_s=0.04`。
5. 反归一化用的统计量键名与权重匹配（`min/max` vs `q01/q99`），gripper 是否期望 0/1 二值
   （[model.md](./model.md) §5.2、§5.3）。
6. 安全：先让机械臂悬空/低速验证方向与限位，再接触物体；异常时能立刻急停。
7. 先用同一批真机数据跑 `scripts/eval_openloop.py` 做开环验证（离线、无风险），
   确认 R² 与逐维误差可接受，再上机闭环。

---

## 5. 部署侧坑位与排查表

| # | 现象 | 原因 | 处理 |
|---|---|---|---|
| 1 | 客户端连不上（connection refused / 超时） | 端口不一致：LIBERO 客户端默认 10095、SimplerEnv 默认 10093、server 默认 10093、（真机 ZMQ）5555 | **永远显式传 `--port`**，两边一致（`model2libero_interface.py:32`、`model2simpler_interface.py:35`、`server_policy.py:47`） |
| 2 | 客户端 import 不到 `deployment.*` / `starVLA.*` | `PYTHONPATH` 没带本仓库根；或 CWD 不在仓库根 | 照 `examples/LIBERO/eval_libero.sh:6-7` 设 `PYTHONPATH`；脚本一律从仓库根执行 |
| 3 | 仿真评测比预期慢很多 | SimplerEnv 客户端每步都推理（LIBERO 是每 7 步一次） | 属预期；嫌慢可调大 `--control-freq` 或改客户端做 chunk 缓存（`model2simpler_interface.py:129-154`） |
| 4 | `star_move_near_*` / `star_pick_coke_can_*` / `star_put_in_drawer_*.sh` 起 server 失败，客户端干等 600 s 后超时 | 脚本里用了 `${starvla_python}`，实际导出的是 `star_vla_python` | 统一变量名，或直接用 `run_policy_server.sh` 手工起服务（对照 `star_bridge.sh:5,57`） |
| 5 | `calc_success_rate.sh` 统计结果为空/找不到视频 | `<log_dir>` 传错：google 系列输出在 `${ckpt_dir}/google_robot_eval`（脚本默认），bridge 输出在 `${ckpt_dir}/widowx_robot_eval` | 显式传第三个参数（`star_bridge.sh:24` vs `calc_metrics_evaluation_videos.py:1007`） |
| 6 | 端口被占用但没有提示 | 只有 `star_bridge.sh:44-55` 会 `lsof` 抢端口；`check_ports_range.sh:4-5` 检查的是 5400-5500，与实际 `base_port=6680` 不一致 | 自行确认端口空闲，或按脚本统一改 `base_port` |
| 7 | 真机上动作"越走越远"或第一帧就大幅跳动 | 客户端被当成**增量**语义：`state_type="qpos"`（出厂默认）会对绝对动作做 `cumsum` 累加 | 按 §4.2 覆盖为 `state_type=joint` + `absolute_action=true`（`...piper_client.py:179-186`） |
| 8 | 真机 state 与训练分布不符 | `state_type="qpos"` 发的是 `qpos` 字段而不是关节角 | 同上；上机前对比数据集 `observation.state` 量级（§4.4-2） |
| 9 | 真机客户端连不上 server，2 s 后急停 | server 未启动/端口不一致/控制端与 server 机器的网络不通 | 先在 server 机器确认端口监听；再核对 §4.2 的 `server_ip`/`port` |
| 10 | 真机首次请求就超时 | 模型未预热（首帧含 CUDA 初始化与权重上卡） | server 载入后做一次 dummy forward（[piper_server_plan.md](./piper_server_plan.md) §5.3、§5.5） |

---

## 6. 部署侧关键文件索引

| 用途 | 文件 |
|---|---|
| 起 websocket 服务 | `deployment/model_server/server_policy.py` |
| 连通性自检（`--test infer` 在干净 clone 上会失败，见 §2） | `deployment/model_server/debug_server_policy.py` |
| server 路由与协议 | `deployment/model_server/tools/websocket_policy_server.py` |
| websocket 客户端 | `deployment/model_server/tools/websocket_policy_client.py` |
| 图像转换（numpy ↔ PIL） | `deployment/model_server/tools/image_tools.py` |
| 仿真客户端（LIBERO） | `examples/LIBERO/model2libero_interface.py`、`eval_libero.py`、`eval_libero.sh` |
| 仿真客户端（LIBERO-Plus） | `examples/LIBERO-Plus/eval_libero_plus.sh`、`libero_plus_init.py` |
| 仿真客户端（SimplerEnv） | `examples/SimplerEnv/eval_files/model2simpler_interface.py`、`start_simpler_env.py`、`run_policy_server.sh` |
| 仿真批量脚本（含变量与产物） | `examples/SimplerEnv/eval_files/auto_eval_scripts/`（见 §3.5 索引表） |
| 开环评估（真机数据，离线无风险） | `scripts/eval_openloop.py`、`scripts/analyze_openloop.py` |
| 真机控制端（独立仓库） | `~/repo/Double_Piper_Teleop/vla_infer/`（ZMQ + `BaseVLAModel`） |
| 模型侧文件（载入/推理/数据） | 见 [model.md](./model.md) §7 |

---

## 附：相关文档

| 文档 | 内容 |
|---|---|
| [model.md](./model.md) | 模型载入、输入格式、数据定义、输出格式（部署前必读） |
| [piper_server_plan.md](./piper_server_plan.md) | 真机 ZMQ 协议理解 + VLA-JEPA server 开发计划（含 §4 风险清单 G1–G11） |
| `README.md`（仓库根） | 项目总览、官方评测（LIBERO / LIBERO-Plus / SimplerEnv） |
| `doc/01_data_conversion.md` | v3 → v2.1 数据转换 + robot_type 注册 |
| `doc/02_training.md` | 训练启动与排错 |
| `doc/03_openloop_testing.md` | 开环测试与指标解读 |
