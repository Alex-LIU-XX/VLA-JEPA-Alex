# AGENTS.md

本文件是注入到每个 agent 的工作区上下文，说明**项目是什么、代码在哪、约定是什么、如何验证**。
修改代码或写文档前先读本文件与 `doc/README.md`。

## Project

**VLA-JEPA-Alex** 是在 [starVLA](https://github.com/starVLA/starVLA) 之上做真机（Piper 双臂/单臂）
落地的 VLA 项目仓库。核心模型（framework 名 `VLA_JEPA`）由三部分组成：

| 组件 | 作用 | 代码位置 |
|---|---|---|
| Qwen3-VL-2B / Qwen2.5-VL | 视觉-语言主干，融合图像 + 语言指令 + 动作 token | `starVLA/model/modules/vlm/` |
| V-JEPA2 世界模型 | 由动作 token 预测未来帧隐表示（`wm_loss`），**仅训练期使用** | `starVLA/model/modules/world_model/` |
| Flow-matching DiT 动作头 | 从噪声迭代采样动作 chunk（`action_horizon` 步） | `starVLA/model/modules/action_model/` |

三条主线任务：
1. **真机数据 → 训练**：Piper 采集数据（LeRobot v2.1）微调；
2. **开环评估**：在真实数据上比对预测动作与真值（`scripts/eval_openloop.py`）；
3. **部署**：仿真走 websocket policy server（`deployment/model_server/`）；真机 Piper 的控制端
   `~/repo/Double_Piper_Teleop/vla_infer` 用的是 **ZMQ REQ/REP + msgpack + JPEG**（与本仓库现有 websocket
   协议**不兼容**），需要按 `doc/robot/piper_server_plan.md` 单独实现 server。

## Repository Layout

```
starVLA/
├── config/            # accelerate / deepspeed 启动配置 + 官方训练 yaml
├── dataloader/        # LeRobot v2.1 读取、modality 解析、state/action 归一化
│   └── gr00t_lerobot/ # robot_type 配置(data_config.py)、mixtures.py、transform/
├── model/
│   ├── framework/     # 各 framework 实现 + FRAMEWORK_REGISTRY 工厂（build_framework）
│   └── modules/       # vlm / world_model / action_model / projector
└── training/          # train_starvla.py 等训练入口与 trainer_utils
deployment/model_server/   # websocket policy server 与 client（仿真部署入口）
examples/                  # 仿真评测：LIBERO / LIBERO-Plus / SimplerEnv
examples/real-robot/       # 真机 ZMQ server 样例（Piper 控制端协议）+ 协议自检客户端
scripts/                   # 训练/评测/数据转换脚本 + scripts/config/*.yaml（逐任务配置）
doc/                       # 中文流程文档（索引见 doc/README.md）；doc/robot/ 为部署相关文档：
                           #   model.md=模型契约，deployment.md=真机+仿真部署，
                           #   piper_server_plan.md=真机 ZMQ server 协议与计划（README.md 为目录索引）
```

## Data Contract (重要)

- 训练数据必须是 **LeRobot v2.1**；v3.0 需先用 `scripts/convert_v3_to_v2_1_aligned.py` 转换。
- 每个 LeRobot 数据集需在 `meta/modality.json` 中声明字段切片（`state`/`action` 的
  `start`/`end`、`video.xxx.original_key`、`annotation...` 任务描述键）；
  示例见 `examples/LIBERO/modality.json`、`examples/SimplerEnv/train_files/bridge_modality.json`。
- `robot_type` 决定 state/action 的字段顺序与归一化方式，注册表在
  `starVLA/dataloader/gr00t_lerobot/data_config.py::ROBOT_TYPE_CONFIG_MAP`
  （已有 `piper`、`libero_franka`、`oxe_bridge`、`oxe_rt1`、`droid_franka` 等）。
- 数据集混合名 `data_mix` 在 `starVLA/dataloader/gr00t_lerobot/mixtures.py::DATASET_NAMED_MIXTURES` 注册。
- 训练产物目录（`checkpoints/<run_id>/`）必须同时包含 `config.yaml`、`config.json`、
  `dataset_statistics.json`：模型载入与动作反归一化都依赖它们，缺一个就会直接报错。

## Environment & Commands

- Python 解释器固定用仓库内 uv 管理的虚拟环境：`.venv/bin/python`（含 `accelerate`；
  搭建/重建步骤见 `doc/04_environment_uv.md`，依赖规格在 `pyproject.toml`）；
  仿真评测的客户端脚本要在**各自仿真环境的 python**（`sim_python`）里跑。
- 权重放仓库内 `weights/`（已进 `.gitignore`）：`weights/Qwen3-VL-2B-Instruct`、
  `weights/vjepa2-vitl-fpc64-256`、`weights/VLA-JEPA/...`；config 里的
  `framework.qwenvl.base_vlm` / `framework.vj2_model.base_encoder` 需指向这些路径。
- 训练**必须**通过 `accelerate launch` 启动，单卡也一样（直接 `python train_starvla.py` 会在
  分布式初始化处失败）；24 GB 卡需 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 常用入口：
  | 目的 | 命令 |
  |---|---|
  | 训练 | `scripts/vlajepa_robot_ft.sh`、`scripts/vlajepa_cotrain.sh` |
  | 开环评估 | `scripts/eval_openloop.py` → `scripts/analyze_openloop.py` |
  | 起推理服务 | `python deployment/model_server/server_policy.py --ckpt_path <run_dir>/final_model/pytorch_model.pt --port <port> --use_bf16` |
  | 仿真评测 | `examples/LIBERO/eval_libero.sh`、`examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh` |
- 详细步骤与排错都写在 `doc/`，不要在别处重复维护命令。

## Conventions

- **代码风格**：black / ruff，`line-length = 121`，`target-version = py310`（见 `pyproject.toml`）；
  Python 要求 `>=3.10`。
- **配置驱动**：新增行为优先做成 config 字段（`scripts/config/*.yaml` 或
  `starVLA/config/training/*.yaml`），不要硬编码路径、维度、端口。
- **工厂/注册表模式**：模型走 `FRAMEWORK_REGISTRY.register("<name>")` + `build_framework(cfg)`，
  数据集走 `ROBOT_TYPE_CONFIG_MAP`，不要新增平行分支式的 if/else 入口。
- **文档语言**：`doc/` 下文档用中文，命令与字段名保留英文原文；
  `doc/README.md` 是流水线索引，新增主文档需同步更新该索引。
- **文档必须可执行**：写进文档的路径、命令、字段名都要来自真实文件（标注来源文件）；
  不确定的内容标 `[待确认]`，不要凭印象编写。
- 提交信息用英文祈使句（如 `Add ...` / `Fix ...`），分支用 `dev/<owner>` 形式。

## Agent Working Rules

1. 改代码前先读相关模块（`starVLA/` 下的实现优先于文档）；文档与代码冲突时以代码为准并指出冲突。
2. 不要提交 `checkpoints/`、`eval_openloop/`、`log/`、数据集与权重；`.gitignore` 已覆盖，
   新增本地工具目录请补进 `.gitignore` 而不是改索引。
3. 不要为了"跑通"而静默改维度、跳过归一化或硬编码形状；形状/字段不一致属于要报告的缺陷。
4. 验证优先于断言：能用脚本验证的（`git status`、最小前向、单样本推理、`--help`）就跑一遍，
   并在结论里给出实际输出或命令。
5. 部署相关改动（server/client 协议、输入输出契约、反归一化）必须同时检查
   `deployment/model_server/` 与 `examples/*/model2*_interface.py` 两侧是否仍然匹配。
6. 真机安全：任何会驱动真实机械臂的脚本，先在仿真或开环评估里验证，再上真机。

## Known Pitfalls (先看这里，省时间)

- `predict_action(..., num_ddim_steps=...)` 这类参数会被 `**kwargs` 吞掉；真正控制动作头迭代步数的是
  `model.action_model.num_inference_timesteps`。
- 动作头输出的是**归一化动作**，必须用 `dataset_statistics.json` 反归一化后才能下发；
  不同数据集的统计量键名不同（`q01/q99` vs `min/max`）。
- 仿真客户端与服务端可能在不同 python 环境里运行，`PYTHONPATH` 必须同时包含 LIBERO/SimplerEnv 与本仓库。
- 部分 `examples/SimplerEnv/eval_files/auto_eval_scripts/star_*.sh` 里存在变量名不一致的历史问题，
  照抄命令前先核对（详见 `doc/robot/`）。
- 真机 Piper 的默认部署配置是 `state_type=joint` + `absolute_action=True`（绝对关节角，7 维 = 6 关节 + 夹爪）；
  客户端出厂默认的 `state_type=qpos` 会送错 state 并把绝对动作当增量累加（详见 `doc/robot/deployment.md` §4.2）。
