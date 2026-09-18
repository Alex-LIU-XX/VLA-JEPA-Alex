# doc/robot —— 部署文档索引

本目录回答两个问题：**模型怎么用**（契约）与 **模型怎么跑起来接到目标上**（部署）。
按你的目标选入口，不要一次读三份。

| # | 文档 | 回答什么问题 | 什么时候看 |
|---|---|---|---|
| 1 | [**model.md**](./model.md) · VLA-JEPA 模型文档 | 怎么载入权重、输入是什么形状/格式、数据怎么定义、输出是什么、怎么反归一化 | 要写推理代码 / 排查"模型吃进去的东西对不对" |
| 2 | [**deployment.md**](./deployment.md) · 部署文档（真机 + 仿真） | 起服务的命令、仿真评测（LIBERO / LIBERO-Plus / SimplerEnv）、真机 Piper 默认配置与上机清单、脚本索引与排查表 | 要把权重跑起来、要上机、要照抄评测脚本 |
| 3 | [**piper_server_plan.md**](./piper_server_plan.md) · 真机 ZMQ server 计划 | 控制端 `vla_infer` 的 ZMQ 协议逐条理解、与本仓库 websocket 的差异、VLA-JEPA server 的实现方案与验收标准 | 要开发/维护真机侧 server（`deployment/piper_zmq/`） |

---

## 推荐阅读路径

- **只想用模型（写推理/评估代码）** → [model.md](./model.md)（§0 速览 → §2 载入 → §3 输入 → §5 输出）。
- **要跑仿真评测** → [deployment.md](./deployment.md)（§2 最小验证 → §3 仿真 + §3.5 脚本索引）。
- **要上真机 Piper** → [deployment.md](./deployment.md) §4（默认配置 + 检查清单），
  server 未就绪时先看 [piper_server_plan.md](./piper_server_plan.md)。
- **要改/扩展部署链路**（协议、客户端、反归一化）→ 两份都读：
  model.md 定契约，deployment.md 定落地方式。

---

## 样例代码

- **真机 server 样例**：[`examples/real-robot/`](../../examples/real-robot/README.md) ——
  `wire.py`（线格式）、`transport.py`（ZMQ REP + 异常兜底）、`policy.py`（观测→动作）、
  `piper_zmq_server.py`（入口）、`selftest_client.py`（协议自检，无需机械臂/GPU）。
- **真机离线回放**：[`scripts/piper_zmq_replay.py`](../../scripts/piper_zmq_replay.py) ——
  用真实权重复现客户端的 letterbox + JPEG/msgpack 观测路径，不连接 ZMQ 或机械臂。
- **控制端示例**（独立仓库）：`~/repo/Double_Piper_Teleop/vla_infer/example/vlajepa/vlajepa_piper_client.py`
  —— 默认值即本项目约定（`state_type=joint` + `absolute_action=True` + `enable_gripper_transform=False`
  + `execute_chunk_steps=7`）。

---

## 关键约定速查（最容易踩的 6 条）

1. **权重目录三件套**：`.pt` 的上一级目录必须同时有 `config.yaml` 与 `dataset_statistics.json`，
   否则载入直接失败 → [model.md](./model.md) §2.1
2. **输出是归一化动作**：必须用 `dataset_statistics.json` 反归一化才能下发；
   统计量键名分 `min/max` 与 `q01/q99` 两套，**不可混用** → [model.md](./model.md) §5.2
3. **Piper 数据是 6 关节角 + 夹爪**（`x/y/z/roll/pitch/yaw` 只是标签名，不是末端位姿），
   且 state 不做归一化 → [model.md](./model.md) §3.4、§4.3
4. **真机默认配置**：`state_type=joint` + `absolute_action=True`（绝对关节角）；
   客户端出厂默认是 `qpos` + 增量解释，会出错 → [deployment.md](./deployment.md) §4.2
5. **两套协议互不兼容**：仿真用 websocket + `__ndarray__` 版 msgpack；真机用 ZMQ + 真 `msgpack_numpy`（`b'nd'`）
   → [deployment.md](./deployment.md) §1.3
6. **`examples/` 脚本的隐含前提**：CWD 必须是仓库根、`sim_python` 是仿真环境而 server 要用 VLA 环境、
   部分 `star_*.sh` 变量名有 bug → [deployment.md](./deployment.md) §3.5、§5

---

## 本目录与其他文档的关系

| 文档 | 内容 |
|---|---|
| `README.md`（仓库根） | 项目总览、官方评测（LIBERO / LIBERO-Plus / SimplerEnv） |
| `doc/01_data_conversion.md` | v3 → v2.1 数据转换、robot_type 注册、**Piper 关节角语义（§6.3）**、指令占位符缺陷（§6.4） |
| `doc/02_training.md` | 训练启动与排错（含配置字段逐项说明） |
| `doc/03_openloop_testing.md` | 开环测试与指标解读（上机前的离线验证手段） |
| **`doc/robot/`（本目录）** | 模型契约 + 部署落地 + 真机 server 计划 |

> 写作约定：本目录文档用中文，命令/字段名保留英文原文；每条结论标注 `[已确认]`（附 `文件:行`）/
> `[建议]` / `[待确认]`。新增或调整内容后，请同步更新本索引与 [deployment.md](./deployment.md) 的脚本索引表。
