# 真机数据（adjust_cup_0409）训练说明与冒烟测试报告

本文档记录将真机数据集 `adjust_cup_0409_1_offset_state_converted` 接入 VLA-JEPA
第二阶段（Robot Fine-tuning）的全过程与冒烟测试结果。

- 原始数据：`/share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_converted`
- 转换后数据：`/share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_v2_1`
- 转换脚本：`scripts/convert_v3_to_v2_1_aligned.py`
- 训练配置：`scripts/config/adjust_cup_test.yaml`（冒烟）、`scripts/config/adjust_cup_train.yaml`（1 万步正式训练）
- 训练日志：`log/`（`adjust_cup_10k.log` 等；启动训练时请把 stdout 重定向到这里）
- 开环测试：`scripts/eval_openloop.py`；loss 分析：`scripts/analyze_loss.py`

---

## 1. 数据情况

| 项目 | 值 |
|---|---|
| 原始格式 | LeRobot **v3.0**（多 episode 混存在同一个 parquet / mp4） |
| 规模 | 50 episodes / 6837 帧 / 33 条任务描述 / 10 fps |
| 图像 | 双视角 `observation.images.image` + `observation.images.wrist_image`，224×224，AV1 编码 |
| 低维 | state 7 维、action 7 维（Piper 6 关节 + 夹爪） |
| 额外列 | `observation.tracks_image` / `observation.tracks_wrist_image`（784×3，训练不需要） |

VLA-JEPA 只支持 LeRobot **v2.1**，因此 v3.0 必须先转换。

关键校验：源视频解码共 6837 帧、pts 严格等于 `i × 0.1s`，与 `meta/episodes` 中的
`from_timestamp` / `to_timestamp` 完全对齐 → 可以按帧号安全切片。

## 2. 格式转换

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA
python scripts/convert_v3_to_v2_1_aligned.py \
  --input  /share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_converted \
  --output /share/home/tm866052366100000/a926312360/LXX/project/Datasets/ICLR_real_world/adjust_cup_0409_1_offset_state_v2_1 \
  --fps 10 --chunk-size 100
```

与旧脚本 `convert_v3_to_v2_1.py` 的区别（旧脚本对本数据集会出错，见下）：

1. **按帧号切片**：单遍顺序解码每个相机视频，用 `from_timestamp × fps` 作为起始帧、
   `length` 作为长度切片，不再对浮点 PTS 做精确匹配；
2. **修复多 parquet 文件的全局索引 bug**：`dataset_from_index/to_index` 是**全局**行号，
   旧脚本直接用它切当前文件，导致第二个数据文件中的 episode 全部取空
   （本数据集有 `file-000`/`file-001` 两个数据文件，旧脚本只会得到 5034/6837 行）；
3. 丢弃 `observation.tracks_*` 等训练用不到的列；
4. 写出真实的 `tasks.jsonl` 文本，以及 VLA-JEPA 需要的 `modality.json`（`x/y/z/roll/pitch/yaw/gripper`
   + 两路 video + `task_index` 标注）与带 `info` 块的 `info.json`；
5. 结束时清理 `meta/steps_*.pkl` 采样缓存，避免其它数据集残留的缓存串味。

转换耗时约 42 秒。校验结果：

```
episodes 50 / frames 6837 / tasks 33
parquet 行数  : 50/50 全部等于 episode length
视频帧数      : 50×2 全部等于 episode length
像素对齐抽查  : ep 0/17/30/49 首末帧与源视频 mean|diff| ≈ 1.3/255（仅 H.264 重编码噪声）
```

## 3. 冒烟测试

### 3.1 单卡（数据 + 模型 + 保存全链路）

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/accelerate_test.yaml \
  --main_process_port 29504 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml
```

配置：`per_device_batch_size: 2`、`max_train_steps: 20`、`num_warmup_steps: 5`、
`eval_interval: 10`、ZeRO-2 + CPU offload。

结果（**通过**）：

```
参数量      : 2770.332 M
dataloader  : 6837 steps / 50 trajectories
              image list[2]×(224,224), video (2,8,256,256,3) uint8,
              action (7,7), state (1,7), lang = "Put the cup ..."
step  1  action_loss=1.4233  wm_loss=0.1934
step  5  action_loss=1.3848  wm_loss=0.1937
step 10  action_loss=1.0884  wm_loss=0.1855  mse=0.1128  mae=0.9276
step 15  action_loss=1.3578  wm_loss=0.1882
step 20  action_loss=1.4485  wm_loss=0.1857  mse=0.1087  mae=0.9131
final_model/pytorch_model.pt 6.16 GB 已保存
```

- 无 NaN / Inf，无异常退出，退出码 0；
- `wm_loss` 由 0.193 缓降至 0.186，`mae_score` 0.928 → 0.913；
- 单步约 4.5 s（含 CPU offload），20 步约 2 分钟。

### 3.2 四卡 ZeRO-2（正式训练命令验证）

```bash
cd /share/home/tm866052366100000/a926312360/LXX/project/VLA-JEPA && \
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/VLA_JEPA/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29505 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/adjust_cup_test.yaml \
  --run_id adjust_cup_smoke_4gpu --trainer.max_train_steps 6 --trainer.eval_interval 100
```

结果（**通过**）：6/6 步完成、退出码 0、LR warmup 正常爬升（2e-6 → 1e-5），
final_model 已保存，显存无 OOM（ZeRO-2 优化器分片，关闭 CPU offload 也可行）。

## 4. 正式训练

复制 `scripts/config/adjust_cup_test.yaml` 后至少修改：

| 参数 | 建议值 | 说明 |
|---|---|---|
| `run_id` | 自定义 | 输出目录 `checkpoints/<run_id>/` |
| `trainer.max_train_steps` | 30000~50000 | 本数据集仅 6837 帧，步数不宜过大 |
| `datasets.vla_data.per_device_batch_size` | 4~8 | 4 卡 8 时总 batch 32；OOM 则调小 |
| `trainer.save_interval` | 10000 | 需小于 `max_train_steps` 才会存 |
| `trainer.num_warmup_steps` | 5000（或 warmup_ratio 0.1） | 测试配置里只有 5 步，正式训练务必调大 |
| `trainer.pretrained_checkpoint` | 可选 | 加载一阶段 co-training 权重可提升效果 |

启动命令同 3.2，把 `--run_id` / 步数换成正式值即可。

## 5. 已知注意事项

1. **采样缓存串味风险**：`/meta/steps_*.pkl` 是硬编码文件名（`steps_332420bad1ab.pkl` /
   `steps_2d5a34b904d2.pkl`）。复制数据集时必须删掉它，否则会加载到别的数据集的采样索引。
   本转换脚本已自动清理。
2. **多卡采样重复**：`get_vla_dataset()` 的 `seed` 固定为 42，各 rank 的
   `(epoch, index, seed)` 相同 → 同一步上 4 张卡拿到的是**同一条 (trajectory, step)**，
   只有数据增强的随机性不同，多卡数据多样性低于预期。如需修正，可在
   `lerobot_datasets.py::get_vla_dataset` 里传入 `seed=cfg.seed + rank`，或给 DataLoader
   加 `DistributedSampler`。
3. **`eval_interval` 会在 step 0 触发一次评估**（`completed_steps % eval_interval == 0`），
   评估内部跑 20 步 DDIM，首次启动会多花一点时间。
4. **训练结束必存权重**：`_finalize_training()` 无条件保存 `final_model/pytorch_model.pt`（约 6.16 GB），
   即使只是冒烟测试也会落盘。本次冒烟产生的两个目录可以随时删除：
   `checkpoints/adjust_cup_smoke/`、`checkpoints/adjust_cup_smoke_4gpu/`。
5. **数据集统计**：首次训练会在数据集 `meta/` 下生成 `stats_gr00t.json` 与 `steps_*.pkl`
   （需要数据集目录可写）。
