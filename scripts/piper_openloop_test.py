# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
Piper 真机开环测试（离线验证版）

用途：
    不连接真实机械臂，直接从 LeRobot v2.1 数据集读取真实观测
    （双相机图 image + wrist_image、state、语言指令），喂给训练好的
    VLA-JEPA checkpoint，让模型预测动作 chunk，并与数据集中的
    ground-truth 动作对比，统计 MAE / MSE 等指标。

    典型用法（适配 piper_test.yaml 训练出的模型）：
        python scripts/piper_openloop_test.py \
            --ckpt_path <run_root>/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
            --data_root_dir /path/to/pick_place_block_all_0408_1_offset_state_v2_1_full \
            --data_mix piper_pick_place \
            --num_episodes 5 \
            --stride 1 \
            --max_steps_per_episode 50

说明：
    * checkpoint 需要与 `config.yaml`、`dataset_statistics.json` 放在同一个 run 目录下
      （即 ckpt 路径形如 .../<run_id>/checkpoints/steps_XXX.pt）。
    * 归一化统计量取自 checkpoint 自带的 dataset_statistics.json，与训练一致。
    * Piper 训练采用 min_max 归一化（joint_1~6 + gripper 的 7 维绝对值），
      反归一化公式为 raw = 0.5*(norm+1)*(max-min)+min。
"""

import argparse
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


# ---------------------------------------------------------------------------
# 归一化 / 反归一化（与 PiperDataConfig 的 min_max 模式保持一致）
# ---------------------------------------------------------------------------
def unnormalize_min_max(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of `Normalizer.forward(mode="min_max")`.

    normalized: shape (T, D) or (D,) in [-1, 1]
    stats: {"min": [...], "max": [...]} per-dimension
    """
    normalized = np.asarray(normalized, dtype=np.float32)
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    return 0.5 * (normalized + 1.0) * (high - low) + low


# ---------------------------------------------------------------------------
# 观测构造（与 LeRobotMixtureDataset.__getitem__ 对齐）
# ---------------------------------------------------------------------------
def build_observation(dataset, trajectory_id: int, step: int, cfg) -> dict:
    """Reconstruct a model-ready observation for a specific (trajectory, step).

    Returns dict with:
        images : list[PIL.Image]        双视角，已缩放到 resolution_size
        lang   : str                    语言指令
        state  : np.ndarray shape (1, state_dim)  归一化后的 proprio state
        action_norm : np.ndarray shape (action_horizon, action_dim)  GT 归一化动作 chunk
    """
    data = dataset.transforms(dataset.get_step_data(trajectory_id, step))

    resolution_size = cfg.datasets.vla_data.resolution_size
    video_resolution_size = cfg.datasets.vla_data.video_resolution_size

    images = []
    for video_key in dataset.modality_keys["video"]:
        video = data[video_key]  # (T, H, W, C) uint8
        video = _resize_video(video, video_resolution_size)
        images.append(Image.fromarray(video[0]).resize((resolution_size, resolution_size)))

    lang = data[dataset.modality_keys["language"][0]][0]

    action = []
    for action_key in dataset.modality_keys["action"]:
        action.append(np.asarray(data[action_key], dtype=np.float32))
    action = np.concatenate(action, axis=1)  # (action_horizon, action_dim)

    sample = {"images": images, "lang": lang, "action_norm": action}

    if cfg.datasets.vla_data.get("with_state", False):
        state = []
        for state_key in dataset.modality_keys["state"]:
            state.append(np.asarray(data[state_key], dtype=np.float32))
        state = np.concatenate(state, axis=1).astype(np.float32)
        sample["state"] = state[0:1]  # (1, state_dim)

    return sample


def _resize_video(video: np.ndarray, N: int) -> np.ndarray:
    import cv2
    T, H, W, C = video.shape
    resized = np.zeros((T, N, N, C), dtype=video.dtype)
    for t in range(T):
        resized[t] = cv2.resize(video[t], (N, N), interpolation=cv2.INTER_LINEAR)
    return resized


# ---------------------------------------------------------------------------
# 指标统计
# ---------------------------------------------------------------------------
class MetricAccumulator:
    def __init__(self, horizon: int, action_dim: int):
        self.horizon = horizon
        self.action_dim = action_dim
        self.step_mae_norm = []  # per step, first-step (horizon==0) MAE in normalized space
        self.chunk_mae_norm = []
        self.chunk_mae_raw = []
        self.chunk_mse_raw = []
        # per-horizon-step accumulators (raw space)
        self.h_step_mae_raw = np.zeros(horizon)
        self.h_step_cnt = np.zeros(horizon)
        # per-dimension accumulators (raw space)
        self.dim_mae_raw = np.zeros(action_dim)
        self.dim_cnt = 0

    def add(self, pred_norm: np.ndarray, gt_norm: np.ndarray, stats: dict) -> None:
        """pred_norm / gt_norm: (horizon, action_dim)"""
        pred_norm = np.asarray(pred_norm, dtype=np.float32)
        gt_norm = np.asarray(gt_norm, dtype=np.float32)

        # normalized-space metrics
        self.step_mae_norm.append(np.abs(pred_norm[0] - gt_norm[0]).mean())
        self.chunk_mae_norm.append(np.abs(pred_norm - gt_norm).mean())

        # raw-space metrics
        pred_raw = unnormalize_min_max(pred_norm, stats)
        gt_raw = unnormalize_min_max(gt_norm, stats)
        err_raw = np.abs(pred_raw - gt_raw)
        self.chunk_mae_raw.append(err_raw.mean())
        self.chunk_mse_raw.append((err_raw ** 2).mean())
        self.h_step_mae_raw += err_raw.mean(axis=1)
        self.h_step_cnt += 1
        self.dim_mae_raw += err_raw.mean(axis=0)
        self.dim_cnt += 1

    def summarize(self) -> dict:
        horizon_mae = None
        if self.h_step_cnt.sum() > 0:
            horizon_mae = (self.h_step_mae_raw / np.maximum(self.h_step_cnt, 1)).tolist()
        return {
            "first_step_mae_normalized": float(np.mean(self.step_mae_norm)) if self.step_mae_norm else None,
            "chunk_mae_normalized": float(np.mean(self.chunk_mae_norm)) if self.chunk_mae_norm else None,
            "chunk_mae_raw": float(np.mean(self.chunk_mae_raw)) if self.chunk_mae_raw else None,
            "chunk_mse_raw": float(np.mean(self.chunk_mse_raw)) if self.chunk_mse_raw else None,
            "per_horizon_step_mae_raw": horizon_mae,
            "per_dim_mae_raw": (self.dim_mae_raw / max(self.dim_cnt, 1)).tolist(),
            "num_chunks": self.dim_cnt,
        }


def _resolve_unnorm_key(norm_stats: dict, unnorm_key: str | None) -> str:
    if unnorm_key is not None:
        assert unnorm_key in norm_stats, (
            f"`unnorm_key` {unnorm_key} not in norm_stats, available: {list(norm_stats.keys())}"
        )
        return unnorm_key
    assert len(norm_stats) == 1, (
        f"模型在多个数据集上训练，请通过 `--unnorm_key` 指定统计量 key: {list(norm_stats.keys())}"
    )
    return next(iter(norm_stats.keys()))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def build_dataset(cfg, data_root_dir, data_mix, robot_type, action_horizon, video_horizon):
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    for d_name, _w, _rt in mixture_spec:
        if _rt != robot_type:
            continue
        dataset = make_LeRobotSingleDataset(
            Path(data_root_dir),
            d_name,
            robot_type,
            delete_pause_frame=False,
            action_horizon=action_horizon,
            video_horizon=video_horizon,
        )
        return dataset
    raise ValueError(
        f"No dataset with robot_type={robot_type} found in data_mix={data_mix}: {mixture_spec}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper 真机开环测试（离线验证）")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="训练好的权重路径，如 <run_root>/<run_id>/checkpoints/steps_20000_pytorch_model.pt")
    parser.add_argument("--data_root_dir", type=str, default=None,
                        help="LeRobot v2.1 数据集根目录（默认取 checkpoint 配置中的 data_root_dir）")
    parser.add_argument("--data_mix", type=str, default=None,
                        help="数据混合名（默认取 checkpoint 配置中的 data_mix）")
    parser.add_argument("--robot_type", type=str, default="piper")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="dataset_statistics.json 中的统计量 key（默认：只有一个数据集时自动选择）")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="测试前 N 个 episode（默认全部）")
    parser.add_argument("--episode_ids", type=str, default=None,
                        help="指定要测试的 episode id，逗号分隔（优先于 num_episodes）")
    parser.add_argument("--stride", type=int, default=1,
                        help="每隔多少步取一个观测（默认 1 = 逐帧）")
    parser.add_argument("--max_steps_per_episode", type=int, default=None,
                        help="每个 episode 最多测试的步数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                        help="以 bf16 加载模型（默认开启）")
    parser.add_argument("--save_dir", type=str, default="openloop_results",
                        help="结果输出目录")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="保存每个 episode 的预测动作 npy")
    parser.add_argument("--save_visual", action="store_true", default=True,
                        help="保存预测 vs GT 轨迹对比图")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 加载模型 ----------------
    print(f"[1/4] Loading checkpoint: {args.ckpt_path}")
    vla = baseframework.from_pretrained(args.ckpt_path)
    device = torch.device(args.device)
    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    cfg = vla.config
    action_dim = int(cfg.framework.action_model.action_dim)
    action_horizon = int(cfg.framework.action_model.action_horizon)
    video_horizon = int(cfg.framework.vj2_model.num_frames)
    with_state = bool(cfg.datasets.vla_data.get("with_state", False))

    data_root_dir = args.data_root_dir or cfg.datasets.vla_data.data_root_dir
    data_mix = args.data_mix or cfg.datasets.vla_data.data_mix
    print(f"    action_dim={action_dim}, action_horizon={action_horizon}, "
          f"video_horizon={video_horizon}, with_state={with_state}")
    print(f"    data_root_dir={data_root_dir}, data_mix={data_mix}")

    # 取反归一化统计量（min_max 模式）
    _, norm_stats = read_mode_config(args.ckpt_path)
    unnorm_key = _resolve_unnorm_key(norm_stats, args.unnorm_key)
    action_stats = norm_stats[unnorm_key]["action"]
    assert "min" in action_stats and "max" in action_stats, (
        f"dataset_statistics.json 中 action 缺少 min/max，实际 keys: {list(action_stats.keys())}"
    )
    assert len(action_stats["min"]) == action_dim, (
        f"action 统计量维度 {len(action_stats['min'])} 与 action_dim {action_dim} 不一致"
    )

    # ---------------- 构建数据集 ----------------
    print("[2/4] Building LeRobot dataset ...")
    dataset = build_dataset(cfg, data_root_dir, data_mix, args.robot_type,
                            action_horizon, video_horizon)
    print(f"    dataset: {dataset}")

    episode_ids = None
    if args.episode_ids:
        episode_ids = [int(x) for x in args.episode_ids.split(",")]
    else:
        num = args.num_episodes or len(dataset.trajectory_ids)
        episode_ids = list(dataset.trajectory_ids[:num])

    # ---------------- 开环推理 ----------------
    print("[3/4] Running open-loop inference ...")
    accumulator = MetricAccumulator(action_horizon, action_dim)
    per_episode_results = {}

    for ep in episode_ids:
        traj_idx = dataset.get_trajectory_index(ep)
        length = int(dataset.trajectory_lengths[traj_idx])
        max_steps = min(args.max_steps_per_episode or (length - action_horizon + 1),
                        length - action_horizon + 1)
        if max_steps <= 0:
            print(f"    [skip] episode {ep}: length={length} < action_horizon={action_horizon}")
            continue

        ep_pred_raw, ep_gt_raw = [], []
        ep_steps = list(range(0, max_steps, args.stride))

        for t in ep_steps:
            sample = build_observation(dataset, ep, t, cfg)
            batch_images = [sample["images"]]
            instructions = [sample["lang"]]
            kwargs = {"use_ddim": True, "num_ddim_steps": 10}
            if with_state:
                kwargs["state"] = [sample["state"]]

            with torch.no_grad():
                output = vla.predict_action(
                    batch_images=batch_images,
                    instructions=instructions,
                    **kwargs,
                )
            pred_norm = np.asarray(output["normalized_actions"][0], dtype=np.float32)  # (H, D)
            gt_norm = sample["action_norm"]

            accumulator.add(pred_norm, gt_norm, action_stats)

            ep_pred_raw.append(unnormalize_min_max(pred_norm, action_stats))
            ep_gt_raw.append(unnormalize_min_max(gt_norm, action_stats))

        per_episode_results[ep] = {
            "num_steps": len(ep_steps),
            "length": length,
        }

        if args.save_predictions:
            np.savez(
                save_dir / f"episode_{ep:06d}.npz",
                pred_actions=np.stack(ep_pred_raw),  # (num_steps, H, D)
                gt_actions=np.stack(ep_gt_raw),
                steps=np.array(ep_steps),
            )

        if args.save_visual:
            _save_visual(save_dir, ep, ep_pred_raw, ep_gt_raw, ep_steps)

        print(f"    episode {ep}: {len(ep_steps)} chunks evaluated (length={length})")

    # ---------------- 汇总指标 ----------------
    print("[4/4] Summarizing metrics ...")
    summary = accumulator.summarize()
    summary["episodes_evaluated"] = len(episode_ids)
    summary["num_chunks_per_episode"] = per_episode_results

    out_path = save_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {out_path}")

    print("\n================= 开环测试结果 =================")
    print(f"  First-step MAE (normalized):  {summary['first_step_mae_normalized']:.4f}")
    print(f"  Chunk MAE     (normalized):   {summary['chunk_mae_normalized']:.4f}")
    print(f"  Chunk MAE     (raw joint):    {summary['chunk_mae_raw']:.4f}")
    print(f"  Chunk MSE     (raw joint):    {summary['chunk_mse_raw']:.4f}")
    print(f"  Per-dim MAE   (raw joint):    "
          + ", ".join(f"{k}={v:.4f}" for k, v in zip(ACTION_DIM_LABELS, summary["per_dim_mae_raw"])))
    print(f"  Per-horizon-step MAE (raw):   "
          + ", ".join(f"h{i}={v:.4f}" for i, v in enumerate(summary["per_horizon_step_mae_raw"])))
    print("=================================================")


def _save_visual(save_dir: Path, ep: int, pred_raw: list, gt_raw: list, steps: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.stack(pred_raw)  # (num_steps, H, D)
    gt = np.stack(gt_raw)

    fig, axes = plt.subplots(len(ACTION_DIM_LABELS), 1, figsize=(12, 2.2 * len(ACTION_DIM_LABELS)),
                             sharex=True)
    for d, label in enumerate(ACTION_DIM_LABELS):
        ax = axes[d]
        ax.plot(steps, gt[:, 0, d], label="GT (first-step)", linestyle="--", alpha=0.8)
        ax.plot(steps, pred[:, 0, d], label="Pred (first-step)", alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Episode {ep}: Predicted vs GT first-step actions (raw)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / f"episode_{ep:06d}.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
