"""Offline replay for the Piper ZMQ deployment path.

This validates the deployment-side observation path without opening a socket or
contacting a robot: dataset samples are encoded and decoded with the real Piper
wire format, then passed through ``VLAJepaPiperPolicy``.  The returned physical
actions are compared with the dataset action chunk in physical and min-max
normalized spaces.
"""

# The real-robot examples are intentionally imported after adding their directory
# to sys.path; Ruff's normal import ordering does not apply to this boundary.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps

from starVLA.dataloader.lerobot_datasets import get_vla_dataset

REAL_ROBOT_DIR = Path(__file__).resolve().parents[1] / "examples" / "real-robot"
sys.path.insert(0, str(REAL_ROBOT_DIR))

from policy import VLAJepaPiperPolicy  # noqa: E402
from wire import pack_payload, unpack_payload  # noqa: E402


DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Piper ZMQ deployment replay")
    parser.add_argument("--config_yaml", required=True, help="training config used by the dataset")
    parser.add_argument("--checkpoint", required=True, help="VLA-JEPA pytorch checkpoint")
    parser.add_argument("--output_dir", required=True, help="directory for replay.json")
    parser.add_argument("--windows_per_episode", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=1, help="0 = all episodes")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--cuda", default="0", help="CUDA device visible to this process")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--num-inference-timesteps", type=int, default=0)
    parser.add_argument("--chunk-steps", type=int, default=0)
    parser.add_argument("--unnorm-key", default="")
    parser.add_argument("--default-instruction", default="")
    parser.add_argument(
        "--no-binarize-gripper",
        action="store_true",
        help="关闭二值化，并按 Piper min/max 统计量比较连续 gripper 物理值",
    )
    return parser.parse_args()


def select_windows(dataset, windows_per_episode: int, num_episodes: int) -> list[tuple[int, int, int]]:
    """Use the same deterministic episode/window selection as eval_openloop."""
    if windows_per_episode < 1:
        raise ValueError("--windows_per_episode must be >= 1")
    if num_episodes < 0:
        raise ValueError("--num_episodes must be >= 0")

    by_traj: dict[int, list[tuple[int, int]]] = {}
    for index in range(len(dataset)):
        _, traj_id, step = dataset.sample_step(index)
        by_traj.setdefault(int(traj_id), []).append((int(step), index))

    selected: list[tuple[int, int, int]] = []
    for traj_id in sorted(by_traj)[: num_episodes or None]:
        items = sorted(by_traj[traj_id])
        picks = np.linspace(0, len(items) - 1, windows_per_episode).round().astype(int)
        for pick in sorted(set(picks.tolist())):
            step, index = items[pick]
            selected.append((traj_id, step, index))
    return selected


def _stat_value(entry, name: str) -> float:
    value = entry[name] if isinstance(entry, dict) else getattr(entry, name)
    return float(np.ravel(value)[0])


def dataset_action_stats(dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    stats = dataset.datasets[0].metadata.statistics.action
    keys = list(stats.keys())
    lo = np.array([_stat_value(stats[key], "min") for key in keys], dtype=np.float64)
    hi = np.array([_stat_value(stats[key], "max") for key in keys], dtype=np.float64)
    if len(keys) != len(DIM_NAMES):
        raise ValueError(f"Piper replay expects 7 action dimensions, got {keys}")
    return lo, hi, keys


def minmax_normalize(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.maximum(hi - lo, 1e-8)
    return 2.0 * (actions - lo) / span - 1.0


def item_to_wire_request(item: dict, jpeg_quality: int) -> dict:
    images = item["image"]
    if len(images) != 2:
        raise ValueError(f"Piper replay expects head+wrist images, got {len(images)} views")

    def client_resize(image: Image.Image) -> np.ndarray:
        # Match vla_infer's adaptive_resize_image: keep aspect ratio and pad white.
        resized = ImageOps.pad(
            image.convert("RGB"),
            (224, 224),
            method=Image.Resampling.BILINEAR,
            color=(255, 255, 255),
        )
        return np.asarray(resized, dtype=np.uint8)

    payload = {
        "image": client_resize(images[0]),
        "wrist_image": client_resize(images[1]),
        "state": np.asarray(item["state"], dtype=np.float32).reshape(-1),
        "cmd": str(item["lang"]),
    }
    # This is the exact client-side JPEG/msgpack boundary used by vla_infer.
    return unpack_payload(pack_payload(payload, jpeg_quality=jpeg_quality))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_yaml)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        seed=int(cfg.get("seed", 42)),
    )
    windows = select_windows(dataset, args.windows_per_episode, args.num_episodes)
    if not windows:
        raise RuntimeError("no replay windows selected")

    lo, hi, action_keys = dataset_action_stats(dataset)
    policy = VLAJepaPiperPolicy(
        ckpt_path=args.checkpoint,
        device=f"cuda:{args.cuda}",
        use_bf16=args.use_bf16,
        num_inference_timesteps=args.num_inference_timesteps,
        chunk_steps=args.chunk_steps,
        unnorm_key=args.unnorm_key,
        default_instruction=args.default_instruction,
        binarize_gripper=not args.no_binarize_gripper,
    )
    policy.load()
    policy.warmup()

    predictions, targets, rows, latencies = [], [], [], []
    for episode, step, index in windows:
        item = dataset[index]
        request = item_to_wire_request(item, args.jpeg_quality)
        started = time.perf_counter()
        response = policy.predict_request(request)
        latency_ms = (time.perf_counter() - started) * 1000.0

        pred_raw = np.asarray(response["action"], dtype=np.float64)
        gt_norm = np.asarray(item["action"], dtype=np.float64)
        # Piper's data_config applies min_max to all seven action keys. The
        # checkpoint's action mask is a model-output convention and is not the
        # dataset transform mask used to reconstruct this ground truth.
        gt_raw = (gt_norm + 1.0) / 2.0 * (hi - lo) + lo
        if pred_raw.shape != gt_raw.shape:
            raise ValueError(f"action shape mismatch at ep={episode} step={step}: {pred_raw.shape} vs {gt_raw.shape}")
        if not np.isfinite(pred_raw).all() or not np.isfinite(gt_raw).all():
            raise ValueError(f"non-finite action at ep={episode} step={step}")

        pred_norm = minmax_normalize(pred_raw, lo, hi)
        norm_error = np.abs(pred_norm - gt_norm)
        raw_error = np.abs(pred_raw - gt_raw)
        predictions.append(pred_raw)
        targets.append(gt_raw)
        latencies.append(latency_ms)
        rows.append(
            {
                "episode": episode,
                "step": step,
                "dataset_index": index,
                "latency_ms": latency_ms,
                "normalized_mae": float(norm_error.mean()),
                "raw_mae": float(raw_error.mean()),
                "raw_mae_per_dim": raw_error.mean(axis=0).tolist(),
                "action_first": pred_raw[0].tolist(),
                "action_last": pred_raw[-1].tolist(),
            }
        )
        print(f"replay {len(rows)}/{len(windows)} ep={episode} step={step} latency={latency_ms:.0f}ms")

    pred = np.stack(predictions)
    gt = np.stack(targets)
    norm_error = np.abs(minmax_normalize(pred, lo, hi) - minmax_normalize(gt, lo, hi))
    raw_error = np.abs(pred - gt)
    result = {
        "status": "complete",
        "source_script": "scripts/piper_zmq_replay.py",
        "checkpoint": args.checkpoint,
        "config_yaml": args.config_yaml,
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "wire": {"jpeg_quality": args.jpeg_quality, "images": ["image", "wrist_image"]},
        "policy": {
            "device": f"cuda:{args.cuda}",
            "use_bf16": args.use_bf16,
            "num_inference_timesteps": int(policy.model.action_model.num_inference_timesteps),
            "chunk_steps": policy.chunk_steps,
            "binarize_gripper": not args.no_binarize_gripper,
        },
        "action_keys": action_keys,
        "policy_action_stats_keys": sorted(policy.action_stats),
        "num_windows": len(rows),
        "num_episodes": len({row["episode"] for row in rows}),
        "chunk_len": int(pred.shape[1]),
        "action_dim": int(pred.shape[2]),
        "normalized": {
            "mae": float(norm_error.mean()),
            "mae_per_dim": norm_error.mean(axis=(0, 1)).tolist(),
        },
        "raw": {
            "mae": float(raw_error.mean()),
            "mae_per_dim": raw_error.mean(axis=(0, 1)).tolist(),
            "dataset_min": lo.tolist(),
            "dataset_max": hi.tolist(),
            "dataset_action_transform": {"mode": "min_max", "inverse_all_dims": True},
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        },
        "windows": rows,
    }
    with open(out_dir / "replay.json", "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({key: result[key] for key in ("num_windows", "normalized", "raw", "latency_ms")}, indent=2))
    print(f"written -> {out_dir / 'replay.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
