"""Offline open-loop rollout through the real Piper ZMQ server/client path.

The server is an external ``piper_zmq_server.py`` process. This client replays
dataset observations frame by frame, extracts the returned action chunk, and
persists both the full chunk comparison and the step-0 open-loop trajectory.
The existing ``plot_openloop_trajectory.py`` renderer is then used to create
the trajectory figures from the persisted prediction record.

This is an offline server-client test only: no robot or actuator is connected.
"""

# Importing the shared dataset client helpers is intentional; it keeps image
# resizing and observation construction identical to selftest_client.py.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import zmq
from omegaconf import OmegaConf

from piper_zmq_dataset_client_test import describe_action, item_to_observation
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REAL_ROBOT_DIR = SCRIPT_DIR.parent / "examples" / "real-robot"
sys.path.insert(0, str(REAL_ROBOT_DIR))

from wire import pack_payload, unpack_payload  # noqa: E402


DIM_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
EXPECTED_ACTION_DIM = len(DIM_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Piper ZMQ open-loop client")
    parser.add_argument("--config_yaml", required=True, help="training config used to construct the dataset")
    parser.add_argument("--output_dir", required=True, help="directory for open-loop artifacts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-episodes", type=int, default=1, help="number of first episodes; 0 = all")
    parser.add_argument("--stride", type=int, default=1, help="send every N-th frame in each selected episode")
    parser.add_argument("--max-steps", type=int, default=0, help="per-episode frame limit; 0 = full episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--server-gripper-mode",
        choices=("continuous", "binary"),
        default="continuous",
        help="metadata describing the server invocation; use continuous with --no-binarize-gripper",
    )
    parser.add_argument("--skip-plot", action="store_true", help="persist predictions but do not render figures")
    return parser.parse_args()


def select_episode_frames(dataset, num_episodes: int, stride: int, max_steps: int) -> list[tuple[int, int, int]]:
    if num_episodes < 0:
        raise ValueError("--num-episodes must be >= 0")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if max_steps < 0:
        raise ValueError("--max-steps must be >= 0")

    by_episode: dict[int, list[tuple[int, int]]] = {}
    for index in range(len(dataset)):
        _, episode, step = dataset.sample_step(index)
        by_episode.setdefault(int(episode), []).append((int(step), index))

    selected: list[tuple[int, int, int]] = []
    episodes = sorted(by_episode)[: num_episodes or None]
    for episode in episodes:
        frames = sorted(by_episode[episode])[::stride]
        if max_steps:
            frames = frames[:max_steps]
        selected.extend((episode, step, index) for step, index in frames)
    return selected


def _stat_value(entry: object, name: str) -> float:
    value = entry[name] if isinstance(entry, dict) else getattr(entry, name)
    return float(np.ravel(value)[0])


def dataset_action_stats(dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    stats = dataset.datasets[0].metadata.statistics.action
    keys = list(stats.keys())
    lo = np.array([_stat_value(stats[key], "min") for key in keys], dtype=np.float64)
    hi = np.array([_stat_value(stats[key], "max") for key in keys], dtype=np.float64)
    if len(keys) != EXPECTED_ACTION_DIM:
        raise ValueError(f"Piper open-loop expects 7 action dimensions, got {keys}")
    return lo, hi, keys


def denormalize(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (np.asarray(actions, dtype=np.float64) + 1.0) / 2.0 * (hi - lo) + lo


def normalize(actions: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return 2.0 * (np.asarray(actions, dtype=np.float64) - lo) / np.maximum(hi - lo, 1e-8) - 1.0


def validate_action(action: object, expected_chunk_len: int) -> np.ndarray:
    if not isinstance(action, np.ndarray):
        raise TypeError(f"response['action'] must be np.ndarray, got {type(action)}")
    if action.shape != (expected_chunk_len, EXPECTED_ACTION_DIM):
        raise ValueError(
            f"response['action'] must have shape ({expected_chunk_len}, 7), got {action.shape}"
        )
    if action.dtype != np.float32:
        raise TypeError(f"response['action'] must be float32, got {action.dtype}")
    if not np.isfinite(action).all():
        raise ValueError("response['action'] contains NaN or Inf")
    return action.astype(np.float64)


def compute_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray, pred_raw: np.ndarray, gt_raw: np.ndarray) -> dict:
    norm_error = pred_norm - gt_norm
    raw_error = pred_raw - gt_raw
    return {
        "normalized": {
            "mae": float(np.abs(norm_error).mean()),
            "mse": float((norm_error**2).mean()),
            "mae_per_dim": np.abs(norm_error).mean(axis=(0, 1)).tolist(),
            "step0_mae": float(np.abs(norm_error[:, 0]).mean()),
            "step0_mae_per_dim": np.abs(norm_error[:, 0]).mean(axis=0).tolist(),
        },
        "raw": {
            "mae": float(np.abs(raw_error).mean()),
            "mae_per_dim": np.abs(raw_error).mean(axis=(0, 1)).tolist(),
            "step0_mae": float(np.abs(raw_error[:, 0]).mean()),
            "step0_mae_per_dim": np.abs(raw_error[:, 0]).mean(axis=0).tolist(),
        },
    }


def render_plots(output_dir: Path, prediction_path: Path, episodes: list[int]) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "plot_openloop_trajectory.py"),
        "--predictions",
        str(prediction_path),
        "--out_dir",
        str(output_dir),
        "--episodes",
        *(str(episode) for episode in episodes),
        "--zoom_episode",
        str(episodes[0]),
    ]
    subprocess.run(command, check=True)
    figure_names = [
        "trajectory_fit.png",
        "trajectory_zoom.png",
        "trajectory_2d.png",
        "error_over_time.png",
        "gripper_timeline.png",
    ]
    missing = [name for name in figure_names if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"plot renderer did not create figures: {missing}")
    return figure_names


def main() -> int:
    args = parse_args()
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be > 0")
    if args.image_size <= 0:
        raise ValueError("--image-size must be > 0")

    cfg = OmegaConf.load(args.config_yaml)
    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        mode="val",
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        seed=args.seed,
    )
    frames = select_episode_frames(dataset, args.num_episodes, args.stride, args.max_steps)
    if not frames:
        raise RuntimeError("no dataset frames selected")

    lo, hi, action_keys = dataset_action_stats(dataset)
    expected_chunk_len = int(cfg.framework.action_model.future_action_window_size) + 1
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_raw: list[np.ndarray] = []
    targets_raw: list[np.ndarray] = []
    latencies: list[float] = []
    rows: list[dict] = []
    meta: list[tuple[int, int]] = []

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(f"tcp://{args.host}:{args.port}")

    try:
        for position, (episode, step, index) in enumerate(frames, start=1):
            item = dataset[index]
            observation = item_to_observation(item, args.image_size)
            wire_bytes = pack_payload(observation, jpeg_quality=args.jpeg_quality)
            target_norm = np.asarray(item["action"], dtype=np.float64)
            if target_norm.shape != (expected_chunk_len, EXPECTED_ACTION_DIM):
                raise ValueError(
                    f"dataset action at ep={episode} step={step} has {target_norm.shape}, "
                    f"expected {(expected_chunk_len, EXPECTED_ACTION_DIM)}"
                )
            target_raw = denormalize(target_norm, lo, hi)

            started = time.perf_counter()
            socket.send(wire_bytes)
            try:
                response = unpack_payload(socket.recv())
            except zmq.error.Again as exc:
                raise TimeoutError(
                    f"server response timeout after {args.timeout_ms} ms at episode={episode} step={step}"
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0

            if not isinstance(response, dict):
                raise TypeError(f"server response must be dict, got {type(response)}")
            action_info = describe_action(response.get("action"))
            prediction_raw = validate_action(response["action"], expected_chunk_len)
            predictions_raw.append(prediction_raw)
            targets_raw.append(target_raw)
            latencies.append(latency_ms)
            meta.append((episode, step))
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "dataset_index": index,
                    "instruction": observation["cmd"],
                    "wire_bytes": len(wire_bytes),
                    "latency_ms": latency_ms,
                    "action": action_info,
                }
            )
            print(
                f"openloop {position}/{len(frames)} ep={episode} step={step} "
                f"latency={latency_ms:.0f}ms action={action_info['shape']}"
            )
    finally:
        socket.close()
        context.term()

    pred_raw = np.stack(predictions_raw)
    gt_raw = np.stack(targets_raw)
    pred_norm = normalize(pred_raw, lo, hi)
    gt_norm = normalize(gt_raw, lo, hi)
    metrics = compute_metrics(pred_norm, gt_norm, pred_raw, gt_raw)
    episodes = sorted({episode for episode, _ in meta})
    prediction_path = output_dir / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        pred=pred_norm,
        gt=gt_norm,
        meta=np.asarray(meta, dtype=np.int64),
        action_min=lo,
        action_max=hi,
    )

    figures = [] if args.skip_plot else render_plots(output_dir, prediction_path, episodes)
    plot_manifest = {
        "status": "complete" if figures else "not_requested",
        "kind": "trajectory-comparison",
        "source_record": str(output_dir / "openloop_result.json"),
        "plot_input": str(prediction_path),
        "normalization": "physical action values reconstructed from Piper min_max [-1, 1]",
        "dimensions": DIM_NAMES,
        "figures": [str(output_dir / name) for name in figures],
    }
    with open(output_dir / "plot_manifest.json", "w") as file:
        json.dump(plot_manifest, file, indent=2)

    result = {
        "status": "passed",
        "server": {
            "host": args.host,
            "port": args.port,
            "gripper_mode": args.server_gripper_mode,
        },
        "config_yaml": args.config_yaml,
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "wire": {"format": "msgpack_numpy", "jpeg_quality": args.jpeg_quality, "image_size": args.image_size},
        "alignment": {
            "observation": "dataset frame t",
            "prediction": "server action chunk starting at t",
            "step0_target": "dataset action chunk starting at t",
            "tail_policy": "dataset-provided padded/aligned action horizon",
        },
        "action_keys": action_keys,
        "num_frames": len(rows),
        "num_episodes": len(episodes),
        "episodes": episodes,
        "chunk_len": expected_chunk_len,
        "action_dim": EXPECTED_ACTION_DIM,
        "metrics": metrics,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p95": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        },
        "prediction_path": str(prediction_path),
        "plot_manifest": str(output_dir / "plot_manifest.json"),
        "windows": rows,
        "predictions_normalized": pred_norm.tolist(),
        "ground_truth_normalized": gt_norm.tolist(),
    }
    with open(output_dir / "openloop_result.json", "w") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({"frames": len(rows), "metrics": metrics, "latency_ms": result["latency_ms"]}, indent=2))
    print(f"written -> {output_dir / 'openloop_result.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
