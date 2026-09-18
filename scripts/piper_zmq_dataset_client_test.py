"""Dataset-backed ZMQ client test for the Piper VLA-JEPA server.

The server remains a separate process. This script only owns the client side:
it loads LeRobot samples, applies the same direct 224x224 resize used by
``examples/real-robot/selftest_client.py``, packs observations with the real
Piper wire format, sends them over ZMQ REQ/REP, and validates the returned
action chunk.

No robot connection or actuator command is involved.
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
import zmq
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.lerobot_datasets import get_vla_dataset

REAL_ROBOT_DIR = Path(__file__).resolve().parents[1] / "examples" / "real-robot"
sys.path.insert(0, str(REAL_ROBOT_DIR))

from wire import pack_payload, unpack_payload  # noqa: E402


EXPECTED_ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-backed Piper ZMQ client test")
    parser.add_argument("--config_yaml", required=True, help="training config used to construct the dataset")
    parser.add_argument("--output_dir", required=True, help="directory for test_result.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=1, help="0 = all episodes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_windows(dataset, windows_per_episode: int, num_episodes: int) -> list[tuple[int, int, int]]:
    """Select deterministic dataset windows without changing the sample values."""
    if windows_per_episode < 1:
        raise ValueError("--windows-per-episode must be >= 1")
    if num_episodes < 0:
        raise ValueError("--num-episodes must be >= 0")

    by_episode: dict[int, list[tuple[int, int]]] = {}
    for index in range(len(dataset)):
        _, episode, step = dataset.sample_step(index)
        by_episode.setdefault(int(episode), []).append((int(step), index))

    selected: list[tuple[int, int, int]] = []
    for episode in sorted(by_episode)[: num_episodes or None]:
        items = sorted(by_episode[episode])
        picks = np.linspace(0, len(items) - 1, windows_per_episode).round().astype(int)
        for pick in sorted(set(picks.tolist())):
            step, index = items[pick]
            selected.append((episode, step, index))
    return selected


def resize_like_selftest(value: object, image_size: int) -> np.ndarray:
    """Match selftest_client._resize: RGB conversion followed by direct resize."""
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        image = Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def item_to_observation(item: dict, image_size: int) -> dict:
    images = item["image"]
    if len(images) != 2:
        raise ValueError(f"Piper test expects head+wrist images, got {len(images)} views")

    state = np.asarray(item["state"], dtype=np.float32).reshape(-1)
    instruction = str(item["lang"])
    return {
        "image": resize_like_selftest(images[0], image_size),
        "wrist_image": resize_like_selftest(images[1], image_size),
        "state": state,
        "cmd": instruction,
    }


def describe_action(action: np.ndarray) -> dict:
    if not isinstance(action, np.ndarray):
        raise TypeError(f"response['action'] must be np.ndarray, got {type(action)}")
    if action.ndim != 2 or action.shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError(f"response['action'] must have shape (T, 7), got {action.shape}")
    if action.dtype != np.float32:
        raise TypeError(f"response['action'] must be float32, got {action.dtype}")
    if not np.isfinite(action).all():
        raise ValueError("response['action'] contains NaN or Inf")
    return {
        "shape": list(action.shape),
        "dtype": str(action.dtype),
        "first": action[0].tolist(),
        "last": action[-1].tolist(),
    }


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
    windows = select_windows(dataset, args.windows_per_episode, args.num_episodes)
    if not windows:
        raise RuntimeError("no dataset windows selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(f"tcp://{args.host}:{args.port}")

    try:
        for position, (episode, step, index) in enumerate(windows, start=1):
            item = dataset[index]
            observation = item_to_observation(item, args.image_size)
            wire_bytes = pack_payload(observation, jpeg_quality=args.jpeg_quality)

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
            row = {
                "episode": episode,
                "step": step,
                "dataset_index": index,
                "instruction": observation["cmd"],
                "observation": {
                    "image": {"shape": list(observation["image"].shape), "dtype": str(observation["image"].dtype)},
                    "wrist_image": {
                        "shape": list(observation["wrist_image"].shape),
                        "dtype": str(observation["wrist_image"].dtype),
                    },
                    "state": {"shape": list(observation["state"].shape), "dtype": str(observation["state"].dtype)},
                    "wire_bytes": len(wire_bytes),
                },
                "response_keys": sorted(response),
                "action": action_info,
                "latency_ms": latency_ms,
            }
            result_rows.append(row)
            print(
                f"test {position}/{len(windows)} ep={episode} step={step} "
                f"latency={latency_ms:.0f}ms action={action_info['shape']} {action_info['dtype']}"
            )
    finally:
        socket.close()
        context.term()

    result = {
        "status": "passed",
        "server": {"host": args.host, "port": args.port},
        "config_yaml": args.config_yaml,
        "data_root_dir": str(cfg.datasets.vla_data.data_root_dir),
        "wire": {
            "format": "msgpack_numpy",
            "jpeg_quality": args.jpeg_quality,
            "image_size": args.image_size,
        },
        "num_windows": len(result_rows),
        "num_episodes": len({row["episode"] for row in result_rows}),
        "latency_ms": {
            "mean": float(np.mean([row["latency_ms"] for row in result_rows])),
            "p95": float(np.percentile([row["latency_ms"] for row in result_rows], 95)),
            "max": float(np.max([row["latency_ms"] for row in result_rows])),
        },
        "windows": result_rows,
    }
    with open(output_dir / "test_result.json", "w") as file:
        json.dump(result, file, indent=2)
    print(f"passed: {len(result_rows)} requests")
    print(f"written -> {output_dir / 'test_result.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
