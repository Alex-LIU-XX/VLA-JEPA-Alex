"""
Convert LeRobot v3.0 dataset to v2.1 format for VLA-JEPA training.

Usage:
    python scripts/convert_v3_to_v2_1.py \
        --input /path/to/v3_dataset \
        --output /path/to/output_v2_1 \
        --fps 10
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3.0 dataset to v2.1 format")
    parser.add_argument("--input", type=str, required=True, help="Path to v3.0 dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output v2.1 dataset")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--chunk-size", type=int, default=100, help="Episodes per chunk")
    return parser.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_v3_episodes(input_path: Path) -> pd.DataFrame:
    ep_files = sorted(input_path.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files in {input_path}/meta/episodes/")
    return pd.read_parquet(ep_files[0])


def read_v3_tasks(input_path: Path) -> pd.DataFrame:
    task_file = input_path / "meta" / "tasks.parquet"
    if not task_file.exists():
        return pd.DataFrame({"task": ["task_0"], "task_index": [0]})
    df = pd.read_parquet(task_file).reset_index()
    if "task_index" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "task_index"})
    return df


def extract_frames(video_path: Path, timestamps: list[float]) -> list[np.ndarray]:
    """Extract frames at given timestamps from an AV1 video using PyAV."""
    frames = []
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate)

        # Use nearest keyframe + forward decode approach
        ts_set = set(round(t, 6) for t in timestamps)
        extracted = set()

        for packet in container.demux(stream):
            for frame in packet.decode():
                pts_sec = float(frame.pts * stream.time_base)
                pts_rounded = round(pts_sec, 6)
                if pts_rounded in ts_set and pts_rounded not in extracted:
                    img = frame.to_ndarray(format="rgb24")
                    frames.append((pts_rounded, img))
                    extracted.add(pts_rounded)
                    if len(extracted) == len(ts_set):
                        break
            if len(extracted) == len(ts_set):
                break

        container.close()

        # Sort by timestamp and return frames in original order
        result = []
        for t in timestamps:
            t_rounded = round(t, 6)
            found = False
            for ts, img in frames:
                if ts == t_rounded:
                    result.append(img)
                    found = True
                    break
            if not found:
                # Fallback: extract at exact timestamp
                container2 = av.open(str(video_path))
                stream2 = container2.streams.video[0]
                container2.seek(int(t / stream2.time_base), stream=stream2)
                for packet2 in container2.demux(stream2):
                    for frame2 in packet2.decode():
                        img = frame2.to_ndarray(format="rgb24")
                        result.append(img)
                        break
                    break
                container2.close()
        return result

    except Exception as e:
        print(f"  Warning: video extraction failed for {video_path}: {e}")
        height, width = 480, 640
        return [np.zeros((height, width, 3), dtype=np.uint8) for _ in timestamps]


def write_video(frames: list[np.ndarray], output_path: Path, fps: int):
    """Write frames to mp4 using PyAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "fast", "crf": "23"}

    for frame_data in frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    fps = args.fps
    chunk_size = args.chunk_size

    print(f"Input (v3.0):  {input_path}")
    print(f"Output (v2.1): {output_path}")

    # Read v3.0 metadata
    print("\n[1/5] Reading v3.0 metadata...")
    info = read_json(input_path / "meta" / "info.json")
    episodes_df = read_v3_episodes(input_path)
    tasks_df = read_v3_tasks(input_path)

    print(f"  Episodes: {len(episodes_df)}")
    print(f"  Tasks: {len(tasks_df)}")
    print(f"  Frames: {info.get('total_frames', '?')}")
    print(f"  Robot type: {info.get('robot_type', '?')}")

    # Determine video/state/action keys
    video_keys = []
    state_names = []
    action_names = []
    for feat_name, feat_info in info.get("features", {}).items():
        if feat_info.get("dtype") == "video":
            video_keys.append(feat_name)
        elif feat_name == "observation.state":
            state_names = feat_info.get("names", [f"dim_{i}" for i in range(feat_info["shape"][0])])
        elif feat_name == "action":
            action_names = feat_info.get("names", [f"dim_{i}" for i in range(feat_info["shape"][0])])

    print(f"  Video keys: {video_keys}")
    print(f"  State dim: {len(state_names)}")
    print(f"  Action dim: {len(action_names)}")

    # Create output directories
    print("\n[2/5] Creating output directory structure...")
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    for vk in video_keys:
        (output_path / "videos" / vk).mkdir(parents=True, exist_ok=True)

    # Create modality.json
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for i, k in enumerate(state_names):
        modality["state"][k] = {"start": i, "end": i + 1}
    for i, k in enumerate(action_names):
        modality["action"][k] = {"start": i, "end": i + 1}
    for vk in video_keys:
        modality["video"][vk] = {"original_key": vk}
    modality["annotation"]["human.action.task_description"] = {"original_key": "task_index"}
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)
    print(f"  Created modality.json")

    # Write tasks.jsonl
    print("\n[3/5] Writing tasks...")
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        for idx, row in tasks_df.iterrows():
            task_text = str(row.get("task", row.get("task_index", f"task_{idx}")))
            entry = {"task_index": int(idx), "task": task_text}
            f.write(json.dumps(entry) + "\n")
    print(f"  Done ({len(tasks_df)} tasks)")

    # Write info.json
    print("\n[4/5] Writing info.json...")
    v2_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", "unknown"),
        "total_episodes": int(episodes_df["episode_index"].nunique()),
        "total_frames": int(episodes_df["length"].sum()),
        "total_tasks": len(tasks_df),
        "chunks_size": chunk_size,
        "fps": fps,
        "data_path": "data/chunk_{episode_chunk:05d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk_{episode_chunk:05d}/episode_{episode_index:06d}.mp4",
        "features": info["features"],
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(v2_info, f, indent=2)
    print(f"  Done")

    # Convert episodes
    print(f"\n[5/5] Converting {len(episodes_df)} episodes...")
    episodes_list = []
    for ep_idx, ep_row in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="Converting"):
        ep_index = int(ep_row["episode_index"])
        ep_length = int(ep_row["length"])
        data_chunk = int(ep_row["data/chunk_index"])
        data_file = int(ep_row["data/file_index"])
        data_from = int(ep_row["dataset_from_index"])
        data_to = int(ep_row["dataset_to_index"])

        # Read parquet slice for this episode
        v3_parquet = input_path / f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet"
        if not v3_parquet.exists():
            continue
        df = pd.read_parquet(v3_parquet)
        ep_data = df.iloc[data_from:data_to].copy().reset_index(drop=True)

        # Write episode parquet
        out_chunk = ep_index // chunk_size
        out_data_dir = output_path / "data" / f"chunk_{out_chunk:05d}"
        out_data_dir.mkdir(parents=True, exist_ok=True)
        ep_data.to_parquet(out_data_dir / f"episode_{ep_index:06d}.parquet", index=False)

        # Extract video frames per camera
        timestamps = ep_data["timestamp"].values
        for vk in video_keys:
            try:
                v3_video_chunk = int(ep_row[f"videos/{vk}/chunk_index"])
                v3_video_file = int(ep_row[f"videos/{vk}/file_index"])
            except (KeyError, ValueError):
                continue

            v3_video = input_path / f"videos/{vk}/chunk-{v3_video_chunk:03d}/file-{v3_video_file:03d}.mp4"
            if not v3_video.exists():
                continue

            frames = extract_frames(v3_video, list(timestamps))
            if frames:
                out_video = output_path / "videos" / vk / f"chunk_{out_chunk:05d}" / f"episode_{ep_index:06d}.mp4"
                write_video(frames, out_video, fps)

        episodes_list.append({"episode_index": ep_index, "length": ep_length})

    # Write episodes.jsonl
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_list:
            f.write(json.dumps(ep) + "\n")

    print(f"\nDone! Dataset converted to v2.1 format at: {output_path}")
    print(f"  Episodes: {len(episodes_list)}")
    print(f"  Action dim: {len(action_names)}")
    print(f"  State dim: {len(state_names)}")
    print(f"  Video keys: {video_keys}")


if __name__ == "__main__":
    main()
