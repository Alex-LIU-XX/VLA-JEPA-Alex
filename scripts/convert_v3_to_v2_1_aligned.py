"""
Convert a LeRobot v3.0 dataset (multi-episode parquet + mp4) to v2.1 format
so that it can be consumed by VLA-JEPA (gr00t lerobot dataloader).

Compared with ``convert_v3_to_v2_1.py`` this version:
  * decodes every source video in ONE sequential pass and slices frames by
    frame index (``from_timestamp * fps`` + ``length``) instead of matching
    floats against packet PTS  ->  more robust and far faster;
  * keeps only the columns the trainer needs (drops e.g. observation.tracks_*);
  * writes the real task texts, a VLA-JEPA compatible ``modality.json``
    (x/y/z/roll/pitch/yaw/gripper + video + annotation) and an ``info.json``
    carrying the ``info`` block required by the dataloader.

Usage:
    python scripts/convert_v3_to_v2_1_aligned.py \
        --input  /path/to/v3_dataset \
        --output /path/to/output_v2_1 \
        --fps 10 \
        --chunk-size 100
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm

# keys expected by starVLA/dataloader/gr00t_lerobot/data_config.py::PiperDataConfig
POSE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
# only the low-dim columns used for training are copied over
KEEP_COLUMNS = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LeRobot v3.0 dataset to v2.1 format")
    parser.add_argument("--input", type=str, required=True, help="Path to v3.0 dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to output v2.1 dataset")
    parser.add_argument("--fps", type=int, default=10, help="Video frame rate")
    parser.add_argument("--chunk-size", type=int, default=100, help="Episodes per chunk")
    return parser.parse_args()


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_video(frames: list[np.ndarray], output_path: Path, fps: int):
    """Encode RGB frames to mp4 (h264) with PyAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "veryfast", "crf": "23"}
    for arr in frames:
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def load_tasks(input_path: Path) -> dict:
    """task_index -> task text."""
    task_file = input_path / "meta" / "tasks.parquet"
    if not task_file.exists():
        return {}
    df = pd.read_parquet(task_file).reset_index()
    text_col = next((c for c in df.columns if c != "task_index"), None)
    if text_col is None:
        return {}
    return {int(r["task_index"]): str(r[text_col]) for _, r in df.iterrows()}


def load_episodes(input_path: Path) -> pd.DataFrame:
    ep_files = sorted(input_path.glob("meta/episodes/**/*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode parquet files in {input_path}/meta/episodes/")
    return pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)


def convert_video_file(src_video: Path, ranges: list, out_paths: dict, fps: int) -> dict:
    """Decode ``src_video`` once, slice out every episode range.

    ranges: list of (episode_index, start_frame, end_frame) sorted by start_frame.
    returns {episode_index: frames_written}
    """
    written = {}
    buf: list[np.ndarray] = []
    idx = 0
    ri = 0
    container = av.open(str(src_video))
    for frame in container.decode(video=0):
        while ri < len(ranges) and idx >= ranges[ri][2]:
            ep = ranges[ri][0]
            if buf:
                write_video(buf, out_paths[ep], fps)
            written[ep] = len(buf)
            buf, ri = [], ri + 1
        if ri < len(ranges) and ranges[ri][1] <= idx < ranges[ri][2]:
            buf.append(frame.to_ndarray(format="rgb24"))
        idx += 1
    container.close()
    while ri < len(ranges):
        ep = ranges[ri][0]
        if buf:
            write_video(buf, out_paths[ep], fps)
        written[ep] = len(buf)
        buf, ri = [], ri + 1
    return written


def main():
    args = parse_args()
    input_path, output_path = Path(args.input), Path(args.output)
    fps, chunk_size = args.fps, args.chunk_size

    print(f"Input  (v3.0): {input_path}")
    print(f"Output (v2.1): {output_path}")

    info = read_json(input_path / "meta" / "info.json")
    features = info["features"]
    episodes_df = load_episodes(input_path)
    tasks = load_tasks(input_path)

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    state_dim = int(features["observation.state"]["shape"][0])
    action_dim = int(features["action"]["shape"][0])

    print(f"  episodes={len(episodes_df)} frames={int(episodes_df['length'].sum())} "
          f"tasks={len(tasks)}")
    print(f"  video_keys={video_keys} state_dim={state_dim} action_dim={action_dim}")

    # ---------------------------------------------------------------- skeleton
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "meta").mkdir(exist_ok=True)
    for vk in video_keys:
        (output_path / "videos" / vk).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ low-dim data
    print("\n[1/4] Writing per-episode parquet ...")
    # dataset_from_index/to_index are *global* row indices while each source
    # parquet holds only a contiguous slice of the dataset -> pre-compute the
    # global index at which every source file starts.
    file_base: dict = {}
    for _, ep in episodes_df.iterrows():
        key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        base = int(ep["dataset_from_index"])
        file_base[key] = min(file_base.get(key, base), base)

    parquet_cache: dict = {}
    kept_rows = 0
    for _, ep in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="parquet"):
        ep_index = int(ep["episode_index"])
        data_key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        if data_key not in parquet_cache:
            src = input_path / f"data/chunk-{data_key[0]:03d}/file-{data_key[1]:03d}.parquet"
            parquet_cache.clear()  # keep memory bounded: one source file at a time
            parquet_cache[data_key] = pd.read_parquet(src)
        df = parquet_cache[data_key]
        local_from = int(ep["dataset_from_index"]) - file_base[data_key]
        local_to = int(ep["dataset_to_index"]) - file_base[data_key]
        ep_df = df.iloc[local_from:local_to].copy()
        ep_df = ep_df[[c for c in KEEP_COLUMNS if c in ep_df.columns]].reset_index(drop=True)
        if len(ep_df) != int(ep["length"]):
            print(f"  !! episode {ep_index}: parquet rows {len(ep_df)} != length {ep['length']}")
        out_chunk = ep_index // chunk_size
        out_dir = output_path / "data" / f"chunk_{out_chunk:05d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_df.to_parquet(out_dir / f"episode_{ep_index:06d}.parquet", index=False)
        kept_rows += len(ep_df)
    print(f"  wrote {len(episodes_df)} parquet files, {kept_rows} rows")

    # ---------------------------------------------------------------- videos
    print("\n[2/4] Extracting per-episode videos ...")
    mismatches = []
    for vk in video_keys:
        groups: dict = {}
        for _, ep in episodes_df.iterrows():
            key = (int(ep[f"videos/{vk}/chunk_index"]), int(ep[f"videos/{vk}/file_index"]))
            groups.setdefault(key, []).append(ep)
        for (chunk, file), eps in tqdm(groups.items(), desc=vk):
            src_video = input_path / f"videos/{vk}/chunk-{chunk:03d}/file-{file:03d}.mp4"
            if not src_video.exists():
                print(f"  !! missing source video {src_video}")
                continue
            ranges, out_paths = [], {}
            for ep in eps:
                ep_index = int(ep["episode_index"])
                start = int(round(float(ep[f"videos/{vk}/from_timestamp"]) * fps))
                length = int(ep["length"])
                ranges.append((ep_index, start, start + length))
                out_paths[ep_index] = (
                    output_path / "videos" / vk
                    / f"chunk_{ep_index // chunk_size:05d}" / f"episode_{ep_index:06d}.mp4"
                )
            ranges.sort(key=lambda r: r[1])
            written = convert_video_file(src_video, ranges, out_paths, fps)
            for ep_index, start, end in ranges:
                if written.get(ep_index, 0) != end - start:
                    mismatches.append((vk, ep_index, end - start, written.get(ep_index, 0)))
    if mismatches:
        print(f"  !! {len(mismatches)} episode video(s) with frame-count mismatch:")
        for m in mismatches[:10]:
            print(f"     key={m[0]} ep={m[1]} expected={m[2]} got={m[3]}")
    else:
        print("  all episode videos match their expected frame counts")

    # ------------------------------------------------------------ metadata
    print("\n[3/4] Writing meta files ...")
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for _, ep in episodes_df.iterrows():
            f.write(json.dumps({
                "episode_index": int(ep["episode_index"]),
                "length": int(ep["length"]),
                "task_index": int(ep["task_index"]) if "task_index" in ep else 0,
            }) + "\n")

    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        if tasks:
            for idx in sorted(tasks):
                f.write(json.dumps({"task_index": int(idx), "task": tasks[idx]}) + "\n")
        else:
            for idx in sorted({int(t) for t in episodes_df.get("task_index", [0])}):
                f.write(json.dumps({"task_index": int(idx), "task": str(idx)}) + "\n")

    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for i in range(state_dim):
        modality["state"][POSE_KEYS[i]] = {"start": i, "end": i + 1}
    for i in range(action_dim):
        modality["action"][POSE_KEYS[i]] = {"start": i, "end": i + 1}
    for vk in video_keys:
        modality["video"][vk] = {"original_key": vk}
    modality["annotation"]["human.action.task_description"] = {"original_key": "task_index"}
    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    v2_features = {}
    for k, v in features.items():
        if k.startswith("observation.tracks"):
            continue  # tracking features are not used by the trainer
        v2_features[k] = v
    for vk in video_keys:
        v2_features[vk]["names"] = ["channels", "height", "width"]
        v2_features[vk].setdefault("info", {})
        v2_features[vk]["info"].update({
            "video.height": int(v2_features[vk]["shape"][1]),
            "video.width": int(v2_features[vk]["shape"][2]),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        })
    v2_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type") or "piper",
        "total_episodes": int(len(episodes_df)),
        "total_frames": int(episodes_df["length"].sum()),
        "total_tasks": len(tasks) if tasks else int(episodes_df["task_index"].nunique()),
        "chunks_size": chunk_size,
        "fps": fps,
        "data_path": "data/chunk_{episode_chunk:05d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk_{episode_chunk:05d}/episode_{episode_index:06d}.mp4",
        "features": v2_features,
    }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(v2_info, f, indent=2)

    # sanity: the dataloader caches sampling steps in a fixed-name pickle that
    # would silently poison the dataset if it leaked over from another dataset
    stale = list((output_path / "meta").glob("steps_*.pkl"))
    for p in stale:
        p.unlink()
        print(f"  removed stale step cache {p.name}")

    print("\n[4/4] Done.")
    print(f"  episodes : {len(episodes_df)}")
    print(f"  frames   : {int(episodes_df['length'].sum())}")
    print(f"  tasks    : {len(tasks)}")
    print(f"  output   : {output_path}")


if __name__ == "__main__":
    main()
