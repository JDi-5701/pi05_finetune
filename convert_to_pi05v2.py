#!/usr/bin/env python3
"""Convert a Franka Cartesian-EE LeRobot dataset into a pi0.5-ingestible v3 dataset.

This is a small variant of convert_to_pi05.py.  The default written dataset keeps the
same action convention as the original converter:

  action[t] = target_action[t] relative to observation[t]

The v2 difference is only the action normalization stats.  When --action-mode
delta is used, meta/stats.json["action"] is recomputed over openpi-style
50-step labels:

  action_chunk[k] = target_action[t + k] relative to observation[t]

This matches a training loop that dynamically rewrites each sampled action chunk
to use the sample's first observation as the delta base.

Usage:
    python convert_to_pi05v2.py --root /path/<src> --out /path/<dst> --rot aa --action-mode delta
"""
import argparse
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset

GRIPPER_MAX_WIDTH = 0.04

ROT_NAMES = {
    "6d": ["r00", "r10", "r20", "r01", "r11", "r21"],
    "aa": ["rx", "ry", "rz"],
    "quat": ["qx", "qy", "qz", "qw"],
}


def encode_rot(R, rep):
    """scipy Rotation -> chosen representation as a float32 vector."""
    if rep == "quat":
        return R.as_quat().astype(np.float32)
    if rep == "aa":
        return R.as_rotvec().astype(np.float32)
    m = R.as_matrix()
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)


def to_np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def make_state(state, rep):
    """[x,y,z, qx,qy,qz,qw, wrench(6), grip0, grip1] -> [xyz, rot, grip]."""
    s = to_np(state)
    pos = s[0:3].astype(np.float32)
    rot = encode_rot(Rotation.from_quat(s[3:7]), rep)
    grip = np.array([float(np.clip(s[13] / GRIPPER_MAX_WIDTH, 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def make_action(action, reference_state, rep, mode):
    """[x,y,z, qx,qy,qz,qw, grip] absolute action -> [xyz, rot, grip].

    mode='absolute': keep the target pose absolute.
    mode='delta': encode position and rotation relative to reference_state.
    Gripper remains absolute in [0, 1].
    """
    a = to_np(action)
    s_ref = to_np(reference_state)
    R_act = Rotation.from_quat(a[3:7])
    grip = np.array([float(np.clip(a[7], 0.0, 1.0))], dtype=np.float32)
    if mode == "delta":
        pos = (a[0:3] - s_ref[0:3]).astype(np.float32)
        R_rel = Rotation.from_quat(s_ref[3:7]).inv() * R_act
        rot = encode_rot(R_rel, rep)
    else:
        pos = a[0:3].astype(np.float32)
        rot = encode_rot(R_act, rep)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def make_state_delta_action(state, next_state, rep):
    """Actual measured motion from state[t] to state[t+1].

    Position and rotation are encoded relative to state[t].  Gripper is the
    measured normalized next gripper width, keeping the same gripper convention
    as the other action modes.
    """
    s = to_np(state)
    s_next = to_np(next_state)
    pos = (s_next[0:3] - s[0:3]).astype(np.float32)
    R_rel = Rotation.from_quat(s[3:7]).inv() * Rotation.from_quat(s_next[3:7])
    rot = encode_rot(R_rel, rep)
    grip = np.array([float(np.clip(s_next[13] / GRIPPER_MAX_WIDTH, 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def chw_float_to_hwc_uint8(img):
    """LeRobot decodes video to CHW float32 [0,1]; the writer wants HWC uint8."""
    t = img if torch.is_tensor(img) else torch.as_tensor(img)
    t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return t.permute(1, 2, 0).contiguous().numpy()


def scalar_int(value):
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def summarize_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_chunk_start_action_stats(frames, rep, chunk_size):
    """Stats for dynamic openpi-style labels over all overlapping chunks."""
    labels = []
    ep_start = 0
    while ep_start < len(frames):
        ep = frames[ep_start]["episode"]
        ep_end = ep_start
        while ep_end < len(frames) and frames[ep_end]["episode"] == ep:
            ep_end += 1

        for start in range(ep_start, ep_end):
            reference_state = frames[start]["state"]
            last = min(start + chunk_size, ep_end)
            for idx in range(start, last):
                labels.append(make_action(frames[idx]["action"], reference_state, rep, mode="delta"))

        ep_start = ep_end

    return summarize_stats(np.stack(labels, axis=0))


def overwrite_action_stats(out_dir, action_stats):
    stats_path = os.path.join(out_dir, "meta", "stats.json")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    stats["action"] = action_stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def load_task_map(root):
    tasks_path = os.path.join(root, "meta", "tasks.parquet")
    if not os.path.exists(tasks_path):
        return {}
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        return {}
    if "task" in tasks.columns:
        return {int(row.task_index): row.task for row in tasks.itertuples(index=False)}
    if tasks.index.name == "task":
        return {int(row.task_index): str(task) for task, row in tasks.iterrows()}
    return {}


def load_lowdim_frames(root):
    """Load state/action/task metadata without decoding video frames."""
    task_map = load_task_map(root)
    data_root = os.path.join(root, "data")
    parquet_files = []
    for dirpath, _, filenames in os.walk(data_root):
        for name in filenames:
            if name.endswith(".parquet"):
                parquet_files.append(os.path.join(dirpath, name))
    parquet_files.sort()
    if not parquet_files:
        raise SystemExit(f"no data parquet files found under {data_root}")

    frames = []
    columns = ["observation.state", "action", "episode_index", "frame_index", "task_index"]
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path, columns=columns)
        df = df.sort_values(["episode_index", "frame_index"])
        for row in df.itertuples(index=False):
            task_index = int(getattr(row, "task_index"))
            frames.append({
                "episode": int(getattr(row, "episode_index")),
                "state": np.asarray(getattr(row, "_0"), dtype=np.float32).copy(),
                "action": np.asarray(getattr(row, "action"), dtype=np.float32).copy(),
                "task": task_map.get(task_index, ""),
            })
    return frames


def main():
    ap = argparse.ArgumentParser(description="convert Franka EE dataset -> pi0.5 format, openpi-style action stats")
    ap.add_argument("--root", required=True, help="source dataset dir (has meta/info.json)")
    ap.add_argument("--out", required=True, help="destination dir for the converted dataset")
    ap.add_argument("--repo-id", default=None, help="repo_id for the new dataset (default: out name)")
    ap.add_argument("--rot", choices=["6d", "aa", "quat"], default="6d",
                    help="rotation representation (default 6d; 'aa'=axis-angle like LIBERO)")
    ap.add_argument("--action-mode", choices=["absolute", "delta", "state-delta"], default="absolute",
                    help="absolute EE, same-frame target delta, or measured state[t]->state[t+1] delta")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="horizon used to recompute openpi-style action stats (default: 50)")
    ap.add_argument("--cam-key", default="observation.images.base",
                    help="SOURCE camera feature key to read from")
    ap.add_argument("--out-cam-key", default="observation.images.base_0_rgb",
                    help="OUTPUT camera key expected by pi05_base's image feature names")
    args = ap.parse_args()

    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if os.path.exists(args.out):
        raise SystemExit(
            f"destination already exists: {args.out}\n"
            f"lerobot create requires a non-existent dir. Delete it (rm -rf) or pick a new --out.")

    src = LeRobotDataset(os.path.basename(args.root), root=args.root)
    print(f"source: {src.num_episodes} episodes, {src.num_frames} frames @ {src.fps} fps")
    print(f"target: rot={args.rot}  action-mode={args.action_mode}  stats-chunk-size={args.chunk_size}")
    H, W, _ = src.features[args.cam_key]["shape"]
    rot_names = ROT_NAMES[args.rot]
    dim = 3 + len(rot_names) + 1
    names = ["x", "y", "z"] + rot_names + ["gripper"]

    features = {
        args.out_cam_key: {"dtype": "video", "shape": (H, W, 3),
                           "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
        "action": {"dtype": "float32", "shape": (dim,), "names": names},
    }

    dst = LeRobotDataset.create(
        repo_id=args.repo_id or os.path.basename(os.path.normpath(args.out)),
        fps=int(src.fps), root=args.out, robot_type="franka_fr3",
        features=features, use_videos=True)

    # Keep only low-dimensional metadata in memory.  Do NOT cache decoded images:
    # 42 episodes of 720p frames can easily exceed tens of GB and freeze the host.
    raw_frames = load_lowdim_frames(args.root)
    if len(raw_frames) != src.num_frames:
        raise SystemExit(f"metadata frame count mismatch: {len(raw_frames)} vs dataset {src.num_frames}")
    print(f"  loaded {len(raw_frames)} low-dim frame records without decoding video")

    for i, frame in enumerate(raw_frames):
        f = src[i]
        ep = frame["episode"]
        is_last_in_episode = i + 1 >= len(raw_frames) or raw_frames[i + 1]["episode"] != ep
        if args.action_mode == "state-delta":
            next_state = frame["state"] if is_last_in_episode else raw_frames[i + 1]["state"]
            action = make_state_delta_action(frame["state"], next_state, args.rot)
        else:
            action = make_action(frame["action"], frame["state"], args.rot, args.action_mode)

        dst.add_frame({
            args.out_cam_key: chw_float_to_hwc_uint8(f[args.cam_key]),
            "observation.state": make_state(frame["state"], args.rot),
            "action": action,
            "task": frame["task"],
        })
        if is_last_in_episode:
            dst.save_episode()
        if i % 500 == 0:
            print(f"  wrote frame {i}/{src.num_frames} (episode {ep})")

    if hasattr(dst, "finalize"):
        dst.finalize()
    if args.action_mode == "delta":
        print("recomputing action stats with openpi-style chunk-start deltas...")
        action_stats = compute_chunk_start_action_stats(raw_frames, args.rot, args.chunk_size)
        overwrite_action_stats(args.out, action_stats)
        print(f"  action stats overwritten in {os.path.join(args.out, 'meta', 'stats.json')}")
    print(f"done -> {args.out}  ({src.num_episodes} episodes, state/action dim={dim})")


if __name__ == "__main__":
    main()
