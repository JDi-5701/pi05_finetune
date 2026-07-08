#!/usr/bin/env python3
"""Offline inference sanity check for the sponge pi05 aa-delta checkpoint.

This loads a finetuned checkpoint, runs real frames from the converted LeRobot
dataset through the saved pi05 processors, converts the predicted 50-step
axis-angle delta action chunk back to absolute EE poses, and plots it against
the dataset's ground-truth absolute target poses.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_CKPT = "./outputs/pi05_wandb_aa_delta_bs8/final"
DEFAULT_ROOT = "./datasets/sponge_pi05_aa_delta"
DEFAULT_REPO_ID = "sponge_pi05_aa_delta"
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
ABS_POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]


def parse_args():
    ap = argparse.ArgumentParser(description="pi05 aa-delta offline inference check")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint dir saved by train_pi05.py")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="converted LeRobot dataset root")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot repo_id label")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--frames", type=int, default=5, help="number of frames to sample")
    ap.add_argument("--start", type=int, default=0, help="first frame index to sample from")
    ap.add_argument("--end", type=int, default=None, help="last frame index to sample from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random", action="store_true", help="sample random frames instead of evenly spaced")
    ap.add_argument("--plot-out", default="./outputs/aa_delta_absolute_pose_compare.png",
                    help="where to save the absolute-pose comparison plot")
    ap.add_argument("--episode-index", type=int, default=None,
                    help="if set, evaluate one full episode instead of sampled frames")
    ap.add_argument("--chunk-stride", type=int, default=None,
                    help="stride between chunk starts in full-episode mode; default = predicted chunk length")
    ap.add_argument("--take-steps", type=int, default=None,
                    help="how many predicted steps to keep from each chunk; default = chunk stride")
    ap.add_argument("--rollout-from-initial-state", action="store_true",
                    help="deprecated/ignored: inference now always rolls out from each chunk start")
    ap.add_argument("--rollout-from-chunk-start", action="store_true",
                    help="kept for command compatibility; this is now the only prediction absolute mode")
    return ap.parse_args()


def resolve_default_path(path, alternates):
    p = Path(path).expanduser()
    if p.exists():
        return p
    for alt in alternates:
        alt_p = Path(alt).expanduser()
        if alt_p.exists():
            print(f"  default path not found, using existing alternate: {alt_p}")
            return alt_p
    return p


def latest_step_dir(root):
    root = Path(root)
    if not root.exists():
        return None
    step_dirs = []
    for child in root.iterdir():
        m = re.fullmatch(r"step_(\d+)", child.name)
        if child.is_dir() and m:
            step_dirs.append((int(m.group(1)), child))
    if not step_dirs:
        return None
    return max(step_dirs, key=lambda item: item[0])[1]


def resolve_ckpt(path):
    p = Path(path).expanduser()
    if p.exists():
        return p
    if path == DEFAULT_CKPT:
        for candidate in (
            "./outputs/pi05_wandb_aa_delta_bs8/step_2000",
            "./outputs/pi05_wandb_aa_delta_bs8/final",
            "./outputs/pi05_overfit_aa_delta/final",
        ):
            candidate_p = Path(candidate).expanduser()
            if candidate_p.exists():
                print(f"  default checkpoint not found, using existing alternate: {candidate_p}")
                return candidate_p
        for root in ("./outputs/pi05_wandb_aa_delta_bs8", "./outputs/pi05_gpu_aa_delta", "./outputs/pi05_overfit_aa_delta"):
            latest = latest_step_dir(root)
            if latest is not None:
                print(f"  default final checkpoint not found, using latest step: {latest}")
                return latest
    return p


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def as_action_tensor(x):
    if isinstance(x, dict):
        x = x[ACTION]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return x.detach().float().cpu().squeeze()


def postprocess_action(postprocess, action):
    """Unnormalize model output when possible; tolerate older/incomplete checkpoints."""
    try:
        return as_action_tensor(postprocess(action))
    except Exception as exc:
        print(f"  postprocess skipped: {type(exc).__name__}: {str(exc)[:120]}")
        return as_action_tensor(action)


def select_frame_indices(n_frames, start, end, count, random, seed):
    end = n_frames - 1 if end is None else min(end, n_frames - 1)
    start = max(0, min(start, end))
    if count <= 1:
        return [start]
    if random:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(np.arange(start, end + 1), size=min(count, end - start + 1), replace=False))
    return np.linspace(start, end, count).round().astype(int).tolist()


def print_action(label, value, dim, names=None):
    value = value[:dim].numpy()
    if names is None:
        names = ACTION_NAMES
    shown_names = names[:dim] if dim <= len(names) else [f"a{i}" for i in range(dim)]
    pairs = "  ".join(f"{name}={val:+.4f}" for name, val in zip(shown_names, value, strict=False))
    print(f"  {label:<10} {pairs}")


def as_numpy_vector(x):
    if torch.is_tensor(x):
        x = x.detach().float().cpu().squeeze().numpy()
    else:
        x = np.asarray(x, dtype=np.float32).squeeze()
    return x.astype(np.float32)


def aa_delta_to_abs_pose(state, delta):
    """Invert convert_to_pi05.py's aa delta convention for one timestep.

    Dataset convention:
      state  = [x, y, z, abs_rotvec(3), gripper]
      action = [dx, dy, dz, rel_rotvec(3), gripper]
      R_delta = inv(R_state) * R_action

    Therefore:
      R_action = R_state * R_delta
    """
    state = as_numpy_vector(state)
    delta = as_numpy_vector(delta)
    pos = state[:3] + delta[:3]
    rot = (Rotation.from_rotvec(state[3:6]) * Rotation.from_rotvec(delta[3:6])).as_rotvec()
    grip = np.array([delta[6]], dtype=np.float32)
    return np.concatenate([pos, rot.astype(np.float32), grip]).astype(np.float32)


def chunk_start_deltas_to_abs_poses(chunk_start_state, deltas):
    """Invert train_pi05.py's chunk-start delta target for a predicted chunk.

    Every predicted delta in the chunk is relative to the same observation.state
    at the chunk start; do not recursively add deltas to previous predictions.
    """
    state0 = as_numpy_vector(chunk_start_state).copy()
    poses = [
        aa_delta_to_abs_pose(state0, delta)
        for delta in np.asarray(deltas, dtype=np.float32)
    ]
    return np.stack(poses, axis=0)


def get_episode_index(sample):
    value = sample.get("episode_index")
    if value is None:
        return None
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def collect_abs_gt_chunk(ds, start_idx, horizon):
    """Collect GT absolute target poses for a contiguous in-episode horizon."""
    first_ep = get_episode_index(ds[int(start_idx)])
    poses = []
    for offset in range(horizon):
        sample = ds[int(start_idx + offset)]
        if first_ep is not None and get_episode_index(sample) != first_ep:
            break
        poses.append(aa_delta_to_abs_pose(sample[OBS_STATE], sample[ACTION]))
    return np.stack(poses, axis=0)


def find_episode_frame_indices(ds, episode_index):
    indices = []
    n = getattr(ds, "num_frames", len(ds))
    for idx in range(n):
        sample = ds[idx]
        if get_episode_index(sample) == episode_index:
            indices.append(idx)
    if not indices:
        raise SystemExit(f"episode_index={episode_index} not found in dataset")
    return indices


def evaluate_episode(
    policy,
    pre,
    post,
    ds,
    episode_index,
    chunk_stride,
    take_steps,
):
    indices = find_episode_frame_indices(ds, episode_index)
    start_frame = indices[0]
    end_frame = indices[-1]
    episode_len = len(indices)

    gt_chunks = []
    pred_chunks = []
    gt_delta_chunks = []
    pred_delta_chunks = []
    chunk_starts = []
    pos = 0

    while pos < episode_len:
        frame_idx = indices[pos]
        sample = ds[int(frame_idx)]
        batch = pre(dict(sample))
        pred = policy.predict_action_chunk(batch)
        pred = postprocess_action(post, pred)
        pred_chunk = pred if pred.ndim == 2 else pred.unsqueeze(0)

        requested_horizon = int(pred_chunk.shape[0])
        stride = requested_horizon if chunk_stride is None else chunk_stride
        keep = stride if take_steps is None else take_steps
        horizon = min(requested_horizon, keep, episode_len - pos)

        gt_abs = collect_abs_gt_chunk(ds, int(frame_idx), horizon)
        kept_pred_delta = pred_chunk[: gt_abs.shape[0]].detach().float().cpu().numpy()
        kept_gt_delta = np.stack([as_numpy_vector(ds[int(frame_idx + offset)][ACTION]) for offset in range(gt_abs.shape[0])])
        pred_abs = chunk_start_deltas_to_abs_poses(sample[OBS_STATE], kept_pred_delta)

        gt_chunks.append(gt_abs)
        pred_chunks.append(pred_abs)
        gt_delta_chunks.append(kept_gt_delta)
        pred_delta_chunks.append(kept_pred_delta)
        chunk_starts.append(frame_idx)

        pos += stride

    gt_episode = np.concatenate(gt_chunks, axis=0)
    pred_episode = np.concatenate(pred_chunks, axis=0)
    gt_delta_episode = np.concatenate(gt_delta_chunks, axis=0)
    pred_delta_episode = np.concatenate(pred_delta_chunks, axis=0)
    length = min(gt_episode.shape[0], pred_episode.shape[0], gt_delta_episode.shape[0], pred_delta_episode.shape[0])
    return {
        "episode_index": episode_index,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "episode_len": episode_len,
        "chunk_starts": chunk_starts,
        "gt_abs": gt_episode[:length],
        "pred_abs": pred_episode[:length],
        "gt_delta": gt_delta_episode[:length],
        "pred_delta": pred_delta_episode[:length],
        "pred_rollout": "chunk-start state",
    }


def save_abs_pose_plot(gt_abs, pred_abs, out_path, title):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    horizon, dim = gt_abs.shape
    t = np.arange(horizon)

    fig, axes = plt.subplots(dim, 1, figsize=(12, 2.25 * dim), sharex=True)
    if dim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        name = ABS_POSE_NAMES[i] if i < len(ABS_POSE_NAMES) else f"dim {i + 1}"
        ax.plot(t, gt_abs[:, i], label="Ground Truth", color="#5266ff", linewidth=1.4)
        ax.plot(t, pred_abs[:, i], label="Prediction", color="#ffad33", linewidth=1.2)
        ax.set_title(f"{title} - {name}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Action Step in Predicted Chunk")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def save_episode_abs_and_delta_plot(gt_abs, pred_abs, gt_delta, pred_delta, out_path, title):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dim_abs = gt_abs.shape[-1]
    dim_delta = gt_delta.shape[-1]
    n_rows = dim_abs + dim_delta
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.05 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    t_abs = np.arange(gt_abs.shape[0])
    for i in range(dim_abs):
        ax = axes[i]
        name = ABS_POSE_NAMES[i] if i < len(ABS_POSE_NAMES) else f"abs_dim_{i + 1}"
        ax.plot(t_abs, gt_abs[:, i], label="Ground Truth", color="#5266ff", linewidth=1.35)
        ax.plot(t_abs, pred_abs[:, i], label="Prediction", color="#ffad33", linewidth=1.15)
        ax.set_title(f"{title} - absolute {name}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right")

    t_delta = np.arange(gt_delta.shape[0])
    for i in range(dim_delta):
        ax = axes[dim_abs + i]
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"delta_dim_{i + 1}"
        ax.plot(t_delta, gt_delta[:, i], label="Ground Truth", color="#5266ff", linewidth=1.35)
        ax.plot(t_delta, pred_delta[:, i], label="Prediction", color="#ffad33", linewidth=1.15)
        ax.set_title(f"{title} - delta {name}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Episode Step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    ckpt = resolve_ckpt(args.ckpt).resolve()
    root = resolve_default_path(args.root, ["./datasets/sponge_pi05_aa_delta_42"]).resolve()
    repo_id = args.repo_id
    if args.repo_id == DEFAULT_REPO_ID and root.name != DEFAULT_REPO_ID:
        repo_id = root.name
    device = torch.device(args.device)

    torch.set_grad_enabled(False)

    hr("0) PATHS")
    print(f"checkpoint : {ckpt}  exists={ckpt.exists()}")
    print(f"dataset    : {root}  exists={root.exists()}")
    print(f"repo_id    : {repo_id}")
    print(f"device     : {device}")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint does not exist: {ckpt}")
    if not root.exists():
        raise SystemExit(f"dataset root does not exist: {root}")

    hr("1) LOAD POLICY + SAVED PROCESSORS")
    policy = PI05Policy.from_pretrained(str(ckpt)).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt))
    print(f"chunk_size       : {getattr(policy.config, 'chunk_size', '?')}")
    print(f"input_features   : {list((policy.config.input_features or {}).keys())}")
    print(f"output_features  : {policy.config.output_features}")

    hr("2) LOAD DATASET")
    try:
        ds = LeRobotDataset(repo_id, root=str(root))
    except Exception as exc:
        raise SystemExit(
            f"could not load dataset with LeRobotDataset: {type(exc).__name__}: {exc}\n"
            "This usually means the converted dataset is incomplete/corrupted. "
            "Re-run convert_to_pi05.py and make sure it finishes with the final 'done -> ...' line."
        ) from exc
    n = getattr(ds, "num_frames", len(ds))
    print(f"frames: {n}")
    sample0 = ds[0]
    for key, value in sample0.items():
        if torch.is_tensor(value):
            print(f"  {key:32s} {tuple(value.shape)} {value.dtype}")
        else:
            print(f"  {key:32s} {type(value).__name__} = {str(value)[:80]}")

    if args.episode_index is not None:
        hr("3) FULL EPISODE INFERENCE: CHUNKED ABSOLUTE POSE VS GT")
        result = evaluate_episode(
            policy,
            pre,
            post,
            ds,
            args.episode_index,
            args.chunk_stride,
            args.take_steps,
        )
        gt_abs = result["gt_abs"]
        pred_abs = result["pred_abs"]
        gt_delta = result["gt_delta"]
        pred_delta = result["pred_delta"]
        compare_dim = min(gt_abs.shape[-1], pred_abs.shape[-1])
        err = np.abs(pred_abs[:, :compare_dim] - gt_abs[:, :compare_dim]).mean()

        saved = save_episode_abs_and_delta_plot(
            gt_abs[:, :compare_dim],
            pred_abs[:, :compare_dim],
            gt_delta,
            pred_delta,
            args.plot_out,
            title=f"aa-delta absolute pose episode {args.episode_index}",
        )
        print(f"episode_index = {args.episode_index}")
        print(f"frames        = {result['start_frame']}..{result['end_frame']} ({result['episode_len']} frames)")
        print(f"chunk starts  = {len(result['chunk_starts'])} chunks")
        print(f"compared len  = {gt_abs.shape[0]}")
        print(f"pred rollout  = {result['pred_rollout']}")
        print(f"mean_abs_error absolute episode = {err:.5f}")
        print(f"plot saved -> {saved}")
        return

    hr("3) INFERENCE: 50-STEP PREDICTED ABSOLUTE POSE VS GT")
    indices = select_frame_indices(n, args.start, args.end, args.frames, args.random, args.seed)
    errors = []

    for idx in indices:
        sample = ds[int(idx)]

        batch = pre(dict(sample))
        pred = policy.predict_action_chunk(batch)
        pred = postprocess_action(post, pred)

        pred_chunk = pred if pred.ndim == 2 else pred.unsqueeze(0)
        requested_horizon = int(pred_chunk.shape[0])
        max_horizon = min(requested_horizon, n - int(idx))
        gt_abs = collect_abs_gt_chunk(ds, int(idx), max_horizon)
        pred_abs = chunk_start_deltas_to_abs_poses(
            sample[OBS_STATE],
            pred_chunk[: gt_abs.shape[0]].detach().float().cpu().numpy(),
        )

        compare_dim = min(gt_abs.shape[-1], pred_abs.shape[-1])
        err = np.abs(pred_abs[:, :compare_dim] - gt_abs[:, :compare_dim]).mean()
        errors.append(float(err))

        pred_first = torch.as_tensor(pred_abs[0])
        gt_first = torch.as_tensor(gt_abs[0])
        print(f"\nframe {idx}")
        print_action("pred_abs[0]", pred_first, compare_dim, ABS_POSE_NAMES)
        print_action("gt_abs[0]", gt_first, compare_dim, ABS_POSE_NAMES)
        print(f"  mean_abs_error absolute chunk = {err:.5f}")
        print(f"  pred delta chunk shape = {tuple(pred.shape)}")
        print(f"  compared horizon = {gt_abs.shape[0]}")

        if idx == indices[0]:
            saved = save_abs_pose_plot(
                gt_abs[:, :compare_dim],
                pred_abs[:, :compare_dim],
                args.plot_out,
                title=f"aa-delta absolute pose chunk @ frame {idx}",
            )
            print(f"  plot saved -> {saved}")

    if errors:
        print(f"\nmean over {len(errors)} sampled frames = {float(np.mean(errors)):.5f}")


if __name__ == "__main__":
    main()
