#!/usr/bin/env python
"""
pi05 自定义 finetune 脚本（DROID Franka 数据）。

要点：
1. 不传 pretrained_path 给 make_pre_post_processors，避免加载 base 里过时的
   'relative_actions_processor' 配置。
2. pi05_base 内部按 32 维 state/action 工作，而数据可能是 8/15 维，
   forward 前手动 pad 到 32（前面真实值，后面补 0）。
3. 训练循环完全暴露，方便改 loss / 加模块做研究。

用法（数据路径等都可命令行覆盖，默认值见下方常量）：
    python train_pi05.py --root /path/to/<dataset_dir> --repo-id pick_up_sponge --steps 200
"""
import argparse
import copy
import json
import os
from pathlib import Path

# Local dataset: don't let lerobot query the HF Hub for a dataset "version" (it 404s on a
# local-only repo_id). Set BEFORE importing lerobot. pi05_base must therefore be CACHED
# already (download once with internet: `huggingface-cli download lerobot/pi05_base`).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

# ---------------- 默认配置（可被命令行覆盖）----------------
REPO_ID = "local/droid_smoke"
ROOT = "/home/prs/ros_ml_ws/src/pi05/datasets/droid_lerobot"
PRETRAINED = "lerobot/pi05_base"
OUTPUT_DIR = "/home/prs/ros_ml_ws/src/pi05/models/droid_smoke"
DEVICE = "cuda"
BATCH_SIZE = 8
STEPS = 200
LOG_EVERY = 20
SAVE_EVERY = 100
LR = 2.5e-5
MODEL_DIM = 32   # pi05_base 期望的 state/action 维度
# -----------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="pi05 finetune (exposed loop)")
    ap.add_argument("--root", default=ROOT, help="dataset dir containing meta/info.json")
    ap.add_argument("--repo-id", default=REPO_ID, help="LeRobot repo_id label")
    ap.add_argument("--pretrained", default=PRETRAINED, help="base ckpt (HF id or local path)")
    ap.add_argument("--output-dir", default=OUTPUT_DIR, help="where to save checkpoints")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", help="model dtype; bfloat16 saves VRAM")
    ap.add_argument("--full-model", action="store_true", help="train all pi05 weights instead of expert only")
    ap.add_argument("--no-gradient-checkpointing", action="store_true", help="disable activation checkpointing")
    ap.add_argument("--wandb", action="store_true", help="log training metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="pi05_finetune")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-log-checkpoints", action="store_true", help="upload saved checkpoints as wandb artifacts")
    ap.add_argument("--action-delta-reference", default="chunk-start",
                    choices=["chunk-start", "dataset"],
                    help="chunk-start rewrites each action chunk to use the sample's first observation as delta base")
    ap.add_argument("--input-action-format", default="same-frame-delta",
                    choices=["same-frame-delta", "absolute"],
                    help="format stored in the dataset before any dynamic rewrite")
    return ap.parse_args()


def pad_to(t, dim):
    """把最后一维 pad 到 dim（补 0）。"""
    if t.shape[-1] < dim:
        return F.pad(t, (0, dim - t.shape[-1]))
    return t


def pad_or_trim_language(batch, length):
    """pi05 expects fixed-length language tokens/masks for attention construction."""
    if OBS_LANGUAGE_TOKENS not in batch or OBS_LANGUAGE_ATTENTION_MASK not in batch:
        return batch

    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    if tokens.shape[-1] > length:
        batch[OBS_LANGUAGE_TOKENS] = tokens[..., :length]
    elif tokens.shape[-1] < length:
        batch[OBS_LANGUAGE_TOKENS] = F.pad(tokens, (0, length - tokens.shape[-1]), value=0)

    if masks.shape[-1] > length:
        batch[OBS_LANGUAGE_ATTENTION_MASK] = masks[..., :length]
    elif masks.shape[-1] < length:
        batch[OBS_LANGUAGE_ATTENTION_MASK] = F.pad(masks, (0, length - masks.shape[-1]), value=False)

    return batch


def save_checkpoint(policy, preprocess, postprocess, path):
    policy.save_pretrained(path)
    preprocess.save_pretrained(path)
    postprocess.save_pretrained(path)


def init_wandb(args, extra_config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this environment") from exc

    config = vars(args).copy()
    config.update(extra_config)
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=config,
    )


def log_checkpoint_artifact(run, path, step):
    if run is None:
        return
    import wandb

    path = Path(path)
    artifact = wandb.Artifact(name=f"{run.name or 'pi05'}-step-{step}", type="model")
    artifact.add_dir(str(path))
    run.log_artifact(artifact)


def read_dataset_fps(root):
    info_path = Path(root) / "meta" / "info.json"
    return int(json.loads(info_path.read_text())["fps"])


def read_dataset_dims(root):
    info_path = Path(root) / "meta" / "info.json"
    features = json.loads(info_path.read_text())["features"]
    state_dim = int(features[OBS_STATE]["shape"][0])
    action_dim = int(features[ACTION]["shape"][0])
    return state_dim, action_dim


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


def load_lowdim_arrays(root):
    data_root = Path(root) / "data"
    parquet_files = sorted(data_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no data parquet files found under {data_root}")

    dfs = [pd.read_parquet(path) for path in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    return df[["episode_index", OBS_STATE, ACTION]]


def adapt_policy_features_to_dataset(policy_config, state_dim, action_dim):
    """Keep pi05's 32-dim container, but expose the real robot dims for loss/output."""
    policy_config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE,
        shape=(state_dim,),
    )
    policy_config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION,
        shape=(action_dim,),
    )


def make_action_delta_timestamps(policy_config, fps):
    indices = getattr(policy_config, "action_delta_indices", None)
    if indices is None:
        indices = range(getattr(policy_config, "chunk_size", 1))
    return {ACTION: [i / fps for i in indices]}


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def same_frame_delta_chunk_to_absolute(state_chunk, action_chunk):
    """Invert convert_to_pi05.py's aa same-frame delta for a whole action chunk."""
    states = to_numpy(state_chunk).astype(np.float32)
    actions = to_numpy(action_chunk).astype(np.float32)
    abs_actions = np.empty_like(actions, dtype=np.float32)
    abs_actions[:, :3] = states[:, :3] + actions[:, :3]
    for i, (state, action) in enumerate(zip(states, actions, strict=False)):
        R_abs = Rotation.from_rotvec(state[3:6]) * Rotation.from_rotvec(action[3:6])
        abs_actions[i, 3:6] = R_abs.as_rotvec().astype(np.float32)
    abs_actions[:, 6:] = actions[:, 6:]
    return abs_actions


def absolute_chunk_to_chunk_start_delta(start_state, abs_actions):
    """Encode absolute aa targets relative to the sample's first observation.state."""
    state0 = to_numpy(start_state).astype(np.float32)
    abs_actions = to_numpy(abs_actions).astype(np.float32)
    deltas = np.empty_like(abs_actions, dtype=np.float32)
    deltas[:, :3] = abs_actions[:, :3] - state0[:3]
    R0_inv = Rotation.from_rotvec(state0[3:6]).inv()
    for i, action in enumerate(abs_actions):
        deltas[i, 3:6] = (R0_inv * Rotation.from_rotvec(action[3:6])).as_rotvec().astype(np.float32)
    deltas[:, 6:] = abs_actions[:, 6:]
    return deltas


def compute_chunk_start_action_stats(root, action_offsets, input_action_format):
    """Compute the exact action stats used by ChunkStartDeltaDataset labels."""
    df = load_lowdim_arrays(root)
    labels = []
    for _ep, group in df.groupby("episode_index", sort=True):
        states = np.stack(group[OBS_STATE].to_numpy()).astype(np.float32)
        actions = np.stack(group[ACTION].to_numpy()).astype(np.float32)
        if input_action_format == "same-frame-delta":
            abs_actions = same_frame_delta_chunk_to_absolute(states, actions)
        else:
            abs_actions = actions

        for start in range(len(states)):
            valid_indices = [
                start + offset
                for offset in action_offsets
                if start + offset < len(states)
            ]
            if not valid_indices:
                continue
            labels.append(
                absolute_chunk_to_chunk_start_delta(
                    states[start],
                    abs_actions[valid_indices],
                )
            )

    if not labels:
        raise RuntimeError("could not compute chunk-start action stats: no labels found")
    return summarize_stats(np.concatenate(labels, axis=0))


class ChunkStartDeltaDataset(torch.utils.data.Dataset):
    """LeRobot wrapper that materializes openpi-style chunk-start action deltas.

    The base dataset provides the current observation/image/task.  The sequence
    dataset provides the future action chunk plus the matching future states.
    """

    def __init__(self, repo_id, root, delta_timestamps, input_action_format):
        self.base = LeRobotDataset(repo_id, root=root)
        sequence_timestamps = {
            ACTION: delta_timestamps[ACTION],
            OBS_STATE: delta_timestamps[ACTION],
        }
        self.sequence = LeRobotDataset(repo_id, root=root, delta_timestamps=sequence_timestamps)
        self.input_action_format = input_action_format

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def meta(self):
        return self.base.meta

    def __getitem__(self, index):
        item = self.base[index]
        sequence = self.sequence[index]
        action_chunk = sequence[ACTION]
        state_chunk = sequence[OBS_STATE]
        if self.input_action_format == "same-frame-delta":
            abs_actions = same_frame_delta_chunk_to_absolute(state_chunk, action_chunk)
        else:
            abs_actions = to_numpy(action_chunk).astype(np.float32)
        chunk_start_delta = absolute_chunk_to_chunk_start_delta(item[OBS_STATE], abs_actions)
        item[ACTION] = torch.as_tensor(chunk_start_delta, dtype=action_chunk.dtype)
        return item


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) 模型
    policy_config = PreTrainedConfig.from_pretrained(args.pretrained)
    policy_config.device = str(device)
    if hasattr(policy_config, "dtype"):
        policy_config.dtype = args.dtype
    if hasattr(policy_config, "train_expert_only"):
        policy_config.train_expert_only = not args.full_model
    if hasattr(policy_config, "gradient_checkpointing"):
        policy_config.gradient_checkpointing = not args.no_gradient_checkpointing

    state_dim, action_dim = read_dataset_dims(args.root)
    adapt_policy_features_to_dataset(policy_config, state_dim, action_dim)

    policy = PI05Policy.from_pretrained(args.pretrained, config=policy_config)
    policy.to(device)
    policy.train()

    # 2) 数据集：pi05 forward 训练 50-step action chunk，不是单步 action
    fps = read_dataset_fps(args.root)
    delta_timestamps = make_action_delta_timestamps(policy.config, fps)
    if args.action_delta_reference == "chunk-start":
        dataset = ChunkStartDeltaDataset(
            args.repo_id,
            root=args.root,
            delta_timestamps=delta_timestamps,
            input_action_format=args.input_action_format,
        )
    else:
        dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)
    print(f"dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames "
          f"({args.root})")
    print(f"robot dims: state={state_dim}, action={action_dim} (padded to {MODEL_DIM} internally)")
    print(f"action chunk: {len(delta_timestamps[ACTION])} steps @ {fps} fps")
    print(f"action delta reference: {args.action_delta_reference}")
    if args.action_delta_reference == "chunk-start":
        print(f"input action format: {args.input_action_format}")

    dataset_stats = copy.deepcopy(dataset.meta.stats)
    if args.action_delta_reference == "chunk-start":
        action_offsets = [int(round(t * fps)) for t in delta_timestamps[ACTION]]
        action_stats = compute_chunk_start_action_stats(
            args.root,
            action_offsets=action_offsets,
            input_action_format=args.input_action_format,
        )
        dataset_stats[ACTION] = action_stats
        print(
            "action normalization stats: recomputed from chunk-start labels "
            f"(count={action_stats['count'][0]}, std[:6]="
            f"{np.asarray(action_stats['std'][:6]).round(6).tolist()})"
        )

    # 3) processor —— 不传 pretrained_path，避免加载坏掉的预存配置
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 4) DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 5) optimizer
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"trainable params: {trainable_count/1e6:.1f}M / {total_params/1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    wandb_run = init_wandb(
        args,
        {
            "dataset_num_episodes": dataset.num_episodes,
            "dataset_num_frames": dataset.num_frames,
            "dataset_fps": fps,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_chunk_steps": len(delta_timestamps[ACTION]),
            "total_params": total_params,
            "trainable_params": trainable_count,
            "trainable_param_fraction": trainable_count / total_params,
            "policy_dtype": getattr(policy.config, "dtype", None),
            "train_expert_only": getattr(policy.config, "train_expert_only", None),
            "gradient_checkpointing": getattr(policy.config, "gradient_checkpointing", None),
        },
    )

    # 6) 训练循环
    step = 0
    try:
        while step < args.steps:
            for batch in loader:
                batch = preprocess(batch)
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }

                # pad state/action -> 32
                batch["observation.state"] = pad_to(batch["observation.state"], MODEL_DIM)
                batch["action"] = pad_to(batch["action"], MODEL_DIM)
                batch = pad_or_trim_language(batch, policy.config.tokenizer_max_length)

                loss, loss_dict = policy.forward(batch)   # ← 想改 loss / 做研究就在这附近
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if step % args.log_every == 0:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"step {step:4d}  loss {loss.item():.4f}  peak_mem {mem:.1f}GB")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": loss.item(),
                                "train/grad_norm": float(grad_norm),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "system/peak_mem_gb": mem,
                                "step": step,
                                **{
                                    f"train/loss_dim_{i}": value
                                    for i, value in enumerate(loss_dict.get("loss_per_dim", []))
                                },
                            },
                            step=step,
                        )

                if step > 0 and step % args.save_every == 0:
                    ckpt_path = f"{args.output_dir}/step_{step}"
                    save_checkpoint(policy, preprocess, postprocess, ckpt_path)
                    print(f"  saved checkpoint -> {ckpt_path}")
                    if wandb_run is not None:
                        wandb_run.log({"checkpoint/step_path": ckpt_path}, step=step)
                        if args.wandb_log_checkpoints:
                            log_checkpoint_artifact(wandb_run, ckpt_path, step)

                step += 1
                if step >= args.steps:
                    break

        final_path = f"{args.output_dir}/final"
        save_checkpoint(policy, preprocess, postprocess, final_path)
        print(f"done. final model -> {final_path}")
        if wandb_run is not None:
            wandb_run.log({"checkpoint/final_path": final_path}, step=step)
            if args.wandb_log_checkpoints:
                log_checkpoint_artifact(wandb_run, final_path, "final")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 改架构做研究：
#   1) 复制 lerobot/policies/pi05/modeling_pi05.py 到本地，改写 PI05Policy / PI05Pytorch
#   2) 这里 import 你的版本代替 from lerobot.policies.pi05 import PI05Policy
#   3) forward() 返回的 loss 可以替换/叠加你自己的 loss 项
# ---------------------------------------------------------------------------
