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
import json
import os
from pathlib import Path

# Local dataset: don't let lerobot query the HF Hub for a dataset "version" (it 404s on a
# local-only repo_id). Set BEFORE importing lerobot. pi05_base must therefore be CACHED
# already (download once with internet: `huggingface-cli download lerobot/pi05_base`).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def read_dataset_fps(root):
    info_path = Path(root) / "meta" / "info.json"
    return int(json.loads(info_path.read_text())["fps"])


def read_dataset_dims(root):
    info_path = Path(root) / "meta" / "info.json"
    features = json.loads(info_path.read_text())["features"]
    state_dim = int(features[OBS_STATE]["shape"][0])
    action_dim = int(features[ACTION]["shape"][0])
    return state_dim, action_dim


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
    dataset = LeRobotDataset(args.repo_id, root=args.root, delta_timestamps=delta_timestamps)
    print(f"dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames "
          f"({args.root})")
    print(f"robot dims: state={state_dim}, action={action_dim} (padded to {MODEL_DIM} internally)")
    print(f"action chunk: {len(delta_timestamps[ACTION])} steps @ {fps} fps")

    # 3) processor —— 不传 pretrained_path，避免加载坏掉的预存配置
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset.meta.stats,
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

    # 6) 训练循环
    step = 0
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

            loss, _ = policy.forward(batch)   # ← 想改 loss / 做研究就在这附近
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {step:4d}  loss {loss.item():.4f}  peak_mem {mem:.1f}GB")

            if step > 0 and step % args.save_every == 0:
                save_checkpoint(policy, preprocess, postprocess, f"{args.output_dir}/step_{step}")
                print(f"  saved checkpoint -> {args.output_dir}/step_{step}")

            step += 1
            if step >= args.steps:
                break

    save_checkpoint(policy, preprocess, postprocess, f"{args.output_dir}/final")
    print(f"done. final model -> {args.output_dir}/final")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 改架构做研究：
#   1) 复制 lerobot/policies/pi05/modeling_pi05.py 到本地，改写 PI05Policy / PI05Pytorch
#   2) 这里 import 你的版本代替 from lerobot.policies.pi05 import PI05Policy
#   3) forward() 返回的 loss 可以替换/叠加你自己的 loss 项
# ---------------------------------------------------------------------------
