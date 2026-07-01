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
import os

# Local dataset: don't let lerobot query the HF Hub for a dataset "version" (it 404s on a
# local-only repo_id). Set BEFORE importing lerobot. pi05_base must therefore be CACHED
# already (download once with internet: `huggingface-cli download lerobot/pi05_base`).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

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
    return ap.parse_args()


def pad_to(t, dim):
    """把最后一维 pad 到 dim（补 0）。"""
    if t.shape[-1] < dim:
        return F.pad(t, (0, dim - t.shape[-1]))
    return t


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 1) 数据集
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    print(f"dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames "
          f"({args.root})")

    # 2) 模型
    policy = PI05Policy.from_pretrained(args.pretrained)
    if hasattr(policy.config, "train_expert_only"):
        policy.config.train_expert_only = True
    policy.to(device)
    policy.train()

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
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

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

            loss, _ = policy.forward(batch)   # ← 想改 loss / 做研究就在这附近
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"step {step:4d}  loss {loss.item():.4f}  peak_mem {mem:.1f}GB")

            if step > 0 and step % args.save_every == 0:
                policy.save_pretrained(f"{args.output_dir}/step_{step}")
                print(f"  saved checkpoint -> {args.output_dir}/step_{step}")

            step += 1
            if step >= args.steps:
                break

    policy.save_pretrained(f"{args.output_dir}/final")
    print(f"done. final model -> {args.output_dir}/final")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 改架构做研究：
#   1) 复制 lerobot/policies/pi05/modeling_pi05.py 到本地，改写 PI05Policy / PI05Pytorch
#   2) 这里 import 你的版本代替 from lerobot.policies.pi05 import PI05Policy
#   3) forward() 返回的 loss 可以替换/叠加你自己的 loss 项
# ---------------------------------------------------------------------------
