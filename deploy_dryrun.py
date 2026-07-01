#!/usr/bin/env python3
"""Offline deploy dry-run for the finetuned pi05 model.

Feeds REAL frames from the DROID LeRobot dataset through the SAME processor
pipeline used in training, runs the policy, and prints the PREDICTED action next
to the GROUND-TRUTH action. This is the sanity check before writing the ROS2 node:
if predicted ~ ground-truth, loading + processors + inference all work.

Run (in the `ros_ml` env, on the machine with the checkpoint + dataset):
    python deploy_dryrun.py

The script is intentionally self-diagnosing: it prints API signatures, the model
config, and the dataset sample structure as it goes. If something mismatches the
installed lerobot 0.5.1 API, paste the whole output back and it's a 1-line fix.
"""
from __future__ import annotations

import inspect
import traceback
from pathlib import Path

import numpy as np
import torch

# ----------------------------------------------------------------- config
HOME = Path.home()
CKPT = HOME / "ros_ml_ws/src/pi05/models/droid_smoke/checkpoints/last/pretrained_model"
DATA_ROOT = HOME / "ros_ml_ws/src/pi05/datasets/droid_lerobot"
REPO_ID = "local/droid_smoke"
N_FRAMES = 3                                   # how many dataset frames to test
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except Exception as e:  # noqa: BLE001
        return f"<no signature: {e}>"


def describe(name, obj):
    if torch.is_tensor(obj):
        flat = obj.flatten().float()
        return (f"{name}: tensor{tuple(obj.shape)} {obj.dtype} "
                f"min={flat.min():.3f} max={flat.max():.3f}")
    return f"{name}: {type(obj).__name__} = {str(obj)[:70]}"


# ----------------------------------------------------------------- 0) paths
hr("0) PATHS / ENV")
print(f"  device      = {DEVICE}")
print(f"  checkpoint  = {CKPT}   exists={CKPT.exists()}")
print(f"  data root   = {DATA_ROOT}   exists={DATA_ROOT.exists()}")
import lerobot  # noqa: E402
print(f"  lerobot     = {lerobot.__version__}")

# ----------------------------------------------------------------- 1) policy
hr("1) LOAD POLICY")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

print("  PI05Policy.from_pretrained", sig(PI05Policy.from_pretrained))
policy = PI05Policy.from_pretrained(str(CKPT)).to(DEVICE).eval()
cfg = policy.config
print("  loaded OK. config fields of interest:")
for k in ("n_action_steps", "chunk_size", "n_obs_steps", "max_action_dim",
          "max_state_dim", "n_action_pred_token", "resize_imgs_with_padding"):
    if hasattr(cfg, k):
        print(f"     {k} = {getattr(cfg, k)}")
print("  input_features :", list(getattr(cfg, "input_features", {}) or {}))
print("  output_features:", list(getattr(cfg, "output_features", {}) or {}))
print("  select_action       ", sig(policy.select_action))
if hasattr(policy, "predict_action_chunk"):
    print("  predict_action_chunk", sig(policy.predict_action_chunk))

# ----------------------------------------------------------------- 2) processors
hr("2) BUILD PRE/POST PROCESSORS (from the SAME checkpoint dir)")
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

print("  make_pre_post_processors", sig(make_pre_post_processors))
pre = post = None
attempts = [
    ("cfg + pretrained_path", dict(policy_cfg=cfg, pretrained_path=str(CKPT))),
    ("config + pretrained_path", dict(config=cfg, pretrained_path=str(CKPT))),
    ("policy_cfg + dataset_stats=None", dict(policy_cfg=cfg, pretrained_path=str(CKPT),
                                             dataset_stats=None)),
    ("positional cfg, path", None),  # handled below
]
for label, kwargs in attempts:
    try:
        if kwargs is None:
            pre, post = make_pre_post_processors(cfg, str(CKPT))
        else:
            pre, post = make_pre_post_processors(**kwargs)
        print(f"  -> OK via [{label}]")
        break
    except Exception as e:  # noqa: BLE001
        print(f"  -> fail [{label}]: {repr(e)[:130]}")
if pre is None:
    print("  !! could not build processors automatically. Signature is printed above;")
    print("     paste it back and I'll pin the exact call.")

# ----------------------------------------------------------------- 3) dataset
hr("3) LOAD DATASET + INSPECT ONE SAMPLE")
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset(REPO_ID, root=str(DATA_ROOT))
n = getattr(ds, "num_frames", None) or len(ds)
print(f"  dataset loaded: {n} frames")
sample0 = ds[0]
print("  sample keys + shapes:")
for k in sample0:
    print("    ", describe(k, sample0[k]))
# language/task key
task_key = next((k for k in sample0 if "task" in k.lower() or "language" in k.lower()
                 or "instruction" in k.lower()), None)
print(f"  detected task/language key: {task_key}")

# ----------------------------------------------------------------- 4) inference
hr("4) INFERENCE: predicted action vs ground-truth")
if pre is None:
    print("  skipped (processors not built).")
else:
    idxs = np.linspace(0, n - 1, N_FRAMES).astype(int)
    for fi in idxs:
        s = ds[int(fi)]
        gt = s.get("action")
        # build a fresh observation dict (tensors as-is; processor adds batch/normalizes)
        obs = {k: v for k, v in s.items()}
        try:
            batch = pre(obs)
            # prefer the full chunk so we can compare cleanly
            if hasattr(policy, "predict_action_chunk"):
                out = policy.predict_action_chunk(batch)
            else:
                out = policy.select_action(batch)
            try:
                out = post(out)
            except Exception:  # post may expect a dict; tolerate raw tensor
                pass
            act = out["action"] if isinstance(out, dict) and "action" in out else out
            act = act.detach().float().cpu().squeeze()
            if act.ndim == 2:           # (chunk, dim) -> first action
                act_first = act[0]
            else:
                act_first = act
            gt_t = gt.detach().float().cpu().squeeze() if torch.is_tensor(gt) else None
            print(f"\n  frame {fi}:")
            print(f"    PRED action[:8] = {np.round(act_first[:8].numpy(), 4)}")
            if gt_t is not None:
                print(f"    GT   action[:8] = {np.round(gt_t[:8].numpy(), 4)}")
                err = (act_first[:8] - gt_t[:8]).abs().mean().item()
                print(f"    mean|pred-gt| (first 8 dims) = {err:.4f}")
            print(f"    pred full shape = {tuple(act.shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"\n  frame {fi}: inference FAILED -> {e}")
            traceback.print_exc()
            print("    (signatures + sample structure above tell us how to fix)")
            break

print("\nDONE. Paste the whole output back.")
