#!/usr/bin/env python3
"""Convert a recorded Franka Cartesian-EE LeRobot dataset into a pi0.5-ingestible format,
written out as a NEW LeRobot v3.0 dataset.

pi05 has NO hardwired action space -- the flow-matching head regresses whatever continuous
vector you give it (zero-padded to 32, quantile-normalized). So it "supports" any rotation
rep mechanically. But its ONLY EE reference embodiment (LIBERO) uses axis-angle(3) + DELTA;
DROID/ALOHA are joint-space. So pick per goal:

  --rot aa  --action-mode delta      -> closest to pi05's LIBERO reference format
  --rot 6d  --action-mode absolute   -> best NN regression target + plugs into our
                                        absolute-EE cartesian_impedance controller (default)

Layout produced (keys stay dotted = HF-lerobot PI05Policy eats them directly; pads 10/7 -> 32):
  observation.state  = [x,y,z, <rot>, gripper]        (rot always ABSOLUTE proprioception)
  action             = [x,y,z, <rot>, gripper]        (ABSOLUTE, or DELTA vs same-frame state)
  observation.images.base = video (model resizes to 224 internally; not resized here)

rot dims: 6d->6, aa->3, quat->4.  gripper normalized to [0,1] (1=open, 0=closed).

Usage (ros_ml env):
    python convert_to_pi05.py --root /path/<src> --out /path/<dst>                 # 6d + absolute
    python convert_to_pi05.py --root /path/<src> --out /path/<dst> --rot aa --action-mode delta
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset

GRIPPER_MAX_WIDTH = 0.04   # per-finger max qpos (m) -> normalises state gripper to [0,1]

ROT_NAMES = {
    "6d": ["r00", "r10", "r20", "r01", "r11", "r21"],
    "aa": ["rx", "ry", "rz"],
    "quat": ["qx", "qy", "qz", "qw"],
}


def encode_rot(R, rep):
    """scipy Rotation -> chosen representation as a float32 vector."""
    if rep == "quat":
        return R.as_quat().astype(np.float32)                       # (x,y,z,w)
    if rep == "aa":
        return R.as_rotvec().astype(np.float32)                     # axis-angle (3)
    m = R.as_matrix()                                               # 6d = first two columns
    return np.concatenate([m[:, 0], m[:, 1]]).astype(np.float32)


def to_np(x):
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def make_state(state, rep):
    """[x,y,z, qx,qy,qz,qw, wrench(6), grip0, grip1] (15) -> [xyz, rot, grip]."""
    s = to_np(state)
    pos = s[0:3].astype(np.float32)
    rot = encode_rot(Rotation.from_quat(s[3:7]), rep)
    grip = np.array([float(np.clip(s[13] / GRIPPER_MAX_WIDTH, 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def make_action(action, state, rep, mode):
    """[x,y,z, qx,qy,qz,qw, grip] (8, ABSOLUTE) -> [xyz, rot, grip].
    mode='absolute': as-is. mode='delta': position - current EE pos (base frame), and the
    relative rotation R_state^-1 * R_action (openpi DeltaActions convention). Gripper stays
    absolute in [0,1] either way."""
    a = to_np(action)
    s = to_np(state)
    R_act = Rotation.from_quat(a[3:7])
    grip = np.array([float(np.clip(a[7], 0.0, 1.0))], dtype=np.float32)
    if mode == "delta":
        pos = (a[0:3] - s[0:3]).astype(np.float32)
        R_rel = Rotation.from_quat(s[3:7]).inv() * R_act
        rot = encode_rot(R_rel, rep)
    else:
        pos = a[0:3].astype(np.float32)
        rot = encode_rot(R_act, rep)
    return np.concatenate([pos, rot, grip]).astype(np.float32)


def chw_float_to_hwc_uint8(img):
    """LeRobot decodes video to CHW float32 [0,1]; the writer wants HWC uint8."""
    t = img if torch.is_tensor(img) else torch.as_tensor(img)
    t = (t.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return t.permute(1, 2, 0).contiguous().numpy()


def main():
    ap = argparse.ArgumentParser(description="convert Franka EE dataset -> pi0.5 format")
    ap.add_argument("--root", required=True, help="source dataset dir (has meta/info.json)")
    ap.add_argument("--out", required=True, help="destination dir for the converted dataset")
    ap.add_argument("--repo-id", default=None, help="repo_id for the new dataset (default: out name)")
    ap.add_argument("--rot", choices=["6d", "aa", "quat"], default="6d",
                    help="rotation representation (default 6d; 'aa'=axis-angle like LIBERO)")
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                    help="absolute EE (default, matches our controller) or delta EE (like LIBERO)")
    ap.add_argument("--cam-key", default="observation.images.base", help="camera feature key")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "meta", "info.json")):
        raise SystemExit(f"destination already has a dataset: {args.out} (delete it or pick a new --out)")

    src = LeRobotDataset(os.path.basename(args.root), root=args.root)
    print(f"source: {src.num_episodes} episodes, {src.num_frames} frames @ {src.fps} fps")
    print(f"target: rot={args.rot}  action-mode={args.action_mode}")
    H, W, _ = src.features[args.cam_key]["shape"]
    rot_names = ROT_NAMES[args.rot]
    dim = 3 + len(rot_names) + 1
    names = ["x", "y", "z"] + rot_names + ["gripper"]

    features = {
        args.cam_key: {"dtype": "video", "shape": (H, W, 3),
                       "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
        "action": {"dtype": "float32", "shape": (dim,), "names": names},
    }

    dst = LeRobotDataset.create(
        repo_id=args.repo_id or os.path.basename(os.path.normpath(args.out)),
        fps=int(src.fps), root=args.out, robot_type="franka_fr3",
        features=features, use_videos=True)

    prev_ep = None
    for i in range(src.num_frames):
        f = src[i]
        ep = int(f["episode_index"])
        if prev_ep is not None and ep != prev_ep:
            dst.save_episode()
        dst.add_frame({
            args.cam_key: chw_float_to_hwc_uint8(f[args.cam_key]),
            "observation.state": make_state(f["observation.state"], args.rot),
            "action": make_action(f["action"], f["observation.state"], args.rot, args.action_mode),
            "task": f["task"],
        })
        prev_ep = ep
        if i % 500 == 0:
            print(f"  frame {i}/{src.num_frames} (episode {ep})")
    dst.save_episode()
    if hasattr(dst, "finalize"):
        dst.finalize()
    print(f"done -> {args.out}  ({src.num_episodes} episodes, state/action dim={dim})")


if __name__ == "__main__":
    main()
