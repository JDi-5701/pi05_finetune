#!/usr/bin/env python3
"""Check whether a recorded LeRobot dataset is compatible with pi05 finetune.

Run on the machine that has the recorded dataset (recorder env, or ros_ml):
    python check_pi05_compat.py --root /path/to/dataset_dir          # dir that contains meta/info.json
    python check_pi05_compat.py --root ... --sample                   # also load one frame & show shapes
    python check_pi05_compat.py --root ... --pi05-ckpt lerobot/pi05_base   # also dry-run pi05 processor

What it does (no writes, read-only):
  1. Reads meta/info.json DIRECTLY (version-agnostic) -> the dataset schema.
  2. Classifies features: cameras (image/video) vs low-dim (state/action/...).
  3. Prints a PASS / WARN / FAIL report against pi05 finetune requirements.
  4. (--sample) loads ds[0] via LeRobotDataset and prints real tensor shapes.
  5. (--pi05-ckpt) tries to build the pi05 processor and push one frame through it.

pi05 finetune requirements (from the working DROID setup):
  - LeRobot v3.0 format
  - >=1 camera image feature   (pi05 is a vision model -> images are mandatory)
  - observation.state present, dim <= 32   (pi05 pads to 32)
  - action present, dim <= 32              (pi05 pads to 32)
  - a language/task field
  DROID camera names: base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb (others are
  fine for finetune but must be remapped in the train config).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # local-only dataset, don't hit the Hub
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

MAX_DIM = 32
DROID_CAMS = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
IMG_DTYPES = {"video", "image"}

PASS, WARN, FAIL = "✅ PASS", "⚠️  WARN", "❌ FAIL"


def hr(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def find_info(root: Path) -> Path:
    for c in (root / "meta" / "info.json", root / "info.json"):
        if c.exists():
            return c
    # maybe they passed the parent; search one level down
    hits = list(root.glob("*/meta/info.json"))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"no meta/info.json under {root}. Pass --root pointing at the dataset dir "
        f"(the one containing the 'meta' folder).")


def is_image(meta: dict) -> bool:
    return meta.get("dtype") in IMG_DTYPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset dir containing meta/info.json")
    ap.add_argument("--repo-id", default=None, help="repo_id (for --sample load; defaults to dir name)")
    ap.add_argument("--sample", action="store_true", help="also load ds[0] and show real shapes")
    ap.add_argument("--pi05-ckpt", default=None, help="also dry-run the pi05 processor with this ckpt path")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    info_path = find_info(root)
    ds_root = info_path.parent.parent          # .../<dataset>/  (parent of meta/)
    info = json.loads(info_path.read_text())
    print(f"reading {info_path}")

    # ---------------------------------------------------------------- 1) header
    hr("1) DATASET HEADER")
    cv = str(info.get("codebase_version", "?"))
    fps = info.get("fps", "?")
    robot = info.get("robot_type", "?")
    n_ep = info.get("total_episodes", info.get("num_episodes", "?"))
    n_fr = info.get("total_frames", info.get("num_frames", "?"))
    print(f"  codebase_version : {cv}")
    print(f"  fps              : {fps}")
    print(f"  robot_type       : {robot}")
    print(f"  episodes / frames: {n_ep} / {n_fr}")

    feats = info.get("features", {})
    if not feats:
        print("  !! no 'features' in info.json — cannot assess. Dump:")
        print(json.dumps(info, indent=2)[:1500]); return

    # ---------------------------------------------------------------- 2) features
    hr("2) FEATURES (cameras vs low-dim)")
    cams, lowdim = {}, {}
    for name, meta in feats.items():
        (cams if is_image(meta) else lowdim).__setitem__(name, meta)
    print("  -- cameras (image/video) --")
    if cams:
        for n, m in cams.items():
            print(f"     {n:32s} {m.get('dtype'):6s} shape={tuple(m.get('shape', []))}")
    else:
        print("     (none)")
    print("  -- low-dim --")
    for n, m in lowdim.items():
        shp = tuple(m.get("shape", []))
        nm = m.get("names")
        print(f"     {n:32s} {str(m.get('dtype')):8s} shape={shp}"
              + (f"  names={nm}" if nm else ""))

    def dim_of(name):
        m = feats.get(name)
        if not m:
            return None
        s = m.get("shape", [])
        return int(s[0]) if s else None

    # ---------------------------------------------------------------- 3) verdict
    hr("3) pi05 FINETUNE COMPATIBILITY")
    rows = []

    # v3.0
    rows.append((PASS if cv.startswith("v3") or cv.startswith("3")
                 else FAIL, f"LeRobot format = {cv}",
                 "" if (cv.startswith("v3") or cv.startswith("3"))
                 else "pi05 finetune needs v3.0; v2.0 too old -> re-record/convert with lerobot 0.5.x"))

    # cameras
    if cams:
        named_ok = [c for c in cams if c.split("observation.images.")[-1] in DROID_CAMS]
        rows.append((PASS, f"{len(cams)} camera(s): {list(cams)}",
                     "" if named_ok else "names differ from DROID (base_0_rgb/left_wrist_0_rgb/"
                     "right_wrist_0_rgb) -> remap in train config, or rename. OK for finetune."))
    else:
        rows.append((FAIL, "0 cameras", "pi05 is a vision model -> images are MANDATORY. "
                     "Enable cameras in recorder.yaml (currently commented out)."))

    # state
    sd = dim_of("observation.state")
    if sd is None:
        rows.append((FAIL, "observation.state missing", "pi05 needs a state vector."))
    else:
        rows.append((PASS if sd <= MAX_DIM else FAIL, f"observation.state dim = {sd}",
                     "" if sd <= MAX_DIM else f">32 won't fit pi05's 32-dim container."))

    # action
    ad = dim_of("action")
    anames = feats.get("action", {}).get("names") or []
    if ad is None:
        rows.append((FAIL, "action missing", "pi05 needs an action vector."))
    else:
        note = "" if ad <= MAX_DIM else ">32 won't fit pi05's 32-dim container."
        # quaternion-in-action smell test (poor VLA regression target)
        if any(str(x).lower() in ("qx", "qy", "qz", "qw") for x in anames):
            note = ("action contains a QUATERNION (qx..qw). Bad regression target for VLA; "
                    "DROID/pi05_base prior is JOINT-position. Convert to euler/6D, or record "
                    "joint-position actions. " + note)
            rows.append((WARN, f"action dim = {ad}  names={anames}", note))
        else:
            rows.append((PASS if ad <= MAX_DIM else FAIL, f"action dim = {ad}  names={anames}", note))

    # language / task
    has_task = "task" in feats or "task_index" in feats or (ds_root / "meta" / "tasks.jsonl").exists() \
        or (ds_root / "meta" / "tasks.parquet").exists()
    rows.append((PASS if has_task else WARN, f"language/task present = {has_task}",
                 "" if has_task else "no task field found; pi05 wants a language instruction per episode."))

    for status, what, note in rows:
        print(f"  {status}  {what}")
        if note:
            print(f"           -> {note}")

    blockers = [w for s, w, _ in rows if s == FAIL]
    print("\n  " + ("🚫 NOT finetune-ready. Blockers: " + "; ".join(blockers)
                    if blockers else "🎉 Schema looks finetune-ready (mind any WARN above)."))

    # ---------------------------------------------------------------- 4) sample
    if args.sample:
        hr("4) SAMPLE FRAME (ds[0])")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            repo_id = args.repo_id or ds_root.name
            ds = LeRobotDataset(repo_id, root=str(ds_root))
            s = ds[0]
            import torch
            for k, v in s.items():
                if torch.is_tensor(v):
                    print(f"  {k:32s} {tuple(v.shape)} {v.dtype}")
                else:
                    print(f"  {k:32s} {type(v).__name__} = {str(v)[:50]}")
        except Exception as e:
            print(f"  could not load sample: {e}")
            print("  (schema report above is still valid — it came from info.json directly)")

    # ---------------------------------------------------------------- 5) processor
    if args.pi05_ckpt:
        hr("5) pi05 PROCESSOR DRY-RUN (deploy-as-is check)")
        print("  NOTE: this tests if pi05_base's processor accepts the frame AS-IS (inference).")
        print("  Finetune builds its OWN processor from the dataset, so a camera-name mismatch")
        print("  here is expected and not a finetune blocker.")
        try:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            policy = PI05Policy.from_pretrained(args.pi05_ckpt)
            pre, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.pi05_ckpt)
            ds = LeRobotDataset(args.repo_id or ds_root.name, root=str(ds_root))
            out = pre(dict(ds[0]))
            print("  processor accepted the frame. keys:", list(out)[:8], "...")
        except Exception as e:
            print(f"  processor dry-run failed: {type(e).__name__}: {e}")
            print("  (likely camera-name mismatch vs pi05_base; fine — finetune remaps names.)")

    print("\nDONE. Paste the whole output back.")


if __name__ == "__main__":
    main()
