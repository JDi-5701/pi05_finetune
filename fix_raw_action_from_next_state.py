#!/usr/bin/env python3
"""Rewrite raw Franka LeRobot actions from the next observation.state.

The recorder action can be noisy or semantically off.  This utility rewrites:

    action[t] = [
        state[t+1].x,
        state[t+1].y,
        state[t+1].z,
        state[t+1].qx,
        state[t+1].qy,
        state[t+1].qz,
        state[t+1].qw,
        normalized_gripper(state[t+1]),
    ]

For the last frame of each episode there is no t+1, so by default it uses that
episode's final state.  No frames are dropped and all non-action columns stay
unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


GRIPPER_MAX_WIDTH = 0.04


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Set raw LeRobot action[t] to the next frame's observation.state-derived EE target."
    )
    ap.add_argument("--root", required=True, help="source dataset directory, e.g. datasets/raw_pick_up_sponge")
    ap.add_argument(
        "--out",
        default=None,
        help="output dataset directory. If omitted, you must pass --in-place.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="modify --root directly. Without this, the script writes to --out.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="allow deleting an existing --out directory before copying.",
    )
    ap.add_argument(
        "--gripper-max-width",
        type=float,
        default=GRIPPER_MAX_WIDTH,
        help="per-finger open width in meters used to normalize state gripper to [0,1].",
    )
    ap.add_argument(
        "--no-update-stats",
        action="store_true",
        help="do not update meta/stats.json action statistics.",
    )
    return ap.parse_args()


def action_from_state(state: np.ndarray, gripper_max_width: float) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[0] < 15:
        raise ValueError(f"expected raw observation.state dim >= 15, got {state.shape}")

    action = np.empty(8, dtype=np.float32)
    action[:7] = state[:7]
    gripper = float(np.mean(state[13:15]) / gripper_max_width)
    action[7] = np.clip(gripper, 0.0, 1.0)
    return action


def rewrite_actions_in_df(df: pd.DataFrame, gripper_max_width: float) -> tuple[pd.DataFrame, int]:
    required = {"observation.state", "action", "episode_index", "frame_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing parquet columns: {sorted(missing)}")

    out = df.copy()
    order = out.sort_values(["episode_index", "frame_index"]).index
    new_actions: dict[int, np.ndarray] = {}

    for _, ep in out.loc[order].groupby("episode_index", sort=False):
        indices = ep.index.to_list()
        states = ep["observation.state"].to_list()
        for i, row_index in enumerate(indices):
            next_state = states[i + 1] if i + 1 < len(states) else states[i]
            new_actions[row_index] = action_from_state(next_state, gripper_max_width)

    out["action"] = [new_actions[idx] for idx in out.index]
    return out, len(new_actions)


def stats_for_matrix(values: np.ndarray) -> dict[str, list]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def update_action_stats(root: Path, action_values: list[np.ndarray]) -> None:
    stats_path = root / "meta" / "stats.json"
    if not stats_path.exists():
        return

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    stats["action"] = stats_for_matrix(np.stack(action_values, axis=0))
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def prepare_output(root: Path, out: Path | None, in_place: bool, overwrite: bool) -> Path:
    if in_place:
        if out is not None and out.resolve() != root.resolve():
            raise SystemExit("--in-place cannot be combined with a different --out")
        return root

    if out is None:
        raise SystemExit("pass --out for a copied dataset, or pass --in-place to modify --root directly")

    if out.exists():
        if not overwrite:
            raise SystemExit(f"output already exists: {out}\nUse --overwrite or choose a new --out.")
        shutil.rmtree(out)

    shutil.copytree(root, out)
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else None

    if not root.exists():
        raise SystemExit(f"source dataset does not exist: {root}")

    target = prepare_output(root, out, args.in_place, args.overwrite)
    parquet_files = sorted((target / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise SystemExit(f"no data parquet files found under {target / 'data'}")

    all_actions: list[np.ndarray] = []
    total = 0
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        df, n = rewrite_actions_in_df(df, args.gripper_max_width)
        df.to_parquet(parquet_path, index=False)
        all_actions.extend(np.asarray(x, dtype=np.float32) for x in df["action"])
        total += n
        rel = parquet_path.relative_to(target)
        print(f"rewrote {n:6d} actions in {rel}")

    if not args.no_update_stats:
        update_action_stats(target, all_actions)
        print("updated meta/stats.json action statistics")

    print(f"done: {target}")
    print("action[t] now uses observation.state[t+1]; last frame of each episode uses its own state.")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    main()
