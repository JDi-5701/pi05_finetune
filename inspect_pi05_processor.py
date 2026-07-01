#!/usr/bin/env python3
"""One-shot inspector for the lerobot pi05_base processor pipeline.

Run on the machine that has the `ros_ml` env + HF cache:
    python inspect_pi05_processor.py

It answers, with zero extra typing:
  1) versions (lerobot / transformers)
  2) where pi05_base's processor JSONs live (HF cache)
  3) the FULL pre- and post-processor step lists, with each step's `enabled` flag
  4) the exact diff between policy_preprocessor.json and its .bak
     (= the only real change you made vs the official published checkpoint)
  5) which lerobot source files define the processor steps (for deeper reading)

Paste the whole output back.
"""
from __future__ import annotations

import difflib
import glob
import importlib
import json
import os
from pathlib import Path


def hr(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- 1) versions
hr("1) VERSIONS")
for mod in ("lerobot", "transformers", "torch"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:12s} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"  {mod:12s} <import failed: {e}>")


# ---------------------------------------------------------------- locate cache
hr("2) LOCATE pi05_base PROCESSOR JSONs (HF cache)")
cache_roots = [
    Path.home() / ".cache/huggingface/hub",
    Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
]
found = {}
for root in filter(None, cache_roots):
    base = root / "models--lerobot--pi05_base"
    if not base.exists():
        continue
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        hits = glob.glob(str(base / "snapshots" / "*" / name))
        if hits and name not in found:
            found[name] = hits[0]
if not found:
    print("  !! could not find pi05_base processor JSONs under any HF cache root.")
    print("     checked:", [str(r) for r in filter(None, cache_roots)])
for k, v in found.items():
    print(f"  {k:28s} -> {v}")


# ---------------------------------------------------------------- step lister
def step_name(step):
    if not isinstance(step, dict):
        return str(step)
    for key in ("class", "type", "registry_name", "name", "_target_"):
        if key in step:
            return step[key]
    return next(iter(step), "<?>")


def find_enabled(step):
    """Search a step dict (one level deep) for any 'enabled' flag."""
    if not isinstance(step, dict):
        return None
    if "enabled" in step:
        return step["enabled"]
    for v in step.values():
        if isinstance(v, dict) and "enabled" in v:
            return v["enabled"]
    return None


def steps_of(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return d
    for key in ("steps", "pipeline", "processors"):
        if key in d and isinstance(d[key], list):
            return d[key]
    return [d]  # fallback: treat the whole thing as one


hr("3) PROCESSOR STEP LISTS (name + enabled flag)")
for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    if name not in found:
        continue
    print(f"\n  --- {name} ---")
    try:
        for i, s in enumerate(steps_of(found[name])):
            print(f"   {i:2d}  {str(step_name(s)):45s} enabled={find_enabled(s)}")
    except Exception as e:  # noqa: BLE001
        print(f"   <could not parse: {e}>")


# ---------------------------------------------------------------- diff vs .bak
hr("4) DIFF  policy_preprocessor.json.bak (official)  ->  current (yours)")
pre = found.get("policy_preprocessor.json")
if pre and Path(pre + ".bak").exists():
    a = open(pre + ".bak").read().splitlines()
    b = open(pre).read().splitlines()
    diff = list(difflib.unified_diff(a, b, "official(.bak)", "current", lineterm=""))
    if diff:
        print("\n".join(diff))
    else:
        print("  (no textual difference)")
elif pre:
    print(f"  no .bak next to {pre}  -> cannot show what was changed.")
    print("  (printing the full current JSON instead, so we can read it)")
    print(json.dumps(json.load(open(pre)), indent=2, ensure_ascii=False))
else:
    print("  preprocessor json not found; skipped.")


# ---------------------------------------------------------------- source files
hr("5) LeROBOT SOURCE that defines the processor steps")
try:
    import lerobot.policies.pi05 as p
    pi05_dir = Path(p.__file__).parent
    print(f"  pi05 dir: {pi05_dir}")
    for f in sorted(pi05_dir.glob("*.py")):
        print(f"    {f.name}")
    lerobot_root = pi05_dir.parents[1]
    print(f"\n  grep for step definitions under {lerobot_root}:")
    needles = ("relative_action", "make_pre_post", "ProcessorStep",
               "Normalize", "Tokeniz", "pad")
    hits = {}
    for py in lerobot_root.rglob("*.py"):
        try:
            txt = py.read_text(errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        for n in needles:
            if n.lower() in txt.lower():
                hits.setdefault(n, set()).add(str(py.relative_to(lerobot_root)))
    for n in needles:
        files = sorted(hits.get(n, []))[:6]
        print(f"    [{n}] -> " + (", ".join(files) if files else "(none)"))
except Exception as e:  # noqa: BLE001
    print(f"  <could not locate source: {e}>")

print("\nDONE. Paste everything above back.")
