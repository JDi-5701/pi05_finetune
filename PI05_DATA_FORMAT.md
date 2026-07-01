# PI0.5 Data Format — Requirements & Compliance Spec

Field-level reference for what `franka_data_recorder` must write so the resulting dataset
fine-tunes **π0.5 (pi05)** out of the box, on **both** implementations:

- **openpi** — Physical Intelligence, <https://github.com/Physical-Intelligence/openpi>
- **HF LeRobot** — the `lerobot` library's own PyTorch port (`PI05Policy`)

It also pins down the **LeRobot dataset on-disk format** (v2.1 vs v3.0) that both consume.

> Researched against upstream `main` (openpi @ 2026-06, lerobot @ 0.5.x/`main`). These repos move
> fast; §7 lists every claim that could not be byte-verified. Re-check before a big data campaign.

---

## 0. TL;DR for this recorder

Our GPU env runs **lerobot 0.5.1 → dataset codebase_version `v3.0`**. What that means concretely:

| Requirement | pi0.5 wants | This recorder | Status |
|---|---|---|---|
| Image dtype in dataset | video (mp4), decoded to `uint8`→float internally | `dtype: video`, `use_videos=True` | ✅ |
| Image pixel format fed to `add_frame` | `uint8`, **HWC**, **RGB**, 0–255 | `image_rgb` extractor → `uint8` HWC RGB | ✅ |
| Image resolution | **any** (model resizes to 224×224 w/ pad) | 1280×720 (even H&W ✔) | ✅ |
| State key | **exactly** `observation.state` | `observation.state` | ✅ |
| Action key | **exactly** `action` | `action` | ✅ |
| State/action dim | ≤ 32 (auto zero-padded to 32) | state 15, action 8 | ✅ |
| Per-frame `task` string | required every frame | recorder injects `task` | ✅ |
| Normalization stats | v3.0 computes mean/std **+ q01..q99** | lerobot computes on `save_episode()` | ✅ (v3.0) |
| **Flush buffered shards** | v3.0 needs **`finalize()`** at end | writer has **no `finalize()`** | ⚠️ **GAP — see §6.1** |
| EE-pose action rep | quaternion is poor for VLA regression | `target_pose` = 7D quaternion | ⚠️ convert at train time (§6.2) |
| openpi cartesian EE policy | none shipped | — | ⚠️ needs custom Inputs/Outputs (§4.6) |

**Bottom line:** the image path is fully compliant. Two things to handle: (1) add a `finalize()`
flush for v3.0, (2) decide the action representation at fine-tune time (quaternion → 6D/euler).

---

## 1. LeRobot dataset format — the container both models consume

### 1.1 Version ↔ codebase_version

| lerobot pip | `codebase_version` | package path |
|---|---|---|
| 0.1.0 – 0.3.3 | **v2.1** | `lerobot/common/datasets/` |
| **0.4.0 – 0.5.x** | **v3.0** | `src/lerobot/datasets/` |

Boundary is **lerobot 0.4.0** (Oct 2025). **Our 0.5.1 → v3.0.**
Source: <https://huggingface.co/blog/lerobot-datasets-v3>

### 1.2 On-disk layout

**v2.1 — one file per episode:**
```
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl                                   # per-episode stats
data/chunk-000/episode_000000.parquet                       # %03d chunk / %06d episode
videos/chunk-000/observation.images.<cam>/episode_000000.mp4
```

**v3.0 — many episodes per file (what we produce):**
```
meta/info.json
meta/stats.json                                             # global aggregate
meta/tasks.parquet                                          # NOT tasks.jsonl (docs are stale)
meta/episodes/chunk-000/file-000.parquet                    # per-episode meta + per-episode stats
data/chunk-000/file-000.parquet                             # MANY episodes concatenated
videos/observation.images.<cam>/chunk-000/file-000.mp4      # {video_key} FRONT; concatenated
```
Path templates: `chunk-{chunk_index:03d}/file-{file_index:03d}` (hyphen). Data/video shards roll
over at `data_files_size_in_mb` (100) / `video_files_size_in_mb` (200 default; big datasets use 500).
Source: `src/lerobot/datasets/utils.py`.

### 1.3 `meta/info.json` (real v3.0 example, `lerobot/pusht`@main)
```json
{
  "codebase_version": "v3.0",
  "robot_type": "franka_fr3",
  "total_episodes": 206, "total_frames": 25650, "total_tasks": 1,
  "chunks_size": 1000, "fps": 30,
  "splits": {"train": "0:206"},
  "data_path":  "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": { ... },
  "data_files_size_in_mb": 100, "video_files_size_in_mb": 500
}
```
**v2.1→v3.0 info.json diff:** removed `total_chunks`, `total_videos`; added `data_files_size_in_mb`,
`video_files_size_in_mb`, and a per-feature `fps`; changed the two path templates.
`splits` values are **string ranges** (`"0:206"`), not lists. `robot_type` may be `"unknown"`/`null`.

### 1.4 `features` dict schema

Every feature = `{"dtype", "shape", "names"}`; video adds `"info"` (v3.0) / `"video_info"` (v2.1).

**Low-dim vector:**
```json
"observation.state": {"dtype":"float32","shape":[15],
  "names":["x","y","z","qx","qy","qz","qw","fx","fy","fz","tx","ty","tz","gripper_0","gripper_1"]}
```
`shape` is a 1-tuple `[dim]`; `names` = ordered channel labels (our writer fills these from
`EXTRACTOR_DIM_NAMES`).

**Image/video feature:**
```json
"observation.images.base": {"dtype":"video","shape":[720,1280,3],
  "names":["height","width","channels"],
  "info":{"video.codec":"av1","video.pix_fmt":"yuv420p","video.fps":30,"video.channels":3, ...}}
```
- **Shape order is `[H, W, C]` (HWC).** Proven in source: routing checks `shape[2] in (1,3)`, and
  `dataset_to_policy_features` transposes `(h,w,c)→(c,h,w)` for the policy. **Datasets store HWC;
  policies receive CHW.**
- `names[2]` may be `"channel"` (real v2.1 data) or `"channels"` (current `main`) — both accepted;
  **emit `"channels"` for v3.0** (our writer does).
- **Allowed dtypes:** `"image"`, `"video"`, `"string"`/`"language"`, or any NumPy dtype string
  (`float32`, `float64`, `int64`, `int32`, `bool`, …). No numeric whitelist.

**DEFAULT_FEATURES — auto-added by LeRobot; must NOT appear in your `create()` features nor frames:**
`timestamp`(f32), `frame_index`(i64), `episode_index`(i64), `index`(i64), `task_index`(i64).

### 1.5 Image vs video storage

Routing is **by `dtype`, not key name**. `dtype:"video"` → mp4 (not a parquet column);
`dtype:"image"` → PNG under `images/` (location partly ambiguous — **prefer `video`**).
- Codec default (both versions): **AV1 via `libsvtav1`**, `pix_fmt yuv420p`, GOP `-g 2`, `-crf 30`,
  input fps = dataset fps. v3.0 encodes via **PyAV** (`src/lerobot/configs/video.py`
  `RGBEncoderConfig`); override with `rgb_encoder=RGBEncoderConfig(vcodec="h264", ...)`.
- **`yuv420p` requires even width AND height.** 1280×720 ✔. No divisibility-by-16 rule in LeRobot
  itself (only whatever the codec imposes).
- **Deploy machine's PyAV/ffmpeg must have SVT-AV1**, or encoding fails — verify on `prs`.

### 1.6 `add_frame` pixel expectation

Canonical: numpy **`uint8`, `(H,W,3)`, RGB, 0–255**. Tolerated: `(C,H,W)`; PIL; torch tensors
(auto→numpy); float in 0.0–1.0 (scaled `*255`). Our `image_rgb` extractor already yields the
canonical form.

### 1.7 Per-frame parquet columns

One row/frame; one column per **non-video** feature + DEFAULT_FEATURES. Video is NOT a column.
Semantics: `frame_index` resets per episode; `index` is global; `timestamp = frame_index/fps`
(per-episode relative); `task_index` FKs into tasks; `next.done` true on last frame.
v3.0 additionally stores per-episode rows in `meta/episodes/*.parquet` with global row ranges
(`dataset_from_index/to_index`) and per-camera video `from/to_timestamp` offsets into the shared mp4.

### 1.8 Statistics — the bit pi0.5 cares about

- **v2.1:** `meta/episodes_stats.jsonl`, keys **min/max/mean/std/count only — NO quantiles**.
- **v3.0:** `compute_stats.py` adds `DEFAULT_QUANTILES=[0.01,0.10,0.50,0.90,0.99]` →
  keys **`q01,q10,q50,q90,q99`** on top of min/max/mean/std/count. Per-episode stats live inside
  `meta/episodes/*.parquet`; `meta/stats.json` is the global aggregate.
- **This directly enables pi0.5's quantile normalization** (see §3/§4) — but only on v3.0.
- Stat shapes: images per-channel `(C,1,1)`, **RGB normalized to [0,1] (÷255)**; low-dim `[N]`.
  Image stats are sampled (`max(100, min(len^0.75, 10000))` frames), not full.

### 1.9 Tasks / language

Every frame row carries `task_index`; you pass `"task": "<string>"` inside the frame dict and
LeRobot resolves/creates the index. v2.1 `meta/tasks.jsonl` `{"task_index":0,"task":"..."}`;
v3.0 `meta/tasks.parquet` (same concept). Per-episode `tasks` list also stored in episodes meta.

### 1.10 Python API (v3.0, lerobot 0.5.x)

```python
LeRobotDataset.create(repo_id, fps, features, root=None, robot_type=None,
                      use_videos=True, tolerance_s=1e-4,
                      image_writer_processes=0, image_writer_threads=0,
                      video_backend=None, batch_encoding_size=1, ...)
```
- `robot=` param was **removed** in v3.0 (v2.1 had it). `features` is now required, positional-3.
  Our writer already calls the v3.0-compatible subset.
- `add_frame(frame)` — dict = user features + `"task"`; `"timestamp"` optional. Must NOT include
  `frame_index/episode_index/index/task_index`.
- `save_episode()` — computes per-episode stats (incl. quantiles on v3.0), encodes video, writes.
- **`finalize()` — v3.0 REQUIRES a final `finalize()`** to flush buffered parquet/video shard
  writers. Missing this can leave the last shard(s) unwritten. **← our gap, §6.1.**

---

## 2. Model input contract at a glance (both implementations)

```
 cameras  observation.images.<name>   uint8 HWC RGB (any res)  ─► model resizes 224² + pads, RGB→[-1,1]
 state    observation.state           float32 [d_s≤32]         ─► zero-pad to 32, normalize
 action   action                      float32 [d_a≤32]         ─► zero-pad to 32, normalize, chunked
 task     "task" string per frame     natural language         ─► tokenized (PaliGemma), ~48 tok
```

---

## 2.5 Reference embodiment data layouts (what "pi format" actually is)

pi0.5's base model is trained on a **large multi-embodiment mix** (paper §IV: state = joint
angles + gripper + torso lift + base velocity; action dim 18–19; *"predict target joint and
end-effector poses"*). The three **fine-tunable openpi configs** below are what people mean by
"the pi reference format" — plus our own Franka layout for contrast.

| Dataset (openpi config) | Robot | Cameras (→ model slots) | `state` | `action` | Space / rotation | abs/delta | fps |
|---|---|---|---|---|---|---|---|
| **LIBERO** (`pi05_libero`) | Panda 1-arm | `image`,`wrist_image` 256² → `base_0_rgb`,`left_wrist_0_rgb` | **8**: EE pos(3)+**axis-angle**(3)+gripper qpos(2) | **7**: ΔEE pos(3)+Δ**axis-angle**(3)+gripper(1) | Cartesian EE / axis-angle | **delta** (gripper abs, −1 open/+1 close) | 10 |
| **DROID** (`pi05_droid`) | Franka 1-arm | `ext_1`/`ext_2`/`wrist` 180×320 → `base_0_rgb`/`base_1`/`left_wrist_0_rgb` | **8**: joint(7)+gripper(1) | **8**: joint velocity | **joint space** | — | 15 |
| **ALOHA** (`pi05_aloha`) | bimanual ALOHA | `cam_high`/`cam_left_wrist`/`cam_right_wrist` → 3 slots | **14**: per-arm joints(6)+gripper(1) | **14**: joint positions | **joint space** | optional delta (gripper abs) | ~50 |
| **This recorder** (Franka, absolute-EE) | Franka 1-arm, Cartesian impedance | `observation.images.base` 720×1280 → `base_0_rgb` (2 wrist slots auto-black) | **10**: EE pos(3)+**6D**(6)+gripper(1) | **10**: EE pos(3)+**6D**(6)+gripper(1) | Cartesian EE / 6D | **absolute** | 30 |

**Model side (pi0.5), uniform across embodiments:** state/action **zero-padded to 32**; action
horizon 50 (`pi05_droid`=15, `pi05_libero`=10); **quantile (1%/99%→[−1,1]) normalization**;
3 fixed image slots `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`, internally resized+padded
to **224²**, missing slots black + mask False.

**Key observation:** the three references each use a **different** space (axis-angle EE-delta /
joint velocity / joint position) — **there is no unified format, and none use 6D.** So "copy the
reference format exactly" is a non-goal; our absolute-EE + 6D is a regression-friendly, controller-
native choice (see §4), not a reference-mandated one.

---

## 3. π0.5 in HF LeRobot (`PI05Policy`)

**It exists natively** — `src/lerobot/policies/pi05/` (`PI05Policy`, `PI05Config`,
`type="pi05"`), checkpoints `lerobot/pi05_base`, `lerobot/pi05_libero_base`. (Siblings: `pi0`,
`pi0_fast` [underscore], `smolvla`.)

- **Feature matching** (`dataset_to_policy_features`): a feature is an image **iff its dataset
  dtype is `image`/`video`** (not by name); STATE/ACTION are matched by **exact** key
  `observation.state` / `action`. Camera slots = exactly `config.image_features`; missing declared
  cameras are padded with a `-1` image + `False` mask; **≥1 real camera is mandatory**. No fuzzy
  name matching — rename with `--rename_map` if dataset keys ≠ config keys. `empty_cameras: int`
  injects extra always-missing slots (`observation.images.empty_camera_{i}`, `(3,224,224)`).
- **Images:** dataset stores video → decode layer yields **float32 CHW [0,1]**; the policy then
  scales **[0,1]→[-1,1]** and pad-resizes to **224×224** internally. So **don't pre-resize or
  pre-normalize**. `DEFAULT_IMAGE_SIZE=224`, square enforced.
- **State/action:** `max_state_dim = max_action_dim = 32`, shorter vectors **zero-padded** on the
  last dim, output truncated back. `chunk_size = n_action_steps = 50`, `n_obs_steps = 1`,
  `num_inference_steps = 10`, `tokenizer_max_length = 48`.
- **Normalization** (`normalization_mapping`): images `IDENTITY`; **pi05 uses `QUANTILES`
  (q01/q99) for STATE and ACTION** (openpi-aligned; plain `pi0` uses `MEAN_STD`). Stats come from
  `dataset.meta.stats` — **hence you want a v3.0 dataset (§1.8) so quantiles exist.**
- **Language:** every frame needs a `task` string; policy consumes **pre-tokenized** tensors
  (`observation.language.tokens/attention_mask`) built by a processor step using PaliGemma
  (`google/paligemma-3b-pt-224`), right-pad/truncate to 48.
- **Fine-tune** (console script `lerobot-train`):
  ```bash
  lerobot-train --dataset.repo_id=<you> --policy.type=pi05 \
    --policy.pretrained_path=lerobot/pi05_base \
    --output_dir=./outputs/pi05_franka --job_name=pi05_franka \
    --policy.dtype=bfloat16 --policy.device=cuda \
    --batch_size=32 --steps=30000
    # add --policy.train_expert_only=true to freeze the VLM (lower VRAM)
  ```
- No manual weight conversion: `PI05Policy.from_pretrained("lerobot/pi05_base")`.

---

## 4. π0.5 in openpi (`pi05`)

Config source: `src/openpi/training/config.py`, model in `src/openpi/models/pi0*.py`.

### 4.1 What pi0.5 is (data-relevant)
Flow-matching continuous action expert (NOT FAST discrete tokens). vs pi0, pi0.5 (a) puts robot
**state into the discrete language tokens** (`discrete_state_input`) and (b) uses adaRMSNorm for the
flow timestep. **You do nothing special for this — state stays a plain float vector; discretization
happens inside `TokenizePrompt`.** Defaults: `action_dim=32`, `action_horizon=50` (config overrides:
`pi05_libero=10`, `pi05_droid=15`, `pi05_droid_finetune=16`). Checkpoints: `pi05_base`,
`pi05_libero`, `pi05_droid`, `pi05_aloha`, …

### 4.2 Images
Fixed slots `IMAGE_KEYS = ("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")`, target
`(224,224)`. Feed **uint8 / 0–255 / RGB / HWC**; `Observation.from_dict` converts to float32
[-1,1] internally. Resize (`resize_with_pad`, letterbox) happens in **model_transforms** → **dataset
can be any resolution.** Missing cameras → black `np.zeros_like` + `image_mask=False`.

### 4.3 State
Native dim per robot (DROID=8, ALOHA=14, LIBERO=8); `PadStatesAndActions` **zero-pads to 32**.

### 4.4 Action
`[B, horizon, 32]`; native dim zero-padded to 32; truncated back on output. Normalization applies to
the **whole padded vector incl. gripper and zero-pad dims**.

### 4.5 Normalization
`Normalize` with `use_quantile_norm = model_type != PI0`, i.e. **pi0.5 & pi0-FAST use quantile
(q01/q99)**, pi0 uses mean/std. **You MUST run** `scripts/compute_norm_stats.py --config-name
<your-config>` before training, or it errors ("Normalization stats not found").

### 4.6 Action representation ⚠️
openpi ships **no cartesian-EE policy** — ALOHA/DROID(JOINT_POSITION) are joint-space, LIBERO is
EE-delta. There is no built-in quaternion/euler/6D EE representation, and `delta` baselines are
"subtract current state". **To use our cartesian `target_pose`, you must write custom
`Inputs`/`Outputs` transforms and a `delta` mask in a registered `TrainConfig`.** Gripper stays
absolute continuous everywhere.

### 4.7 LeRobot ingestion
openpi reads a LeRobotDataset via a `RepackTransform` that maps your keys → model schema, e.g.:
```python
# ALOHA-style (dotted keys, like ours)
RepackTransform({"images": {"cam_high": "observation.images.top"},
                 "state": "observation.state", "actions": "action"})
```
Pipeline order: `repack → data_transforms(+Delta) → Normalize(quantile for pi0.5) →
model_transforms(Resize 224 → InjectDefaultPrompt → TokenizePrompt → PadStatesAndActions)`.
Prompt auto-filled from `task` when `DataConfig(prompt_from_task=True)`. Action chunking uses
LeRobot `delta_timestamps = [t/fps for t in range(action_horizon)]` → **set a valid `fps`.**
openpi docs only name LeRobot **v2.0** explicitly; v2.1/v3.0 ingestion is **not documented** — if
you feed openpi directly, verify it reads our v3.0 dataset (or export v2.0 / use the HF path).

### 4.8 Standard workflow
```bash
uv run scripts/compute_norm_stats.py --config-name <your_pi05_config>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <your_pi05_config> --exp-name=franka --overwrite
```
Reuse base norm-stats when fine-tuning from a same-embodiment checkpoint (docs/norm_stats.md).

---

## 5. openpi vs HF-LeRobot — which path?

| | openpi (`pi05`) | HF LeRobot (`PI05Policy`) |
|---|---|---|
| Data input | LeRobotDataset (docs say v2.0) or RLDS | LeRobotDataset **v3.0** (native) |
| Norm stats | separate `compute_norm_stats.py` (quantile) | in `meta.stats` from `save_episode` (v3.0 quantiles) |
| Cartesian-EE policy | none — write custom Inputs/Outputs | works (generic normalized vector), quaternion suboptimal |
| Our dataset (v3.0) | **verify it ingests v3.0** | **native fit** |
| Train cmd | `uv run scripts/train.py` | `lerobot-train --policy.type=pi05` |

**Recommendation for us:** the **HF LeRobot path is the lower-friction fit** — our recorder already
writes v3.0 with quantile stats, keys match, `lerobot/pi05_base` loads directly. Keep openpi as a
secondary target and, if used, confirm v3.0 ingestion or export v2.0.

---

## 6. Compliance & gaps in *this* recorder

Cross-checked against `config/recorder.yaml`, `franka_data_recorder/lerobot_writer.py`,
`extractors.py` (see §0 table for the pass/fail summary).

### 6.1 ⚠️ GAP — v3.0 needs `finalize()` (data-loss risk)

`lerobot_writer.py` implements `create / add_frame / save_episode / discard_episode` but **no
`finalize()`**. On lerobot 0.4.0+ (v3.0), episodes are buffered into shared shards; without a final
`finalize()` the trailing shard(s) may never be written. Add a `close()`:

```python
def close(self):
    """v3.0: flush buffered shard writers. No-op on v2.1 (method absent)."""
    if hasattr(self.ds, "finalize"):
        self.ds.finalize()
```
…and call it from the recorder node's shutdown / `destroy_node` (once, after the last
`save_episode()`). Verify against the installed 0.5.1 API — the method name is `finalize` on `main`.

### 6.2 ⚠️ Action representation

`action` = `target_pose` (7D incl. **quaternion**) + gripper. Quaternion is a poor regression target
for VLAs (double-cover, non-Euclidean). Both stacks will train on it, but **convert to euler or 6D
rotation at fine-tune time** (in the openpi `Inputs` transform, or a lerobot dataset edit). Storing
the raw quaternion is fine — keep the richest form and convert downstream.

### 6.3 ⚠️ openpi cartesian-EE

If targeting openpi, there is no ready cartesian-EE policy — a custom `TrainConfig` with
`Inputs`/`Outputs` + delta mask is required (§4.6). Not needed for the HF-LeRobot path.

### 6.4 ✅ Already compliant
- Image: `dtype video`, `uint8` HWC RGB via `image_rgb`, 1280×720 (even), `use_videos=True`.
- Keys: `observation.state`, `action` (exact), `observation.images.base`.
- Dims: state 15, action 8 — both ≤ 32 (auto-pad).
- `task` string injected per frame; fps 30 valid.
- v3.0 quantile stats produced by `save_episode()`.
- Image `names=["height","width","channels"]` (plural, v3.0-correct).

### 6.5 Open decisions
- **Camera count:** single `observation.images.base` now. pi0.5 checkpoints expect more slots
  (openpi 3; lerobot uses `config.image_features` + `empty_cameras`). Fine to train with one real
  camera; a **wrist camera** would help. At train time map/rename or set `empty_cameras`.
- **SVT-AV1 on `prs`:** confirm PyAV has `libsvtav1`, else set `rgb_encoder` to h264.
- **openpi v3.0 ingestion:** verify or plan a v2.0 export if that path is used.

---

## 7. Flagged / unverified

- openpi's LeRobot ingestion is documented only for **v2.0**; v2.1/v3.0 support not confirmed in docs.
- Exact `pi05` `normalization_mapping` in lerobot (reported `QUANTILES` via docs + openpi parity;
  config not byte-opened).
- Release tag where lerobot `pi05` first shipped (Hub release dates were unreadable).
- Binary `meta/*.parquet` contents derived from source + migration code, not byte-parsed live.
- `dtype:"image"` (PNG) exact storage location not re-confirmed — **use `video`**.
- No explicit resolution-divisibility assert in LeRobot; even-H/W is a `yuv420p` codec requirement.
- pi0.5 minimum episode count / mandatory fps: **not** officially specified.

## 7.5 lerobot pi05 reliability & known bugs (researched ~2026-07)

**Verdict:** the HF-lerobot pi05 port is **actively maintained** (≈monthly releases, staff-driven,
scoped bugs fixed in days) but **genuinely rough in the processor / normalization / image-feature
layer** — the two bugs we hit are *representative, not isolated*. openpi (JAX) is more faithful but
**not easier** (JAX + 70 GB VRAM full-ft / 22.5 GB LoRA + Ubuntu-22.04 + incomplete PyTorch backend).
Stay on lerobot; only switch to openpi for max fidelity at that cost.

**Our two bugs, upstream:**
- **"All image features are missing"** on a custom dataset = the camera keys don't match the base
  model's `input_features` (issue **#3845**, pi0_fast/factory keeps base slots). **Workaround:** name
  the camera to a base slot — `observation.images.base_0_rgb` — which `convert_to_pi05.py` does by default.
- **`relative_actions_processor`** = registry key renamed `delta_actions_processor`→`relative_actions_processor`
  and pi05_base's processor JSON migrated (**PR #3711**, Jun 2026); design flaw still open (**#3863**:
  assumes `state[:action_dim]` aligns with action → can silently corrupt relative targets).
  `train_pi05.py` sidesteps it by **not passing `pretrained_path` to `make_pre_post_processors`**.

**Other live hazards:**
- **Reproducibility gap:** `pi05_libero_base` reports ~**0% zero-shot** on LIBERO vs openpi's 96.85%
  (#2533/#3638); parity only after extra fine-tuning + your own stats.
- **#1 real-robot pain (whole community): normalization.** pi05 needs **QUANTILE** stats computed
  from YOUR dataset — reusing base/LIBERO stats gives erratic behavior. (v3.0 datasets carry quantiles;
  `train_pi05.py` uses `dataset.meta.stats` = fresh, correct.)
- Quick-start dim mismatch #2963 (32 vs 8), transformers-version key mismatches (#1406 fixed, #2179 open).

**Recommendation:** pin lerobot ≥ the #3711 fix (post-v0.5.1 / track main); always compute+use your
dataset's quantile stats; watch #3845 (image features) and #3863 (relative-action alignment). Re-check
open issues before a big training run — this space moves weekly.

## 8. Primary sources
- LeRobot format: `src/lerobot/datasets/{utils,lerobot_dataset,compute_stats,dataset_writer}.py`;
  <https://huggingface.co/blog/lerobot-datasets-v3>; real `lerobot/pusht` @ `v2.1` vs `main`.
- LeRobot pi0/pi05: `src/lerobot/policies/{pi0,pi05}/`, `utils/{constants,feature_utils}.py`,
  `configs/policies.py`, `processor/normalize_processor.py`; <https://huggingface.co/docs/lerobot/pi05>.
- openpi pi0.5: `src/openpi/{models/pi0*.py, transforms.py, training/{config,data_loader}.py,
  policies/*_policy.py}`, `scripts/compute_norm_stats.py`, `examples/libero/convert_*`;
  <https://www.pi.website/blog/pi05>; KI paper arXiv:2505.23705.
