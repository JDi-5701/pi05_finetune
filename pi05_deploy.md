# pi05 Franka Finetune + ROS2 Deploy — Agent Memory

Last updated: 2026-06-28 (离线部署 dry-run 验证通过 ✅ — 见「离线部署 dry-run」段)

## 终极目标 (Ultimate Goal)
在**自己的 Franka 机器人**上：
1. 用自己采集的数据 finetune pi0.5 (pi05)
2. 把 finetuned 模型部署到 **ROS2**：订阅传感器话题 → 推理 → 发布动作话题
3. 有能力**改 pi05 架构做研究**

关键事实：**用户的机器人 setup == DROID**（同款 Franka + 相机布局）。
所以 DROID 数据不是"替身"，而是和真实机器人**同构**的数据。

## 当前状态：冒烟测试 ✅ 通过
200 步 finetune 跑通，无报错无 OOM，loss 正常下降 (0.396 → 0.306)，checkpoint 已生成。
**整条链路验证完成**：真实 Franka 数据(DROID) → 自己的转换脚本 → LeRobot v3.0
→ pi05_base finetune → 收敛 + 存 checkpoint。

## ★ 最终可行方案（FINAL WORKING RECIPE，照这个做）★
1. 数据：DROID raw → LeRobot v3.0，相机名用 base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb，
   state/action = 8 维 (7关节+1夹爪)
2. **删掉 pi05_base 预存 processor 里过时的 `relative_actions_processor` 步骤**
   （enabled=false 但 registry 不认，会报错）。文件：
   ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/
   7de663972b7817d2c4cf2d84c821153dfea772e9/policy_preprocessor.json (有 .bak)
3. 用**官方 lerobot-train 命令行**（不要自己写脚本绕过 processor！），加：
   - `--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'`
   - `--policy.train_expert_only=true`（省显存，只训 action expert 693M/4B）
   - `--policy.gradient_checkpointing=true --policy.dtype=bfloat16`
   - `--policy.push_to_hub=false`
4. 模型保持 32 维，官方 processor 自动把 8 维 pad 到 32。

### 跑通的完整训练命令（一行）
```
cd ~/ros_ml_ws/src/pi05 && lerobot-train --dataset.repo_id=local/droid_smoke --dataset.root=/home/prs/ros_ml_ws/src/pi05/datasets/droid_lerobot --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base --policy.push_to_hub=false --policy.device=cuda --policy.dtype=bfloat16 --policy.train_expert_only=true --policy.gradient_checkpointing=true --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' --batch_size=8 --steps=200 --save_freq=100 --log_freq=20 --output_dir=/home/prs/ros_ml_ws/src/pi05/models/droid_smoke --wandb.enable=false
```
真正训练时把 --steps 调大（3000~20000），可去掉 train_expert_only 做全量（更吃显存）。

## ★ 部署加载路径（DEPLOY）★
finetuned 模型在：
```
/home/prs/ros_ml_ws/src/pi05/models/droid_smoke/checkpoints/last/pretrained_model
```
这个文件夹含 config + model.safetensors + pre/post processor（含归一化 safetensors）。
部署用 PI05Policy.from_pretrained(<上面路径>) + make_pre_post_processors(从同路径加载)，
processor 会自动处理 pad 8→32、token化、反归一化、取前8维。**不要手动 pad**。
checkpoint 目录：000100/ 000200/ last/，每个含 pretrained_model/ + training_state/。

## ★ 离线部署 dry-run 验证 ✅（2026-06-28，整条推理链路跑通）★
脚本 `scripts/deploy_dryrun.py`：加载 finetuned ckpt → 从 DROID 数据集取真实帧 → 过同一套
processor → 推理 → 打印 **预测动作 vs 真值**。**全绿，预测贴着真值。**

**已确认的 deploy API（照抄即可，0.5.1 实测）**：
```python
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

policy = PI05Policy.from_pretrained(CKPT).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT)  # ←这个调用法对
batch = pre(obs_dict)                       # obs_dict = 数据集样本(或真实观测), processor 自动 batch/pad/归一化/tokenize
chunk = policy.predict_action_chunk(batch)  # -> (50, 8) 已切到 8 维; 或 select_action(batch)->单步
# post(chunk) 可选(反归一化已在内部); select_action/predict_action_chunk 签名都是 (batch: dict)->Tensor
```
**关键事实**：
- config: `chunk_size=n_action_steps=50`, `n_obs_steps=1`, `max_action_dim=max_state_dim=32`。
- input_features = 3 相机 `observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}` + `observation.state`(8)。
- 数据集样本(`ds[i]`)：3 相机 (3,224,224) float[0,1]、state(8)、action(8)、`task`=语言字符串。
- 输出 **(50,8) 动作块**(已是真实关节位置空间)。**部署:推理一次→执行前 K 个→再重规划,别每周期都推理**。
- **结果**(在训练帧上、200步冒烟模型):mean|pred−gt| ≈ 0.12–0.21 rad/关节(~7–12°),模型确在跟演示;
  **夹爪那维最弱**(欠训正常)。⚠️ 这是训练数据上的 sanity check,证明「pipeline+学习 OK」,非泛化质量。
- 另有 `scripts/inspect_pi05_processor.py`:打印 processor 步骤/版本/diff,排查 processor 用。

**下次升级 dry-run**:改成在**留出 episode** 上、画整条轨迹的逐关节 pred-vs-GT 曲线 → 才是真质量评估。

## 环境 (已就绪)
- RTX 5090 (32GB, Blackwell sm_120), 远程 SSH
- conda env `ros_ml`: RoboStack ROS2 + LeRobot 0.5.1 同居，数据采集已跑通
- PyTorch 2.10.0+cu128, CUDA 12.8, transformers **5.3.0**（精确要求）
- HF 已登录(持久), PaliGemma gated repo 已授权(token 要 Read 权限不是 fineGrained)

### 环境坑（不影响训练）
- nvidia-smi 报 NVML mismatch：当前内核 6.17.0-29，驱动为 6.17.0-35 建的，但 35 显示器坏。
  训练不受影响。看显存用 `python -c "import torch; print(torch.cuda.max_memory_allocated()/1e9)"`。
  **远程别重启**（GRUB 默认=坏内核35）。等能物理接触机器再修。

## 目录结构
```
~/ros_ml_ws/src/pi05/
├── datasets/
│   ├── droid_raw/2023-12-04/        # 30 条 DROID 原始演示 (1.6GB)
│   ├── droid_raw/aggregated-annotations-030724.json
│   └── droid_lerobot/               # 转换后 LeRobot v3.0 (repo_id=local/droid_smoke, 30ep/7756frames)
├── models/droid_smoke/checkpoints/  # 训练输出 (last/pretrained_model = 部署用)
└── scripts/
    ├── convert_droid.py             # DROID raw -> LeRobot v3.0 (可改去处理自己的数据)
    └── train_pi05.py                # 自定义训练脚本(已弃用，绕过processor有bug，但改架构时可参考)
```

## pi05 维度机制（核心理解）
- 跨本体基础模型，固定 32 维标准容器装 state/action (max_state_dim=max_action_dim=32)
- Franka=8维(7关节+1夹爪)，pad 到 32：前8维真实，后24维补0。训练和部署位置必须一致。
- token化：图像→SigLIP每图256(3相机=768)；语言→tokenizer pad到200；state(32)→投影成1token
- 三路拼 prefix 进 PaliGemma VLM，action expert 输出32维，取前8维=机器人动作

## DROID 数据格式（= 自己 Franka 数据该有的格式）
- state = joint_positions(7) + gripper_position(1) = 8  [obs/robot_state/...]
- action = joint_position(7) + gripper_position(1) = 8  [action/..., joint position 动作空间]
- 相机名必须用：base_0_rgb(←ext1 23404442) / left_wrist_0_rgb(←wrist 19824535) / right_wrist_0_rgb(←ext2 29838012)
- 图像：DROID mp4 1280x720 双目并排，取左半缩到 224x224；视频比轨迹少1帧(脚本补齐)
- 语言：按 metadata uuid 去 annotations json 查 language_instruction1

## 踩过的坑 + 解法
1. transformers 版本 → 装 transformers==5.3.0 (lerobot[pi] 自带)
2. push_to_hub 报 repo_id missing → --policy.push_to_hub=false
3. IPEC droid_lerobot 是 v2.0 太旧 → 不能用，下原始 DROID 自己转
4. PaliGemma gated → 网页授权 + hf auth login (token Read 权限，fineGrained 会 403)
5. 相机名不匹配 → base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb
6. 8 vs 32 维 mismatch (LeRobot bug #2963) → 保持模型32维 + 官方processor自动pad
7. 1018 vs 976 mask 报错 → 是自己写脚本绕过 processor 漏了语言token padding。
   **教训：别绕过官方 processor，用 lerobot-train。**
8. relative_actions_processor not found → 从 policy_preprocessor.json 删掉该步骤

## 下一步 (TODO)
1. [x] 冒烟测试跑通 (200步)
1.5 [x] 离线部署 dry-run 验证(加载+processor+推理全通, pred 贴 GT) — scripts/deploy_dryrun.py
2. [ ] 真正 finetune：--steps 3000~20000，可考虑全量(去 train_expert_only)
   → 训完用 deploy_dryrun.py 在留出 episode 上重测(改成逐关节轨迹曲线)看真实质量
3. [ ] **写 ROS2 部署 node**：
       - 加载 models/droid_smoke/checkpoints/last/pretrained_model
       - PI05Policy.from_pretrained + make_pre_post_processors(同路径)
       - 订阅：3 相机 Image + joint_states；预处理走 processor（自动 pad/归一化）
       - observation key 名必须和训练一致 (observation.images.base_0_rgb 等)
       - 推理：policy.select_action / predict_action_chunk
       - action chunking：pi05 chunk_size=50，推理一次执行一个 chunk，别每周期都推理
       - 发布：动作话题（取前8维=7关节+夹爪）
4. [ ] 用自己采集的 Franka 数据替换 DROID（改 convert_droid.py 读取部分，格式同上）
5. [ ] (研究) 改架构：复制 modeling_pi05.py 改写 PI05Policy/forward，回自定义训练脚本

## 速查
- 看显存: python -c "import torch; print(torch.cuda.max_memory_allocated()/1e9,'GB')"
- pi05 支持的 finetune: train_expert_only / freeze_vision_encoder / 全量 / LoRA(use_peft) /
  相对动作(use_relative_actions) / 归一化(QUANTILES 或 MEAN_STD)
- 模型代码: site-packages/lerobot/policies/pi05/modeling_pi05.py (forward 在 policy.forward)