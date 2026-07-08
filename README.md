# pi05_finetune

当前默认流程是把 Franka 录制出来的原始 LeRobot 数据处理成 pi0.5 使用的
7 维 axis-angle delta action 数据，然后用 `train_pi05.py` 微调，最后用
`infer_aa_delta.py` 做离线推理 debug。

所有命令默认在 `ros_ml` 环境里运行：

```bash
rosml
cd ~/ros_ml_ws/pi05_finetune
```

## 1. 数据处理

第一步先修正原始数据集里的 `action`。录制时保存的 action 是 target pose，但实际训练里我们希望
`action[t]` 对应下一帧真实到达的 EE pose，所以先用 `fix_raw_action_from_next_state.py`：

```bash
python fix_raw_action_from_next_state.py \
  --root ./datasets/raw_pick_up_sponge_42 \
  --out ./datasets/raw_pick_up_sponge_42_next_state \
  --overwrite
```

这个脚本会把每一帧的 raw action 改成：

```text
action[t] = observation.state[t + 1] 里提取出来的 EE pose + gripper
```

最后一帧没有 `t + 1`，就使用本帧自己的 state。除了 `action` 和 `meta/stats.json` 里的 action 统计，其他数据不改。

第二步用 `convert_to_pi05v2.py` 转成 pi0.5 训练格式：

```bash
rm -rf ./datasets/sponge_pi05_aa_delta_42_next_state_v2

python convert_to_pi05v2.py \
  --root ./datasets/raw_pick_up_sponge_42_next_state \
  --out ./datasets/sponge_pi05_aa_delta_42_next_state_v2 \
  --repo-id sponge_pi05_aa_delta_42_next_state_v2 \
  --rot aa \
  --action-mode delta \
  --chunk-size 50
```

转换后的数据格式是：

```text
observation.state = [x, y, z, rx, ry, rz, gripper]
action            = [dx, dy, dz, drx, dry, drz, gripper]
```

也就是 7 维：

```text
3 维位置 + 3 维 axis-angle/rotvec + 1 维 gripper
```

这里的 `action` 存在数据集里时是 same-frame delta，也就是 target 相对同一帧 `observation.state` 的 delta。
训练时 `train_pi05.py` 会再动态改成 chunk-start delta，让一个 50-step action chunk 里的每一步都相对 chunk 第一帧的 state。

## 2. 训练

默认训练命令：

```bash
python train_pi05.py \
  --root ./datasets/sponge_pi05_aa_delta_42_next_state_v2 \
  --repo-id sponge_pi05_aa_delta_42_next_state_v2 \
  --output-dir ./outputs/pi05_aa_delta_42_next_state_v2 \
  --steps 2000 \
  --batch-size 8 \
  --log-every 50 \
  --save-every 500 \
  --num-workers 4 \
  --action-delta-reference chunk-start \
  --input-action-format same-frame-delta
```

如果要开 W&B：

```bash
python train_pi05.py \
  --root ./datasets/sponge_pi05_aa_delta_42_next_state_v2 \
  --repo-id sponge_pi05_aa_delta_42_next_state_v2 \
  --output-dir ./outputs/pi05_aa_delta_42_next_state_v2 \
  --steps 2000 \
  --batch-size 8 \
  --log-every 50 \
  --save-every 500 \
  --num-workers 4 \
  --action-delta-reference chunk-start \
  --input-action-format same-frame-delta \
  --wandb \
  --wandb-project pi05_finetune \
  --wandb-name pi05_aa_delta_42_next_state_v2
```

训练脚本里比较重要的两个参数是：

```text
--action-delta-reference chunk-start
--input-action-format same-frame-delta
```

含义是：数据集里存的是每一帧自己的 same-frame delta，但 pi0.5 训练用的是 50-step action chunk。
所以训练时会把每个 chunk 的所有动作重新编码成相对 chunk 起点 state 的 delta，并且用同一套 chunk-start
label 重新计算 action normalization stats。

checkpoint 会保存到：

```text
./outputs/pi05_aa_delta_42_next_state_v2/final
./outputs/pi05_aa_delta_42_next_state_v2/step_<N>
```

## 3. infer_aa_delta 运行和 debug 原理

运行命令：

```bash
python infer_aa_delta.py \
  --ckpt ./outputs/pi05_aa_delta_42_next_state_v2/final \
  --root ./datasets/sponge_pi05_aa_delta_42_next_state_v2 \
  --repo-id sponge_pi05_aa_delta_42_next_state_v2 \
  --episode-index 0 \
  --chunk-stride 30 \
  --take-steps 30 \
  --rollout-from-chunk-start \
  --plot-out ./outputs/episode0_aa_delta_debug.png \
  --device cuda
```

如果想看某个中间 checkpoint：

```bash
python infer_aa_delta.py \
  --ckpt ./outputs/pi05_aa_delta_42_next_state_v2/step_2000 \
  --root ./datasets/sponge_pi05_aa_delta_42_next_state_v2 \
  --repo-id sponge_pi05_aa_delta_42_next_state_v2 \
  --episode-index 0 \
  --chunk-stride 30 \
  --take-steps 30 \
  --rollout-from-chunk-start \
  --plot-out ./outputs/episode0_step2000_aa_delta_debug.png \
  --device cuda
```

debug 原理：

`infer_aa_delta.py` 会加载训练保存的 model 和 processor，然后从转换后的数据集里读取真实图像和真实
`observation.state`，调用：

```python
policy.predict_action_chunk(batch)
```

模型输出的是一个 50-step 的 7D delta action chunk：

```text
[dx, dy, dz, drx, dry, drz, gripper]
```

脚本会用 checkpoint 里的 postprocessor 把模型输出反归一化，然后把预测的 aa-delta 还原成 absolute EE pose：

```text
position_pred = chunk_start_position + dxyz
rotation_pred = chunk_start_rotation * delta_rotation
```

同时，它也会把数据集里的 ground-truth delta action 还原成 absolute pose。最后画图对比：

```text
pred absolute pose vs ground-truth absolute pose
pred delta action  vs ground-truth delta action
```

`--chunk-stride 30 --take-steps 30` 的意思是：模型每次预测 50 步，但只拿前 30 步做对比；
下一段从新的真实 chunk 起点 state 重新开始。这个 debug 不是闭环控制，只是在真实数据帧上检查：

```text
同一个观测输入下，模型预测的 action chunk 是否接近数据集里的 action chunk
```

如果图里 delta 和 absolute pose 都能跟住 ground truth，说明训练标签、processor、checkpoint 加载和
aa-delta 反解逻辑大体是对的。反之，如果 loss 很低但 infer 图完全不对，优先查 action 表示、
normalization stats、checkpoint 路径和 `repo-id/root` 是否匹配。
