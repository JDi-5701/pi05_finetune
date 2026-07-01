# pi05_finetune

在**自己的 Franka**上,用 `franka_data_recorder` 采集的遥操作数据 **finetune π0.5 (pi05)**,
并最终部署回 ROS2。本仓库是这条链路的**验证 + 转换 + 训练 + 部署**工具集,在 GPU 机 `prs`
的 `ros_ml` conda 环境里运行。仓库同步于 <https://github.com/JDi-5701/pi05_finetune>。

> 详细的部署笔记 / DROID 冒烟记录 / pi05_base processor 坑见 [`pi05_deploy.md`](pi05_deploy.md)。
> 本 README 是入口和操作手册。

---

## 1. 机器 & 环境 (prs)

- 机器:**GPU PC `prs`**(操作端)。数据由 NUC 上的 `franka_data_recorder` 录制,存在
  `~/ros_ml_ws/src/franka_data_recorder/data/<task>_<timestamp>/`。
- 环境:conda **`ros_ml`**(RoboStack ros-jazzy + torch CUDA + **lerobot 0.5.1** + pi05)。
  进环境:`rosml`(或 `conda activate ros_ml`)。
- 数据集格式:lerobot 0.5.1 = **LeRobot v3.0**(含 quantile 统计,pi05 归一化需要)。
- checkpoint:`lerobot/pi05_base` 已缓存在 `~/.cache/huggingface/hub/models--lerobot--pi05_base`。
  离线可用;没缓存时 `HF_HUB_OFFLINE=0 huggingface-cli download lerobot/pi05_base` 下一次。

---

## 2. 全流程

```
[NUC] franka_data_recorder  ──录制──▶  LeRobot v3.0 数据集 (EE 位姿+四元数+力+夹爪, 单相机)
                                          │
[prs] check_pi05_compat.py   ──验证──▶   PASS/WARN/FAIL vs pi05 要求
                                          │
[prs] convert_to_pi05.py     ──转换──▶   pi05 通用格式 (xyz+6D旋转+gripper, 丢力)
                                          │
[prs] train_pi05.py          ──微调──▶   finetuned checkpoint
                                          │
[prs] deploy_dryrun.py       ──验证──▶   预测动作 ≈ 真值  →  写 ROS2 部署节点
```

---

## 3. 脚本清单

| 脚本 | 作用 | 一行用法 |
|---|---|---|
| **`check_pi05_compat.py`** | 只读检查一个数据集是否符合 pi05 微调要求(schema + PASS/WARN/FAIL;`--sample` 加载一帧看真实 shape;`--pi05-ckpt` 试跑 processor) | `python check_pi05_compat.py --root <dataset> --sample` |
| **`convert_to_pi05.py`** | 把 Franka 笛卡尔-EE 数据集转成 pi05 通用格式。`--rot {6d,aa,quat}`、`--action-mode {absolute,delta}` | `python convert_to_pi05.py --root <src> --out <dst> --rot 6d --action-mode absolute` |
| **`train_pi05.py`** | pi05 微调(**暴露式训练循环**,便于改 loss/做研究)。argparse:`--root/--repo-id/--output-dir/--steps/...`。绕过 pi05_base 坏掉的预存 processor;state/action 自动 pad 到 32 | `python train_pi05.py --root <dataset> --repo-id <name> --output-dir ./outputs/x --steps 2000` |
| **`deploy_dryrun.py`** | 离线部署 sanity check:真实帧过同一 processor+模型,打印**预测动作 vs 真值** | `python deploy_dryrun.py` |
| **`inspect_pi05_processor.py`** | 一键 dump pi05_base 的 pre/post processor 步骤 + 版本 + JSON 差异(排查 processor 坑) | `python inspect_pi05_processor.py` |
| **`commands.txt`** | 上述命令的**单行**版,方便直接复制进 SSH | `cat commands.txt` |

---

## 4. 数据格式:pi05 到底要什么(有出处)

核对了 π0/π0.5/Knowledge-Insulation 论文 + openpi + lerobot 代码/文档,结论:

**没有任何官方来源规定 EE 的旋转表示或坐标系。** "和参考数据集完全一致"是伪命题——不存在
一个规定好的自定义机器人格式。三个内置 config 就是反证:LIBERO=7 维 EE-delta、DROID=8 维
关节速度、ALOHA=14 维关节,各用各的。

**被真正规定、决定成败的只有:**
1. proprio → `observation.state`,动作 → `action`(dotted key,HF-lerobot `PI05Policy` 直接吃);
2. 维度映射对,**模型 pad 到 32**(`action_dim=32`);
3. **算新的 quantile 归一化统计**(v3.0 数据集自带 quantile;`train_pi05.py` 用
   `dataset.meta.stats` = 新统计,正确)。π0.5 默认 **1%/99% quantile 归一化到 [−1,1]**。

**旋转表示是用户自选**:6D 在 pi05 生态零背书,但避免 ±π 断裂/双重覆盖(Zhou 2019,通用 ML
结论,非 pi05 特有)。RT-X 用欧拉、LIBERO 用轴角,没人用 6D。→ **本仓库默认 6D + 绝对 EE**:
绝对是 lerobot 默认动作空间、也是我们 `cartesian_impedance_node` 的原生空间,部署即插即用;
6D 是好回归目标。合规且合理,但记住"它不是唯一正确答案"。

### 录制的原始 schema vs 转换后的格式
| | 原始(recorder 输出) | 转换后(`convert_to_pi05.py --rot 6d`) |
|---|---|---|
| `observation.images.base` | video (720,1280,3) | 同左(模型内部 resize 到 224) |
| `observation.state` | 15 维:`xyz + 四元数(4) + 力/力矩(6) + 双指(2)` | **10 维**:`xyz + 6D旋转 + gripper` |
| `action` | 8 维:`xyz + 四元数(4) + gripper`,**绝对** | **10 维**:`xyz + 6D旋转 + gripper`,**绝对** |
| gripper | action∈[0,1](1=开);state 是指位(m) | 统一归一到 [0,1] |
| 力/力矩 | 保留 | **丢弃**(pi05 参考无力通道) |

---

## 5. 在 prs 上一步步怎么做

```bash
# 0) 进环境 + 拉最新脚本
rosml
cd ~/ros_ml_ws/pi05_finetune && git pull

# 设你要处理的数据集(改时间戳换 run)
SRC=/home/prs/ros_ml_ws/src/franka_data_recorder/data/pick_up_sponge_20260701_155201

# 1) 验证原始数据集
python check_pi05_compat.py --root $SRC --sample
#    看到 v3.0 / 1 camera / state,action / task 全 PASS 即可(四元数会 WARN,正常)

# 2) 转成 pi05 格式(6D + 绝对)。目标目录必须不存在
rm -rf ${SRC%/*}/sponge_pi05_6d_abs
python convert_to_pi05.py --root $SRC --out ${SRC%/*}/sponge_pi05_6d_abs --rot 6d --action-mode absolute
#    每条 episode 会用 SVT-AV1 编码视频(日志啰嗦但没卡),结束打印 "done -> ... dim=10"

# 3) 复查转换后(quaternion WARN 应消失,state/action = (10,))
python check_pi05_compat.py --root ${SRC%/*}/sponge_pi05_6d_abs --sample

# 4) 过拟合冒烟微调(验证数据↔标注对得上;pi05_base 已缓存,离线读)
python train_pi05.py --root ${SRC%/*}/sponge_pi05_6d_abs --repo-id sponge_pi05_6d_abs \
    --output-dir ./outputs/pi05_overfit --steps 2000 --batch-size 8 --log-every 50
#    loss 明显下降 = 数据是对的;不降/发散 = 查动作-图像同步/gripper 方向/归一化
```

> 想 A/B 对比"更贴近 LIBERO"的格式:`convert_to_pi05.py ... --rot aa --action-mode delta`。
> 也可**跳过转换直接训原始数据集**(四元数,回归差点,冒烟无所谓)。
> 官方 `lerobot-train` CLI 路线(需先修 pi05_base processor JSON)见 `pi05_deploy.md`。

---

## 6. 关键坑

- **`HF_HUB_OFFLINE`**:加载**本地**数据集时若不 offline,lerobot 会去 HF Hub 查版本 → 404。
  `check_pi05_compat.py` / `train_pi05.py` 已在 import lerobot 前设好 offline;所以 pi05_base
  **必须先缓存**(见 §1)。
- **转换目标目录必须不存在**:lerobot `create()` 用 `mkdir(exist_ok=False)`;残留目录先 `rm -rf`。
- **pi05_base 坏掉的 `relative_actions_processor`**:官方预存 processor 里有个 enabled=false 但
  registry 不认的步骤,走 `lerobot-train`/`make_pre_post_processors(pretrained_path=...)` 会报错。
  `train_pi05.py` 通过**不传 pretrained_path 给 processor**规避;CLI 路线需手动删该步骤(有 .bak)。见 `pi05_deploy.md`。
- **7 条只够冒烟**:验证管线 + 过拟合 sanity check,**训不出可用策略**。正式采集需几十~上百条。
- **相机必须固定死 + 训练/部署同位姿**:pi05 不吃相机外参,视角被隐式焊进权重;一挪就 OOD。
- **动作表示是建模选择**:部署时本来就有一层"策略输出→控制器"适配器,abs/delta 随时可转,不锁死。

---

## 7. 当前状态

- ✅ 录制管线 + 相机(usb_cam rgb8 1280×720@30)通;7 条 `pick_up_sponge` 已录。
- ✅ `check_pi05_compat.py`:原始数据集 schema 全 PASS(仅四元数 WARN)。
- ✅ `convert_to_pi05.py`:6D+绝对 转换可跑。
- ⏳ 过拟合冒烟微调 / 转换后复查:进行中。
- ⬜ 正式采集(数十~上百条,相机固定,起始位姿随机化)。
- ⬜ ROS2 部署节点(参考 `deploy_dryrun.py` + `pi05_deploy.md`)。
