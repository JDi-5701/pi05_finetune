# pi05_finetune

在**自己的 Franka**上,用 `franka_data_recorder` 采集的遥操作数据 **finetune π0.5 (pi05)**,
并最终部署回 ROS2。本仓库是这条链路的**验证 + 转换 + 训练 + 部署**工具集,在 GPU 机 `prs`
的 `ros_ml` conda 环境里运行。仓库同步于 <https://github.com/JDi-5701/pi05_finetune>。

> **本仓库自带三份文档,接手只读这个文件夹即可:**
> - [`README.md`](README.md)（本文）— 入口 + 操作手册 + 交接(§0)。
> - [`PI05_DATA_FORMAT.md`](PI05_DATA_FORMAT.md) — LeRobot v2.1/v3.0 格式 + pi05 要求的**字段级规范**
>   (研究结论;旋转格式的最终结论以本 README §4 为准)。
> - [`pi05_deploy.md`](pi05_deploy.md) — 部署笔记 / DROID 冒烟记录 / pi05_base processor 坑。
>
> **录制新数据**用的是**另一个仓库** `franka_data_recorder`(在 NUC/prs 的 `~/ros_ml_ws/src/`,
> 有自己的 README)。本仓库不负责采集,只消费它产出的数据集。

---

## 0. 你的任务(START HERE)

### 任务
**让现有的 7 条 `pick_up_sponge` 数据集 finetune 跑通。** 即:`train_pi05.py` 正常迭代、
loss 明显下降、存出 checkpoint。这是**过拟合/冒烟验证**(7 条学不出可泛化策略,目的是证明
数据↔标注对得上、整条 pipeline 通)——**不追求效果,只追求"跑通 + loss 下降"**。

### 完成标准 (Definition of Done)
- `train_pi05.py` 跑满 `--steps`,**无报错、无 OOM**;
- **loss 明显下降**(参考 DROID 冒烟:0.4 → 0.3 甚至更低);
- `./outputs/pi05_overfit/final/` 里有 checkpoint;
- 把你**改了什么** + **最终 loss** 记进本 README 的 §7。

### 铁律
- **只改本仓库(`~/ros_ml_ws/pi05_finetune`)的代码/脚本。绝不碰任何其他文件夹或仓库。**
- **所有产出只落在本仓库内**:转换数据 → `./datasets/`,checkpoint → `./outputs/`(均已 gitignore)。
- 报错就改**这里的脚本**(`convert_to_pi05.py` / `train_pi05.py`),改完在 §7 记一句。
- 环境:先 `rosml` 进 conda。`pi05_base` 已缓存、脚本强制 `HF_HUB_OFFLINE`,离线可训。

### 一次性准备:把原始数据拷进仓库(之后全程不读外部)
原始 7 条在仓库外(recorder 只读产出)。**拷一份进来**,让后续所有读写都在本仓库内:
```bash
rosml && cd ~/ros_ml_ws/pi05_finetune && git pull
mkdir -p datasets
cp -r ~/ros_ml_ws/src/franka_data_recorder/data/pick_up_sponge_20260701_155201 datasets/raw_pick_up_sponge
```

### 跑通流程(全部在本仓库内)
```bash
# 1) 转换 -> ./datasets/（相机自动输出 base_0_rgb，对上 pi05_base 槽位）
rm -rf ./datasets/sponge_pi05_6d_abs
python convert_to_pi05.py --root ./datasets/raw_pick_up_sponge --out ./datasets/sponge_pi05_6d_abs --rot 6d --action-mode absolute

# 2) 复查（应看到 state/action=(10,)、相机 observation.images.base_0_rgb、无 WARN）
python check_pi05_compat.py --root ./datasets/sponge_pi05_6d_abs --sample

# 3) 过拟合冒烟微调
python train_pi05.py --root ./datasets/sponge_pi05_6d_abs --repo-id sponge_pi05_6d_abs --output-dir ./outputs/pi05_overfit --steps 2000 --batch-size 8 --log-every 50
```
**loss 不降 / 发散** → 查:动作-图像是否同步、gripper 方向、归一化;改 `train_pi05.py` /
`convert_to_pi05.py`。**已知会踩的坑见 §6**(相机键名、目录不能存在、pi05_base processor)。

> 这个任务**只在本仓库内闭环**,不需要采新数据,也不用碰 `franka_data_recorder`。
> 改 loss / 模型架构做研究:在 `train_pi05.py` 的 `policy.forward(batch)` 一行附近(文件末尾有说明)。

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
| 相机 | `observation.images.base` video (720,1280,3) | **改名 `observation.images.base_0_rgb`**(对上 pi05_base 槽位;模型内部 resize 到 224) |
| `observation.state` | 15 维:`xyz + 四元数(4) + 力/力矩(6) + 双指(2)` | **10 维**:`xyz + 6D旋转 + gripper` |
| `action` | 8 维:`xyz + 四元数(4) + gripper`,**绝对** | **10 维**:`xyz + 6D旋转 + gripper`,**绝对** |
| gripper | action∈[0,1](1=开);state 是指位(m) | 统一归一到 [0,1] |
| 力/力矩 | 保留 | **丢弃**(pi05 参考无力通道) |

---

## 5. 通用流程参考(换数据集时)

> **本次交接任务的确切命令在 §0**;这里是处理**任意**数据集的通用版。产出一律放仓库内
> `./datasets/`(转换)+ `./outputs/`(checkpoint)。想处理一个新的外部数据集时,先
> `cp -r <外部数据集> datasets/<名字>`,之后全部用仓库内路径:

```bash
rosml && cd ~/ros_ml_ws/pi05_finetune && git pull
DS=./datasets/sponge_pi05_6d_abs           # 转换输出(仓库内)

# 1) 转换(相机自动输出 base_0_rgb)。目标目录必须不存在
rm -rf $DS
python convert_to_pi05.py --root ./datasets/raw_pick_up_sponge --out $DS --rot 6d --action-mode absolute
#    每条 episode 用 SVT-AV1 编码视频(日志啰嗦但没卡),结束打印 "done -> ... dim=10"

# 2) 复查(quaternion WARN 应消失,state/action = (10,))
python check_pi05_compat.py --root $DS --sample

# 3) 过拟合冒烟微调
python train_pi05.py --root $DS --repo-id sponge_pi05_6d_abs \
    --output-dir ./outputs/pi05_overfit --steps 2000 --batch-size 8 --log-every 50
```

> A/B 对比"更贴近 LIBERO"的格式:`convert_to_pi05.py ... --rot aa --action-mode delta`。
> 官方 `lerobot-train` CLI 路线(需先修 pi05_base processor JSON)见 `pi05_deploy.md`。

---

## 6. 关键坑

- **`HF_HUB_OFFLINE`**:加载**本地**数据集时若不 offline,lerobot 会去 HF Hub 查版本 → 404。
  `check_pi05_compat.py` / `train_pi05.py` 已在 import lerobot 前设好 offline;所以 pi05_base
  **必须先缓存**(见 §1)。
- **转换目标目录必须不存在**:lerobot `create()` 用 `mkdir(exist_ok=False)`;残留目录先 `rm -rf`。
- **相机键名必须匹配 pi05_base 槽位**:`pi05_base` config 期望 `base_0_rgb/left_wrist_0_rgb/
  right_wrist_0_rgb`,按**精确名**匹配;键名不对会在 forward 报 `All image features are missing`。
  `convert_to_pi05.py` 默认已把相机输出成 `observation.images.base_0_rgb`(`--out-cam-key`),
  所以转换后的数据集自动对得上;别再用原始 `observation.images.base` 那个数据集直接训。
- **pi05_base 坏掉的 `relative_actions_processor`**:官方预存 processor 里有个 enabled=false 但
  registry 不认的步骤,走 `lerobot-train`/`make_pre_post_processors(pretrained_path=...)` 会报错。
  `train_pi05.py` 通过**不传 pretrained_path 给 processor**规避;CLI 路线需手动删该步骤(有 .bak)。见 `pi05_deploy.md`。
- **7 条只够冒烟**:验证管线 + 过拟合 sanity check,**训不出可用策略**。正式采集需几十~上百条。
- **相机必须固定死 + 训练/部署同位姿**:pi05 不吃相机外参,视角被隐式焊进权重;一挪就 OOD。
- **动作表示是建模选择**:部署时本来就有一层"策略输出→控制器"适配器,abs/delta 随时可转,不锁死。

---

## 7. 当前状态

- ✅ 录制管线 + 相机(usb_cam rgb8 1280×720@30)通;7 条 `pick_up_sponge` 已录。
- ✅ `check_pi05_compat.py`:原始 + 转换后数据集 schema 全 PASS(转换后 state/action=(10,),无 WARN)。
- ✅ `convert_to_pi05.py`:6D+绝对 转换可跑,相机自动输出 `base_0_rgb`(已修命名 bug)。
- ⏳ **过拟合冒烟微调(§0 的下一步)**:模型加载 + 相机匹配已通,待确认 loss 下降。← 接手从这里继续
- ⬜ 正式采集(数十~上百条,相机固定,起始位姿随机化)。
- ⬜ 加大 steps 认真训 + 调超参。
- ⬜ ROS2 部署节点(参考 `deploy_dryrun.py` + `pi05_deploy.md`)。
