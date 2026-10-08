# G1D Food Sorting Experiments

G1D 菊花、红枣、柠檬片任务的完整数据处理、subtask 标注、pi0.5 baseline、训练对照、真机执行和 EFT 数据接口交付。**当前 baseline 为官方 OpenPI pi0.5 七个 subtask 的 LoRA step7500**，保留 2026-09-26 曾成功抓杯的权重；不是 EFT 权重。用户报告当前控制修正后的效果可用，但尚无系统闭环成功率统计。

> 发布进行中：大型数据包/权重的完成状态以 `ARTIFACTS.json` 中 `phase=complete` 为准。标注已完成上传。

## 交付布局

|目录|内容|
|---|---|
|`data/`|197条原episode的五阶段标注、七类短prompt、原episode分组划分与固定验证观测|
|`scripts/`|数据/权重下载、可迁移的baseline服务、同条件LoRA/full FT训练|
|`deploy/client/`|当前真机客户端：高度实测、重力补偿、连续插值、异步预取、人工N切换|
|`deploy/server/`|现有sys01服务管理与契约，路径含现场机器布局|
|`g1d_experiments/eft_dataset.py`|LeRobot到EFT历史输入/未来监督适配器|
|`configs/eft_g1d.yaml`|三路视图、16D状态/动作、30Hz的EFT实验数据配置|
|`vendor/openpi/`|运行时OpenPI/PSI源码快照及原LICENSE、依赖配置|
|`vendor/xr_teleoperate_g1d/`|G1D控制、相机、URDF/mesh及原LICENSE|
|`third_party/eft/`|固定版本的EFT上游submodule|
|`historical/`|原始转换/训练脚本和现场创建训练队列脚本；不是所有脚本都可无参数迁移|
|`experiments/`|关键审计、训练配置、部署说明、时序诊断与历史状态|
|`ARTIFACTS.json`|Hugging Face revision、SHA256、下载定位|
|`PROVENANCE.json`|源码来源与commit；vendor包含现场修改|

## 数据、标注和划分

原始数据已在 [Hugging Face dataset](https://huggingface.co/datasets/yky666/humanoid_long_horizon_tasks) 的 `food_sorting.tar`。新增文件在 `g1d_food_sorting/` 下：

- `annotations/`：可审阅标注与划分；时间索引细节以 `historical/tools/make_subtask_lerobot_dataset.py` 为准，标注端点可共用，勿自行改为不同区间语义。
- `processed/whole_197.tar`：完整episode LeRobot v2.1。
- `processed/subtasks_all.tar`：已切分七类prompt的全部subtask片段。
- `processed/subtasks_train.tar`、`subtasks_validation.tar`：baseline实际使用的885/100片段、183647/21907帧，按原episode分组后划分，禁止对片段随机再划分造成泄漏。共177个train/20个validation原episode。

归一化只来自训练集。所有processed压缩包展开视频符号链接，不依赖本机路径。三路映射：raw `color_0/2/3` → `egocentric/left_wrist/right_wrist`。状态/动作顺序：left_arm7, right_arm7, left_gripper, right_gripper，绝对关节qpos，臂rad；升降和腰部不在16D模型输入/动作中。

```bash
python -m pip install -r requirements-data.txt
python scripts/download_artifacts.py --dataset subtasks_train --dataset subtasks_validation --baseline
# 如需重新切分或完整任务训练：另下载 whole_197 / subtasks_all
```

Baseline模型仓库为私有：[yky666/g1d-pi05-subtask-baseline](https://huggingface.co/yky666/g1d-pi05-subtask-baseline)。需有权限的HF账号登录或通过HF_TOKEN提供权限，勿把token写入代码。GitHub私有仓库访问权限不会自动授予HF私有模型权限。

## 安装baseline环境

运行时参考Python3.12、JAX0.5.3/CUDA、Flax NNX、Orbax、LeRobot v2.1生态。`requirements-baseline-sys01.txt` 记录当前sys01核心运行依赖版本；其中LeRobot为0.1.0/v2.1数据生态，不能直接替换成最新版本。`vendor/openpi/pyproject.toml` 是历史PSI工程配置（Python3.10），不是本次Python3.12 JAX baseline的安装入口。训练关键模块是vendored `openpi`。按上述版本安装运行依赖及对应CUDA支持，再加源码路径：

```bash
export PSI_HOME="$PWD/vendor/openpi"
export DATA_HOME="$PWD/artifacts/data"
export PYTHONPATH="$PWD/vendor/openpi/src:$PWD/vendor/openpi/src/openpi/openpi-client/src:$PYTHONPATH"
# openpi-client: python -m pip install -e vendor/openpi/src/openpi/openpi-client
```

GPU、CUDA、torch/LeRobot视频解码依赖按目标机器配置；训练脚本记录实际配置、数据/统计hash和可训练参数量。仓库不承诺无需安装依赖即可运行。

## 训练可迁移入口

另行下载官方pi05 base JAX params（不是用本项目7500params充当base）。

```bash
export HF_HOME="$PWD/artifacts/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES=0 python scripts/train_subtask.py --variant lora --steps 40000 --base-params /path/to/pi05_base_jax/params
CUDA_VISIBLE_DEVICES=0 python scripts/train_subtask.py --variant fullft --steps 40000 --base-params /path/to/pi05_base_jax/params
# 从已完成40k继续到总计100k：同一variant增加 --resume --steps 100000
```

可迁移入口复用已有训练/验证逻辑，数据、统计、batch1、无EMA、AdamW一阶动量bfloat16、seed42和学习率条件匹配。每2500步验证保存。续训保留原40k LR日程末值1e-6。原A800精确脚本创建记录见 `historical/compare_training/`，其机器路径是审计材料。

交付的比较权重位于同一私有模型仓库 `comparisons/{lora,fullft}_{step}`，仅含推理params/assets，不包含optimizer续训状态。

已完成的40k对照：LoRA validation loss0.0126283，full FT0.0122344；首步抓杯0.0141312/0.0142837。loss不是闭环成功率。100k续训状态见实验报告，其静态状态可能落后于训练进程；不要把未完成训练当作结果。

## baseline服务和真机

```bash
python scripts/serve_baseline.py --checkpoint artifacts/baseline/7500 --port 9016
```

此入口会将`deploy/client/g1d_profiles_baseline.json`的checkpoint字段更新为实际路径，以绑定准确契约。将生成后的同一client目录复制到G1D；修改URDF路径、Python解释器及XR_TELEOP_ROOT后部署。保持server/client源码、profile、norm stats的hash一致，不绕过metadata验证。

现场G1D指令（保持原目录和机器网络）：

```bash
cd /home/unitree/pi05_g1d_deploy_alignment_20261007
G1D_PROFILE=official G1D_VARIANT=baseline POLICY_HOST=192.168.10.116 POLICY_PORT=9016 \
TASK=chrysanthemum SUBTASK_INDEX=0 STEPS=3600 GO_READY_BEFORE_START=1 NETWORK_INTERFACE=eth0 \
GRAVITY_COMPENSATION=1 ASYNC_PREFETCH=1 PREFETCH_REMAINING_STEPS=4 \
RECORD_DIR="/home/unitree/pi05_g1d_deploy_alignment_20261007/rollouts/baseline7500_$(date +%Y%m%d_%H%M%S)" \
./run_client_on_g1d.sh --execute
```

先以`--check-server-only`核对契约。固定原子阶段时不按N；只有已确认当前阶段完成后才人工推进。高度按TASK为0.065/0.150/0.145m，是升降机构目标而非夹爪离桌距离。原子起点/GO_READY细节见`experiments/subtask_atomic_20261007/RUN_ATOMIC.md`。当前没有自动成功判定/自动终止输出。

## EFT匹配：已完成接口，训练仍依赖上游实现

[EFT](https://github.com/HERMES-Unifyed-Robotic-Control/EFT-Embodied-Future-Transitions) 的当前版本是RoboCasa骨架，多处`NotImplementedError`，不能声称已有可评测EFT policy。本交付固定上游commit并提供G1D数据适配，未伪造EFT训练结果或权重。

适配器输入仅包含`inputs.rgb_history/state_history/language`；未来图像/状态/动作只在`targets`中，不得传入U predictor。RGB为未归一化uint8，编码器需完成自身normalization。历史4帧/0.5s；G1D按30Hz保留1.5s未来窗口H45，三路视图。官方baseline独立使用H32；两种horizon不可混淆。窗口不跨episode/subtask边界，训练/验证继承原episode分组划分。

```python
from g1d_experiments.eft_dataset import G1DPredictiveWindowDataset
train = G1DPredictiveWindowDataset('artifacts/data/g1d_subtasks_official_train_v1')
# sample = train[i]; EFT预测器只消费sample['inputs']
```

要测师兄的EFT模型，还需接入实际实现后的编码器/U predictor/动作头、其检查点与输入输出协议，再在同一holdout及真实起始条件下比较。本次没有新增模型训练或真机评测。

## 进度评分（满分10）

|里程碑|分值|
|---|---:|
|抓住并提起黑杯|1|
|黑杯搬到中间并稳定释放|1|
|抓起物料并稳定离开料盘（空夹不计）|3|
|物料释放到黑杯内|1|
|左手抓起白杯|1|
|右手抓稳黑杯|1|
|明确将水倒入黑杯|1|
|两杯稳定放回并完成归位|1|

人工代完成不得计分。记录是否人工prompt切换，物料目标抓取量预先固定；最终10分需场景正确、无杯子倾倒等破坏性失败。分别报告progress/10、物料抓取成功率、全任务成功率。

## 复现边界与许可证

历史机器路径、旧服务状态作为证据保留，当前默认baseline7500与修正后的客户端。已去除源码中的凭据字面量，不包含SSH配置、账户密码或token。未运行新增测试；数据/权重上传完成后记录HF不可变revision和校验值。

vendor保留各上游许可证；数据沿用既有HF dataset card。EFT上游未附许可证时不推定其许可，使用其submodule和模型前确认上游授权。GitHub代码、HF数据及私有模型权限独立管理。
