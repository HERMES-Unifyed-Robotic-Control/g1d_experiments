# G1D Food Sorting Experiments

G1D 菊花、红枣、柠檬片任务的完整数据处理、subtask 标注、pi0.5 baseline、训练对照、真机执行和 Dexjoco C01 G1D训练与评测适配。**当前 baseline 为官方 OpenPI pi0.5 七个 subtask 的 LoRA step7500**，保留 2026-09-26 曾成功抓杯的权重；它是独立baseline，尚不是G1D C01权重。用户报告当前控制修正后的效果可用，但尚无系统闭环成功率统计。

> 标注、四个视频数据包和7500baseline均已上传并固定revision。比较权重受A800网络/跳板断线影响尚未完成上传；以 `ARTIFACTS.json` 中 `phase=complete` 为准。

## 交付布局

|目录|内容|
|---|---|
|`data/`|197条原episode的五阶段标注、七类短prompt、原episode分组划分与固定验证观测|
|`scripts/`|数据/权重下载、可迁移的baseline服务、同条件LoRA/full FT训练|
|`deploy/client/`|当前真机客户端：高度实测、重力补偿、连续插值、异步预取、人工N切换|
|`deploy/server/`|现有sys01服务管理与契约，路径含现场机器布局|
|`g1d_experiments/c01_dataset.py`|G1D原生16D的C01窗口接口，推理观测与离线监督分开|
|`g1d_experiments/c01_decoder.py`|复用C01架构的16D decoder与GT-AUX/gripper-Δ组件|
|`configs/c01_g1d.yaml`|C01 G1D接口与尚需完成的训练环节|
|`vendor/openpi/`|运行时OpenPI/PSI源码快照及原LICENSE、依赖配置|
|`vendor/xr_teleoperate_g1d/`|G1D控制、相机、URDF/mesh及原LICENSE|
|`third_party/dexjoco/`|指定C01分支、固定commit的Dexjoco submodule|
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

A800训练使用Python3.12；当前sys01推理环境为Python3.10.20，核心为JAX0.5.3/CUDA、Flax NNX、Orbax、LeRobot v2.1生态。`requirements-baseline-sys01.txt` 记录当前sys01核心运行依赖版本；其中LeRobot为0.1.0/v2.1数据生态，不能直接替换成最新版本。`vendor/openpi/pyproject.toml` 是历史PSI工程配置（Python3.10），不是本次Python3.12 JAX baseline的安装入口。训练关键模块是vendored `openpi`。按上述版本安装运行依赖及对应CUDA支持，再加源码路径：

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

待上传的比较权重规划位于同一私有模型仓库 `comparisons/{lora,fullft}_{step}`，仅含推理params/assets，不包含optimizer续训状态。

已完成的40k对照：LoRA validation loss0.0126283，full FT0.0122344；首步抓杯0.0141312/0.0142837。loss不是闭环成功率。100k续训状态见实验报告，其静态状态可能落后于训练进程；不要把未完成训练当作结果。

## 固定holdout离线评估

```bash
python scripts/evaluate_subtask.py --checkpoint artifacts/baseline/7500 --variant lora --output artifacts/evaluation/baseline7500
python scripts/download_artifacts.py --comparison fullft_40000
python scripts/evaluate_subtask.py --checkpoint artifacts/comparisons/fullft_40000 --variant fullft --output artifacts/evaluation/fullft40000
```

使用相同200条观测、固定采样噪声种子，输出16D物理动作MAE/RMSE、逐task/关节/时间步误差及预测npz。MAE混合不同关节维度，比较时看逐维指标；该holdout曾用于选ckpt，不能作为独立测试集或真机成功率。本次交付没有重新运行评估。

## baseline服务和真机

```bash
python scripts/serve_baseline.py --checkpoint artifacts/baseline/7500 --port 9016
```

此入口会将`deploy/client/g1d_profiles_baseline.json`的checkpoint字段更新为实际路径，以绑定准确契约。将生成后的同一client目录复制到G1D；通过PYTHON_BIN指定解释器，并修改URDF路径及XR_TELEOP_ROOT后部署。G1D还需Unitree SDK2/Python、Pinocchio、相机服务及openpi-client；升降示例源代码与SDK commit见`deploy/height_control/`。在目标G1D编译height_control，人工安装`install_height_permissions.sh`中的三种精确高度/停止权限。保持server/client源码、profile、norm stats的hash一致，不绕过metadata验证。

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

## Dexjoco C01匹配（已按用户更正）

模型参考改为 [Dexjoco `zbk/c01-single-decoder-finger-delta`](https://github.com/HERMES-Unifyed-Robotic-Control/Dexjoco/tree/zbk/c01-single-decoder-finger-delta)，commit `90fa71f83c25b1ac0c5c4ba9676464015ba069b8`。该分支提供C01 E训练/导出、π0.5单decoder+GT-AUX+finger-Δ训练及评测。之前的EFT接口已撤下。

|接口|上游C01|G1D匹配|
|---|---|---|
|动作/状态|单臂22/23D、双臂44/46D；末端位姿+16指关节/手|16/16D，双臂绝对qpos+两个夹爪电机|
|动作头|单臂、双臂头|新增原生16D头，不能复用44D输出层|
|E序列|30×256|保留30×256接口，G1D的E需要重新生成并训练|
|时间|30Hz、H30|30Hz、H30；7500baseline另为H32|
|finger-Δ|每手16维手指|对应夹爪维度14、15的gripper-Δ|
|机器人响应|FK根位姿、4指尖、16指关节/手|组件提供实测16D关节响应；这是G1D适配变体，不是原FK目标|
|视觉|指定仿真视图与DINO/PCA世界变化特征|三路真实RGB；DINO与train-only PCA需按G1D重新准备|
|划分|原11任务100episode规则|沿用177/20原episode分组，不套用0–89/90–99|

已实现G1D H30窗口、16D状态/动作、仅相邻有效帧参与的gripper-Δ损失，以及复用上游ActionDecoder架构的16D decoder组件。RGB为uint8，状态/动作为物理单位。统计量需从train拟合。缺失的触觉、指尖或世界特征不会补零后冒充实测有效数据。

```python
from torch.utils.data import DataLoader
from g1d_experiments.c01_dataset import G1DC01Windows, collate_c01_windows
loader = DataLoader(G1DC01Windows('artifacts/data/g1d_subtasks_official_train_v1'),
                    batch_size=1, collate_fn=collate_c01_windows)
# π推理只消费batch['observation']；batch['supervision']仅用于E/损失监督
```

**这不是已训练好的G1D C01 policy。** G1D state/action实测维数、train-only统计已完成；已实现视觉准备、E训练/cache导出、16D JAX decoder/pi桥接、GT-AUX校准、π训练及动作评估入口。运行流程见 [RUN_C01.md](experiments/g1d_c01_20261008/RUN_C01.md)。DINOv3已通过Meta邮件授权下载，原生`.pth`加载与A800真实视频特征提取已运行。E30k→π LoRA40k队列已启动；尚无完成的C01策略或对比结果。上游仿真C01检查点不能直接插到G1D执行。上游GT-AUX权重0.25628781345139295和finger-Δ权重0.05只是参考，不声称在G1D已校准或有效。详见[匹配说明](DEXJOCO_G1D_MATCHING.md)。真机服务/独立H30客户端已实现，尚未执行真机动作。

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

vendor保留各上游许可证；数据沿用既有HF dataset card。Dexjoco及第三方组件遵循其原许可证。GitHub代码、HF数据及私有模型权限独立管理。

## 训练监控

TensorBoard访问、tmux后台会话、文件迁移进度指令见[TENSORBOARD.md](TENSORBOARD.md)。C01训练JSONL通过`scripts/jsonl_to_tensorboard.py`持续导出，不重启训练。

E/π、训练监督和在线推理链路说明见[MODEL_GUIDE.md](MODEL_GUIDE.md)。

## C01首轮A800 40k结果

E30k与π LoRA40k已完成，选中π40000。200个holdout观测、前30步native16动作误差对比与独立G1D客户端入口见[首轮结果](experiments/g1d_c01_20261008/results_a800_40k/README.md)。MAE下降11.60%，RMSE下降16.14%；尚未进行该权重的真机成功率评测。A100迁移完成后已启动双实例LoRA训练，global batch2。
