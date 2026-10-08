最新状态：用户反馈后已恢复 subtask/9016 默认入口，QC/9020 暂停。下文保留 whole 部署复现配置，运行 QC 前需 `manage_server.py switch qc`。原子阶段验证见 `../subtask_atomic_20261007/RUN_ATOMIC.md`。

# 三个完整任务的新版部署

2026-10-07。默认入口已从 subtask 基准切到 QC whole-task：只设置 TASK，整段使用训练时的完整任务 prompt，不创建 N handler，不按阶段定时改 prompt。三个任务共用同一 joint checkpoint，不是三套独立权重。模型仍每次预测 32 步动作、执行其中 8 步后重规划，模型不会直接输出整条 episode。

## 当前服务

|variant|端口|权重|GPU|
|---|---:|---|---:|
|qc（默认）|9020|QC-balanced 153 episode，full FT step31000|0|
|joint197|9019|197 episode，full FT step36000|1|
|baseline|9016|9/26 subtask LoRA step7500，当前暂停|0|

两套 whole 权重已接到同一最新版客户端，含：实测高度到位检查与日志、数采静态重力补偿 scale=1、连续命令插值、剩余 4 步预取、按观测后实际执行步数对齐、过期结果丢弃、原始动作/时序日志。原 9019 服务已迁移到新版契约，原客户端不再与其匹配。模型参数和 norm stats 没有修改，没有重新训练。启动过程有模型 warmup；本轮未发送真机动作，也未运行新增测试。

## G1D 启动（QC，菊花）

```bash
cd /home/unitree/pi05_g1d_deploy_alignment_20261007
G1D_PROFILE=official G1D_VARIANT=qc \
POLICY_HOST=192.168.10.116 POLICY_PORT=9020 \
TASK=chrysanthemum STEPS=3600 GO_READY_BEFORE_START=1 NETWORK_INTERFACE=eth0 \
GRAVITY_COMPENSATION=1 ASYNC_PREFETCH=1 PREFETCH_REMAINING_STEPS=4 \
ENABLE_ACTION_SMOOTHING=1 \
RECORD_DIR="/home/unitree/pi05_g1d_deploy_alignment_20261007/rollouts/qc153_whole_aligned_chrysanthemum_$(date +%Y%m%d_%H%M%S)" \
./run_client_on_g1d.sh --execute
```

无需 SUBTASK_INDEX，不需要按 N。红枣改 `TASK=red_dates`、录制目录标记 red_dates；柠檬改 `TASK=lemon_slices`、目录标记 lemon_slices。请勿沿用菊花的 `FIXED_HEIGHT_M=0.065` 覆盖其他任务。

对照 197 模型时，用同一目录改 `G1D_VARIANT=joint197 POLICY_PORT=9019`，目录标记 joint197_whole。

可先 `./run_client_on_g1d.sh --check-server-only`；默认是 QC/9020，显式 variant/port 可以核查另一套服务，不创建相机或电机控制器。

## 三个任务的物理高度

菊花 0.065 m、红枣 0.150 m、柠檬片 0.145 m，按 TASK 自动选择。数值是升降机构状态/目标，不是夹爪离桌高度。

2026-10-07 17:42：已在 G1D 安装精确任务高度权限，三个目标的免密启动和停止规则均检查通过。启动调用同一 SDK 程序 `/home/unitree/unitree_sdk2/build/bin/g1d_height_control`；客户端把参数格式化为三位小数（0.150 与 0.15 数值相同，但 sudoers 按文本精确匹配，所以免密调用请写 0.150）。停止 helper 精确匹配 0.065/0.145/0.150，旧 helper 保留备份。本轮只安装权限和查询规则，没有执行高度控制或改变机器人姿态。凭据未保存进项目文件。无需再次安装，可以按 TASK 选择三个任务。

## 任务推进的能力边界

人工切 prompt 若早于真实完成，会让下一段输入处于训练之外的状态；使用 whole prompt 可避免人为切换时刻不匹配，并让模型根据视觉/关节状态隐式推进。但它仍可能在抓取失败、物体错位或异常姿态下进入错误轨迹。当前 whole 模型没有显式 done 输出、抓取成功判定、可靠恢复或自动结束判据；STEPS 是步数上限，不代表任务完成。

## 服务管理

```bash
python /mnt/data2/yangky/test/g1d_server_alignment_20261007/manage_server.py status
python /mnt/data2/yangky/test/g1d_server_alignment_20261007/manage_server.py switch baseline
python /mnt/data2/yangky/test/g1d_server_alignment_20261007/manage_server.py switch qc
```

baseline/QC 互占 GPU0，切换前停止客户端；joint197 使用 GPU1，保持独立运行。
