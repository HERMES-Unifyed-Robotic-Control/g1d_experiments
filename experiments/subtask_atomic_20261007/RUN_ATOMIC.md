# Subtask 原子能力验证

2026-10-07。按用户反馈，默认重新回到 9/26 曾成功抓杯的 subtask LoRA step7500（9016）。保留实测高度到位检查、数采重力补偿、连续命令插值和剩余 4 步预取/实际执行步数对齐。当前不是 QC whole 权重换短 prompt，而是训练过这些 subtask prompt 的原基准权重。没有重新训练或执行机器人动作。

GPU0 已由 QC/9020 切回 baseline/9016，QC 权重和配置保留；GPU1 的 197 whole/9019 保持独立运行。

## 本次 whole 反馈的配置

`qc153_whole_aligned_chrysanthemum_20261007_175421` 实际 resolved_config 是 `task=red_dates`、`fixed_height_m=0.150`、whole QC153 step31000、新版 executed-step 对齐。目录名不能用作实际任务标签。用户报告抓杯很难实现，本轮只核查配置，未基于录像独立判定失败原因。该结果不直接证明所有 whole 模型均不具备长程能力，但目前部署尚没有可靠长程成功证据，先验证原子动作更可诊断。

## 固定一个阶段

```bash
cd /home/unitree/pi05_g1d_deploy_alignment_20261007
G1D_PROFILE=official G1D_VARIANT=baseline \
POLICY_HOST=192.168.10.116 POLICY_PORT=9016 \
TASK=chrysanthemum SUBTASK_INDEX=0 STEPS=3600 GO_READY_BEFORE_START=1 NETWORK_INTERFACE=eth0 \
GRAVITY_COMPENSATION=1 ASYNC_PREFETCH=1 PREFETCH_REMAINING_STEPS=4 \
ENABLE_ACTION_SMOOTHING=1 \
RECORD_DIR="/home/unitree/pi05_g1d_deploy_alignment_20261007/rollouts/atomic7500_chrysanthemum_s0_$(date +%Y%m%d_%H%M%S)" \
./run_client_on_g1d.sh --execute
```

不按 N，固定当前短 prompt；默认 fixed schedule 不会按时间自动推进。当前实现仍保留 N handler，按键会切换，原子验证期间不要使用。没有自动成功检测，STEPS 只是步数上限。

|SUBTASK_INDEX|能力|起始场景与人工确认事件|
|---:|---|---|
|0|右手抓黑杯并搬到中间|黑杯在右侧；抓住、离桌、移到中间并释放|
|1|放当前 TASK 的食材|黑杯已在中间，食材位置接近示教；食材放入杯内|
|2|左手抓白杯|白杯/机械臂起始状态接近对应示教；稳定抓住并提起|
|3|右手抓稳黑杯|黑杯在中间，白杯等状态接近示教；黑杯抓稳|
|4|倒水并归位|两杯与双臂状态接近倒水示教起点；倾倒/回放过程完成|

后续阶段单独测试：改 SUBTASK_INDEX，并在场景和关节起点准备好后用 `GO_READY_BEFORE_START=0`、initial_state=current，避免固定 ready 动作覆盖已准备好的阶段起点。底层 controller 的初始化行为仍存在，应确认接管后的实际状态；不能宣称任意姿态直接启动某阶段都在训练分布内。每个阶段用独立 RECORD_DIR，如 s1/s2 等。先记录成功条件、失败现象与重试次数，再讨论串联。

换 TASK 时同时修改目录标签。菊花/红枣/柠檬自动高度为 0.065/0.150/0.145 m，启动/停止精确权限已安装。请勿给红枣/柠檬额外强制 FIXED_HEIGHT_M=0.065。

## 后续安排

先分别评估五个原子阶段；单项尚不稳定时不以手动串联成功假设替代证据。等单项可靠后，再讨论基于观测的完成判定与阶段接续。当前没有新增完成阈值、自动切换或训练任务。
