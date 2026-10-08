当前默认已恢复 baseline/9016 subtask 原子验证，QC/9020 暂停，joint197/9019 保留。最新启动与分阶段说明见 `RUN_ATOMIC.md`；下文 whole 模式状态是历史记录。

# 9/26 基准 + 高度实测 + 重力补偿 + 异步执行

最新状态（2026-10-07）：默认已切换 QC whole-task，GPU0/9020 是 QC153 step31000，GPU1/9019 是同步修正的 joint197 step36000；9016 subtask 暂停。三个完整任务启动见 `RUN_WHOLE.md`。以下保留 subtask 基准历史配置，使用 9016 前需要先切换 GPU0 服务。

机器人新版目录：`/home/unitree/pi05_g1d_deploy_alignment_20261007`。所有真机动作由操作员通过 `--execute` 启动；此次开发只做状态订阅、模拟控制器验证、模型服务与不发送指令的 dry-run。

## 高度选择

当前只读实测高度 `0.065664 m`、升降速度 0；腰 yaw 约 -0.00696 rad，pitch 0。这是当前测量，不代表 9/26 的高度已被还原。

基准菊花默认 0.065 m，与该任务示教数值一致。红枣默认 0.150 m、柠檬默认 0.145 m。启动器支持用 `FIXED_HEIGHT_M` 覆盖目标，但机器人当前 sudoers 的免密规则只允许 `eth0 0.065`，停止 helper 也仅匹配这一条命令。红枣/柠檬目标及 0.1 m 实机试验必须先配置对应的精确启动权限和停止规则；当前可直接执行的是菊花 0.065 m 基准。

0.1 m 可以作为单独试验设置，与菊花默认相差 +3.5 cm，并不等于实测高度或三个任务共同对齐高度。每次控制前会读取真实高度，再等待目标误差不超过 3 mm、速度不超过 0.005 m/s 持续 0.5 s；15 s 未到位或 DDS 状态过期则停止启动流程。

## 默认优化参数

|参数|值|
|---|---|
|异步预取|开启，执行 8 步、剩余 4 步时开始预取|
|chunk 边界混合|前 2 步|
|chunk 内插值|每个 30 Hz waypoint 分 4 个子步|
|额外动作平滑|开启，alpha=0.35，臂 max_dq=0.08 rad|
|重力补偿|开启，scale=1.0，使用同一 G1D URDF 和数采的静态 RNEA(q,0,0)|
|启动姿态|go-ready 后读 current 实测状态|

重力补偿应用于 ready 目标和所有策略插值子步。退出仍调用原有 controller 的 go-home 流程，该流程使用原有零前馈行为。当前实现假定腰部回到零姿态，与数采模型的固定腰部 reduction 一致。

异步 N 切换已接入：当前 chunk 停止，旧预取 generation 作废，从实测状态请求下一条 subtask prompt。whole 模式不创建 N 切换 handler。

## 验证结果

- 新重力模型与机器人数采缓存模型比较 128 组姿态，最大力矩差 0 Nm。
- 模拟控制器验证 4 个插值子步均收到相应姿态的重力力矩；没有创建真实电机控制器。
- 验证旧异步 generation 的结果被丢弃，新阶段 prompt 正常返回。
- 机器人端 server-only 契约检查通过：checkpoint、norm stats、profiles、客户端与 runtime helper 的 hash 对应。
- 真实相机/网络/模型参与的 180 步无动作 dry-run：28.45 waypoint/s，步间隔中位数 33.46 ms，p99=50.01 ms。此结果不包含真实电机控制、重力执行与接触，不能宣称真机已达到相同频率。

## 在 G1D 上启动

先检查服务，不创建控制器：

```bash
cd /home/unitree/pi05_g1d_deploy_alignment_20261007
G1D_PROFILE=official G1D_VARIANT=baseline POLICY_HOST=192.168.10.116 POLICY_PORT=9016 \
./run_client_on_g1d.sh --check-server-only
```

需要只读查看实测高度及补偿力矩时：

```bash
/home/unitree/miniconda3/envs/tv/bin/python robot_alignment.py --network-interface eth0
```

基准真机执行：

```bash
G1D_PROFILE=official G1D_VARIANT=baseline \
POLICY_HOST=192.168.10.116 POLICY_PORT=9016 TASK=chrysanthemum SUBTASK_INDEX=0 \
STEPS=3600 GO_READY_BEFORE_START=1 FIXED_HEIGHT_M=0.065 NETWORK_INTERFACE=eth0 \
GRAVITY_COMPENSATION=1 ASYNC_PREFETCH=1 ENABLE_ACTION_SMOOTHING=1 \
RECORD_DIR="/home/unitree/pi05_g1d_deploy_alignment_20261007/rollouts/baseline7500_gravity_async_h0065_$(date +%Y%m%d_%H%M%S)" \
./run_client_on_g1d.sh --execute
```

0.1 m 试验参数为 `FIXED_HEIGHT_M=0.1`，录制目录标记 `h0100`。目前尚不具备该高度的启动/停止权限，不能直接执行这一试验；只改参数会在高度控制启动阶段报错。完成对应精确 sudoers 与停止 helper 配置后，这个设置才会实际改变升降高度。

## 在 sys01 切换 GPU0 的基准/QC 服务

操作前停止 G1D 客户端。管理程序会检查客户端是否仍在运行，只终止本目录管理的模型进程。

```bash
python /mnt/data2/yangky/test/g1d_server_alignment_20261007/manage_server.py switch qc
```

QC 就绪后，在 G1D 用同一新版目录，将 `G1D_VARIANT=qc POLICY_PORT=9020`，录制目录标记 `qc31000`。该变体使用 whole prompt，无 N 切换。

切回基准：

```bash
python /mnt/data2/yangky/test/g1d_server_alignment_20261007/manage_server.py switch baseline
```

原始 QC 版本仍保存在 `/mnt/data2/yangky/test/g1d_server_qc_balanced_153_20261007` 和机器人原 QC 客户端目录。新服务的契约绑定新版客户端；原客户端不能与新服务混用。

## 录制内容

`resolved_config.json` 保存实际采用的设置；`events.jsonl` 保存启动前/到位后的实测高度、每步高度/速度/新鲜度、腰 yaw/pitch、补偿力矩和 N 切换事件；`raw_policy_chunks/` 保存混合/平滑前完整 32x16 模型动作；原 action CSV/NPZ 继续记录最终目标与实测关节。

下一次真机评估先检查升降到位、补偿力矩、实际 waypoint 频率与抓杯/离桌事件，再与 9/26 成功轨迹比较。

## 16:56 真机反馈后的插值修正

该次真机实测约 25.96 Hz，存在 chunk 边界停顿和策略目标往返。已将每步插值起点改为上一条已发送目标，避免从滞后实测值重新起步产生锯齿；首周期仍从实测起步，保留目标与实测的偏差限幅。权重、PD、重力比例及其他优化参数保持本次对照设置。`resolved_config.json` 记录 `control_revision=continuous_command_interpolation_20261007`。原启动命令继续有效，请使用新录制目录；此次修正尚无真机结果。详细证据见 sys01 的 `oscillation_audit_20261007/REPORT.md`。

## 17:16 反馈后的预取调整

已将预取从剩余 8 步延后到剩余 4 步，按观测构建完成之后实际执行的 waypoint 数对齐新 chunk，保持等待不计作动作进度。新增 `chunk_activation` 时序事件。版本为 `late_prefetch_executed_step_alignment_20261007`。180 步无动作联调约 25.42 Hz，后续块均消费 offset=3；尚无新版真机结论。启动时可显式设置 `PREFETCH_REMAINING_STEPS=4`，使用新录制目录 `baseline7500_prefetch4_stepalign_h0065_<timestamp>`。
