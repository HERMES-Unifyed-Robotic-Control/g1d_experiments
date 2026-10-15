# C01 Full FT72500 控制修复 v2

已同步至 G1D：`/home/unitree/pi05_g1d_deploy_c01_fullft72500_control_v2_20261013`。
模型仍使用 sys01:9023 / C01 Full FT72500。无需重启server。旧client目录 `/home/unitree/pi05_g1d_deploy_c01_fullft72500_sys01_20261013` 保留。

## 本轮已实现

- 默认 `CHUNK_BLEND_STEPS=0`，移除向静止旧位置重复拉回的接管blend。保留EMA alpha=.35、arm每步增量.08、相对实测位置限幅.3、夹爪约束、4次插值、重力补偿。
- 默认 remaining6 / execute8 / 30Hz，按单调时钟安排waypoint起始时刻；接管处理消耗当前周期余量，不再无条件额外追加到完整周期之后。发生超时不突发补发动作。
- 两个有界录制队列：关键CSV/events/raw NPZ独立worker，视频独立worker。图像/数组入队时复制，机器人状态在控制线程采样，写盘在后台。视频拥塞丢弃整个chunk的录像并记数；关键记录拥塞/写盘错误显式报错，不静默丢失。
- 移除每个chunk、每轮空队列等待的同步控制台打印；阶段操作仍显示。
- `MAX_OBSERVATION_AGE_MS=500`：接管时丢弃超过此年龄的预测，保持上一目标并重新取图推理；同时保留按实际执行步跳过旧时间槽的逻辑。500ms是本轮工程阈值，不是学到的最优值；网络持续很差时可能反复持位重取，日志会显示。
- 新日志：`control_tick`、`empty_queue_wait`、带reason的chunk discard、`control_summary`、`recording_summary.json`（后台I/O耗时、队列峰值、录像丢弃、错误）。

这是已证实缺陷的第一轮修复，不含新的C1轨迹桥接/加速度规划器，也未修改模型随机采样或训练。保留这些变量以便评估本轮效果。后台线程并不构成硬实时保证；原始模型预测仍可能改方向，闭环效果尚待真机记录。

## G1D 启动

先检查（已执行通过，未初始化相机/控制器）：

```bash
cd /home/unitree/pi05_g1d_deploy_c01_fullft72500_control_v2_20261013
G1D_PROFILE=official POLICY_HOST=192.168.10.116 POLICY_PORT=9023 \
./run_client_on_g1d.sh --check-server-only
```

运行（交给操作者执行；本次未自动执行机器人运动）：

```bash
cd /home/unitree/pi05_g1d_deploy_c01_fullft72500_control_v2_20261013
G1D_PROFILE=official POLICY_HOST=192.168.10.116 POLICY_PORT=9023 \
TASK=chrysanthemum SUBTASK_INDEX=0 STEPS=3600 \
GO_READY_BEFORE_START=1 NETWORK_INTERFACE=eth0 \
GRAVITY_COMPENSATION=1 ASYNC_PREFETCH=1 \
PREFETCH_REMAINING_STEPS=6 EXECUTE_CHUNK_STEPS=8 \
INTERPOLATION_SUBSTEPS=4 CHUNK_BLEND_STEPS=0 \
ENABLE_ACTION_SMOOTHING=1 ACTION_SMOOTH_ALPHA=0.35 \
RECORDING_MODE=async MAX_OBSERVATION_AGE_MS=500 \
RECORD_DIR=/home/unitree/pi05_g1d_deploy_c01_fullft72500_control_v2_20261013/rollouts/ft72500_v2_chrysanthemum_s0_$(date +%Y%m%d_%H%M%S) \
./run_client_on_g1d.sh --execute
```

初次对照建议固定同一阶段、摆放和任务，避免一边改阶段一边评估晃动。N键手动阶段切换仍可用。

## 分别归因的三组对照

全部在v2目录，权重、remaining6、execute8、30Hz、调度方式、500ms阈值一致；替换上面命令中的两个变量并使用不同RECORD_DIR：

|组|CHUNK_BLEND_STEPS|RECORDING_MODE|比较目的|
|---|---:|---|---|
|A|2|sync|v2调度下的旧平滑与同步录制|
|B|0|sync|A→B只改变接管blend|
|C（默认）|0|async|B→C只改变录制方式|

旧目录是完整旧版本回退，不等同于表中A（调度与超时处理不同）。不要同时增加remaining、降低alpha或修改fps。记录成功/倾倒、完成时间、接管P95、空队列等待、原始目标与发送目标反向率。

## 契约与验证

`control_compatibility.json`固定完整的现有server metadata（checkpoint、E/归一化/图像、H30/native16/backend等），单独校验v2控制文件SHA256。原server的旧client SHA不伪装成新client SHA；resolved_config同时记录新控制revision/文件hash与旧server契约。错误模型、错误归一化、被修改的控制文件均拒绝。

8项离线检查在sys01及G1D实际Python环境通过：匀速轨迹、时钟超时无突发追赶、主函数参数匹配、录制复制/排空/时间戳、关键队列过载、阶段请求失效、旧预测丢弃、契约拒绝错误配置。全部未调用SDK、相机或机器人动作。另从G1D实际连接9023完成metadata检查。

- `TEST_RESULTS.txt`：sys01离线结果。
- `G1D_OFFLINE_TESTS.txt`：G1D离线结果。
- `G1D_SERVER_CHECK.txt`：实际握手检查。
- `deploy/c01_client_control_v2/`：部署代码、profile、兼容清单。
- `test_control_v2.py`：离线行为检查。

后续带丢弃chunk的记录不能直接用旧脚本假设每个chunk均有动作行；应根据 `discarded` 过滤后统计。`recording_summary` 中 `video_chunks_dropped` 是丢弃的三路图像组数，并非单个视频帧数。

本目录发布的是已部署版本的历史契约快照。换机器/权重路径时，应先按新server metadata重新生成并校验独立client契约；不能把旧路径契约直接用于新server。测试结果为原2026-10-13记录，本次发布未重跑测试或机器人运动。
