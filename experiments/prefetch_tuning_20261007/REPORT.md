# 异步预取时序调整（2026-10-07）

## 最新真机记录

来源 `baseline7500_continuous_interp_h0065_20261007_171633`。配置确认 `control_revision=continuous_command_interpolation_20261007`，确实已使用连续命令插值，并非误跑旧客户端。3600 个 waypoint，实际 26.41 Hz；步间隔 p99 76.46 ms、最大 295.38 ms。416 个完整 chunk 通过 raw/logged action 匹配推断消费 offset=8，即主要消费第 8–15 步。全程右肘新块 raw 起点与上块最终目标绝对差 p95≈0.252 rad。

原配置执行 8 步且剩余 8 步就预取，新块观测在旧块刚开始时取得。到接管时已过去约一个块。原实现还按请求启动以来的墙钟时间跳过动作，把相机读取、暂停和记录时间混入动作进度。连续插值解决了每步从滞后实测值起步的锯齿，但未解决新块使用较早观测、较远未来预测带来的轨迹修正。

## 本次修改

- baseline/QC 默认 `prefetch_remaining_steps=4`，`execute_chunk_steps=8` 保持：旧块后半段才开始取新观测。
- worker 在观测构建结束后记录 `observed_at` 和 `observed_step`，与 q-state 采样接近；这不是相机传感器曝光时间，各路图像同步问题仍需专门检查。
- 新块接管时按 `current_step - observed_step` 跳过已实际执行的 waypoint，不按请求墙钟年龄跳过。等候中保持旧目标不算额外动作进度。
- 对超过整个 32 步预测窗口的结果丢弃并重新请求，不再强行选取末尾动作。
- `chunk_activation` 事件记录观测年龄、读取时间、观测到结果时间、结果等待接管时间、跳过步数及丢弃标志。`resolved_config.json` 版本为 `late_prefetch_executed_step_alignment_20261007`。
- 保留连续命令插值、重力补偿、PD、EMA 与 2 步边界混合，未改模型权重。N 切换的旧 generation 丢弃逻辑保留。

边界混合锚点是上一条已发送目标，不是旧观测姿态；它仍会延缓接近新目标。未来可单独比较 `CHUNK_BLEND_STEPS=0`，当前先固定这一项以观察时序改变的效果。策略本身有往返时，当前修改不能保证消除所有晃动。

## 不发送动作的联调

已部署到 G1D 原 alignment 目录，刷新 9016 同一权重服务，服务契约检查通过。

180 步真实相机/网络/服务参与的 dry-run，没有 --execute，没有创建电机控制器：25.42 Hz；除首块 offset=0，其余 22 个块 offset=3。观测到接管年龄中位数 142.96 ms、p95 233.08 ms，结果待接管中位数 1.04 ms。剩余 4 步的预取窗口仍不足以覆盖部分请求，因此可能发生队列等待；不能声称真机频率或晃动已改善。

dry-run 路径：G1D `rollouts/dryrun_prefetch4_20261007`。本地 `dryrun_summary.json`、`events.jsonl`、`dryrun.log` 保存本次证据。下一次真机先用相同初始条件评估首步，观察往返幅度、chunk 时序、杯子离桌及抓取稳定性，再决定是否调整预取 5 步或边界混合。

旧源码/配置在 `before_fix/`；最新用户真机复制记录在 `rollout/`。`tracking.png` 中插值红线按照连续命令源码重建，并非低层电机命令抓包。
