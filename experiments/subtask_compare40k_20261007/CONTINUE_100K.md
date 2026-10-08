# LoRA / 全量 FT：40k → 总计100k

2026-10-08，按用户要求启动续训。每项增加60000个optimizer steps，目标总步数100000，不是另训100000步。单张A800顺序：LoRA续训 → full FT续训；当前LoRA进程873612处于续训启动/数据准备阶段，尚未核实首个40000以后的更新。队列PID873611。

## 依据

|step|LoRA验证loss|全量FT验证loss|
|---:|---:|---:|
|30000|0.013355|0.013645|
|32500|0.013096|0.013043|
|35000|0.012795|0.012399|
|37500|0.012686|0.012301|
|40000|0.012628|0.012234|

末段仍下降，但下降幅度减小。终点恰好最优是继续训练的依据，不是未收敛或真机一定改善的证明。继续保留整体与七个subtask的heldout指标。

## 续训设置

从各自原目录的40000检查点恢复模型参数、optimizer train_state和step；不只加载推理params，不重新初始化优化器。启动断言恢复step至少40000。原检查点保留，40000是2500周期保留点。数据、split、norm stats、batch1、seed、无EMA、AdamW及可训练集合保持此前对照设置。

原cosine LR schedule的decay_steps保持40000，原末值1e-6在40k之后保持不变。仅把总训练上限扩展到100000；不会把cosine日程拉长、导致40k处LR回升到约1.6e-5，也不重新warmup。这个实验评估原学习率日程末值下继续更新，不能与“从头100k cosine”视为同一训练。

仍每2500步验证/保存，best_checkpoint沿用40k时的最佳值，只在验证更好时更新。报告路径名/exp_name仍含40k，以保持正确的恢复目录，实际目标见resolved_run.json的num_train_steps=100000。

## 元数据与日志

远端根目录保持 `/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_subtask_compare40k_20261007`。

- `run_queue_100k.py`、`queue_100k.log`、`queue_100k.pid`：新续训队列。
- `QUEUE_STATUS.json`：当前状态；旧完成状态存`QUEUE_STATUS_40000.json`。
- 每项`logs/train_100k.log`：续训日志。
- 每项`logs/report/resolved_run.json.40000_snapshot`、`best_checkpoint.json.40000_snapshot`、`TRAINING_COMPLETE.40000_snapshot`：40k元数据快照；旧完成标记移到`TRAINING_COMPLETE_40000`，避免续训期间误显示已完成。
- 原training_metrics.jsonl/validation_metrics.jsonl保留并追加。
- 每项`code/train_40000.py`保留原脚本。

## TensorBoard

地址不变：http://100.122.105.65:6024/#scalars 。已确认API显示四个run：

- lora、fullft：原0–40k历史，保持100%完成。
- lora_to100k、fullft_to100k：40k–100k续训；从总进度40%起，后续自动追加。

指标同步writer PID873610，仍5秒更新，不导入GPU训练运行时。两个新run当前只有40k起点，fullFT正在排队；显示40%不意味着它已开始续训。本次没有修改真机部署权重。
