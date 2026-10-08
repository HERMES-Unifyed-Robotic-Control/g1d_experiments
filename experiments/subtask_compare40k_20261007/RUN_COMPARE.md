# 七个 subtask：40k LoRA / 全量 FT 对照

2026-10-07，用户授权启动训练。真机基准服务保持 9016 LoRA7500 和当前控制修正，不修改机器人部署权重。

## 发现的历史训练

A800 历史 `g1d_pi05_lora_subtasks_20260930` 的验证记录到 13500，最佳 13500 validation loss 0.0122581；没有证据表明该项已完成 40k。历史 `g1d_pi05_fullft_subtasks_20260929` 验证记录已到 40000，validation loss 0.0120189。它们使用相同七阶段数据和归一化，但旧 LoRA batch=8、无 EMA、标准 AdamW，旧 full FT batch=1、EMA=.99、AdamW mu_dtype=bfloat16，因此不是只改变参数更新范围的严格对照。保留两项历史成果，不覆盖、不删除。

## 新对照固定条件

|配置|两项共同设置|
|---|---|
|数据|原七 subtask grouped source-episode holdout，非 whole QC153 数据|
|split SHA256|03f1c8f938121e978773fd437ab395fc6074620e119b3923c386359e6cb71d59|
|norm stats SHA256|5801d57c3c40374d7882ef4edbe2ebfa0e3d9a103ae18cb1fd679e79deb770b9|
|初始化|同一 pi0.5 base，各自重新初始化训练；不续接旧 7500 LoRA|
|训练|40,000 optimizer steps，batch=1，seed=42，单张 A800|
|学习率|warmup500、peak2.5e-5、decay40000、终值1e-6|
|优化器|相同 AdamW 超参数，mu_dtype=bfloat16（一阶动量），grad clip1.0|
|EMA|两项均关闭|
|模型接口|pi05、horizon32、模型 action_dim32、物理16D、三路图像、official_quantile_unclipped|
|验证|同一 heldout 200 个固定观测、validation seed20260924；每2500步验证并保存|

变化项：LoRA 使用 gemma_2b_lora/gemma_300m_lora 和官方 freeze filter；full FT 使用 gemma_2b/gemma_300m，freeze_filter=Nothing，全参数更新。LoRA 的官方可训练集合不应简单称为“只有 adapter”。

各自内部比较7500/10000/20000/40000，检验训练步数；同一步数 LoRA vs full FT 检验更新方式。新 batch=1 和旧 9/26 batch=8 不同，旧7500只能作为历史真机基准，不能把旧7500与新40000的差别全部归于步数。validation 最优与终点40000分别保留；离线 loss 不等于抓取/倒水成功率。

## 路径与队列

远端根目录：
`/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_subtask_compare40k_20261007`

- `lora/code/train.py`、`fullft/code/train.py`：各项训练入口。
- `lora/logs/train.log`、`fullft/logs/train.log`：日志。
- 各自 `logs/report/resolved_run.json`、`validation_metrics.jsonl`、`best_checkpoint.json`、`TRAINING_COMPLETE`：训练元数据与结果。
- 各自 `runs/g1d_subtasks_pi05_compare_<variant>_40k_v1/matched_subtasks_40k/<step>`：检查点。
- `QUEUE_STATUS.json`：当前运行项、训练 PID、排队项，失败时记录退出码；`queue.log` 为队列日志。

单卡顺序：LoRA → full FT；LoRA 失败则队列停止并记录失败，不偷偷更改超参数继续另一项。已修复旧 venv 缺依赖的导入路径，复用历史训练的两层 site-packages；早期启动失败分别发生在模块导入和新 cfg 名称的 assets 查找阶段，无训练更新；已补齐依赖路径与独立 cfg assets，并校验原 norm stats hash。训练队列为 detached 后台进程，不依赖本次 SSH 会话保持连接。

本地 `bootstrap_remote.py` 保留创建流程；它拒绝覆盖已存在的对照目录，不可直接重复执行。此处只创建训练和队列，没有新增测试或真机执行。

后续权重部署保持当前高度、重力补偿、插值、预取和单 subtask 测试条件相同，分别评估抓杯、放食材、拿白杯、扶杯、倒水。训练完成后再为新权重绑定 norm stats/客户端契约，不提前切换 9016。

## 本轮结束时启动状态

队列 PID1439203，LoRA 子进程 PID1439204；已读入183647训练帧，正在准备固定heldout验证观测，尚未确认首个 optimizer step。full FT 在同一队列等待LoRA完成。此处是任务启动状态，不是两项已经训练完成；后续以远端 QUEUE_STATUS.json 和 training_metrics.jsonl 为准。HF/数据缓存已改到共享存储，避免容器 HOME 磁盘空间不足；空 checkpoint 目录使用 --resume 时管理器自动识别无检查点并从基础权重开始，不是续训旧模型。

## TensorBoard 已上线

访问 [训练对照面板](http://100.122.105.65:6024/#scalars)（需要与 kaiyuan 跳板的 Tailscale 地址可达）。同一页面有 lora/fullft 两个 run；选择 Step 横轴比较 train/loss、train/grad_norm、train/seconds、validation/loss、七个 validation/subtask_<index>/loss 与 progress/completed_percent。fullft 当前排队，只有0%进度；开始训练后无需手工操作，指标会自动同步。

镜像脚本每5秒读各自JSONL，已回填已有指标，重启时按tag/step去重，不导入GPU训练运行时。root/tensorboard/lora 与 root/tensorboard/fullft 是事件目录；root/tensorboard_status.json 为镜像状态。writer PID1677350，server PID1688288，A800端口6024；kaiyuan1 Windows SSH forward PID68136，监听 Tailscale IP100.122.105.65:6024。从sys01访问HTTP200且scalars API读到两项run及LoRA训练/验证曲线。

本次查询LoRA step5420/40000（13.55%），最近验证step5000 loss0.0202062；初始化loss0.0844760，step2500 loss0.0229946。实际步数持续增加，以页面和队列状态为准。

重启转发可在 kaiyuan1 Windows 上运行：`ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 100.122.105.65:6024:127.0.0.1:6024 pytorch-ng-2223591`。已有转发运行时不要重复占用6024。

## 2026-10-08：训练完成与访问恢复

远端 QUEUE_STATUS.json 显示两项完成，均有 step40000 训练/验证记录；TensorBoard 同步状态 complete=true。两项 best_checkpoint 都选到40000。LoRA验证loss0.01262825394，全量FT0.01223437281；首步抓黑杯对应subtask0为0.01413115470/0.01428367549，全量FT该阶段并未更低，不能用整体loss替代抓取成功率。训练权重仍位于远端，本次没有替换真机部署。

掉线定位：A800 的6024服务HTTP200且实时镜像健康；kaiyuan跳板的6024监听已消失。已重新建立转发，并用 `C:\Users\kaiyuan1\.ssh\tensorboard_tunnel_watch.ps1` 管理断线重试（5秒后重连、30秒keepalive，单实例mutex）。watcher PID160644，日志 `tensorboard_compare_6024_watch.log`。本地保存同名脚本；此次没有配置开机自启，跳板重启或watcher被关闭后仍需重新启动脚本。恢复后从sys01访问HTTP200，并通过scalar API读到两项40000步、100%及验证指标。访问地址保持 http://100.122.105.65:6024/#scalars 。

## 2026-10-08：追加到总计100k

已建立从各自40000完整train_state恢复到100000的单卡顺序队列，保持原日程末值LR1e-6与所有对照条件。40k检查点及元数据保留；TensorBoard新增lora_to100k/fullft_to100k。当前LoRA续训在启动/数据准备，FT排队。见 [CONTINUE_100K.md](CONTINUE_100K.md)。
