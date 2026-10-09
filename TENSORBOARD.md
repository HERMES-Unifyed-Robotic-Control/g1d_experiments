# C01训练进度与文件迁移监控

## 当前部署（2026-10-09）

- A800 C01 E/π：<http://100.74.87.113:6008>。
- A100双实例 C01 π：<http://100.74.87.113:6009>。资产迁移完成、训练开始后才有曲线。
- sys01转发：`tmux attach -t g1d_tb_forward`。
- A800或A100的817588xt25上，事件导出与TensorBoard：`tmux attach -t g1d_c01_tb`。
- `Ctrl+b`，然后`d`，退出查看并保留后台进程；不要用Ctrl+c停止服务。
- 转发具备断线重连；tmux不保证主机重启后自动恢复。转发密码只在进程内存中，不保存到文件。

A800 run目录为`$HOME/HDD_POOL/yangky/g1d_c01_20261008/run`；A100 run目录为`$HOME/HDD_POOL/yangky/g1d_c01_20261009/run`。训练JSONL每10步记录，事件导出每10秒扫描。图表包含train/validation的loss、effect_loss、gtaux、gripper_delta，以及train/grad_norm；E包含action/response/world。

## 复用事件导出

使用已有训练venv（当前TensorBoard 2.16.2）。事件导出不导入JAX或Torch，不占用GPU，不重启训练。

```bash
# 替换C01_ROOT、PYTHON为相应主机路径
mkdir -p "$C01_ROOT/monitoring"
tmux new-session -d -s g1d_c01_tb -n metrics \
  "$PYTHON -u scripts/jsonl_to_tensorboard.py --run-root $C01_ROOT/run --logdir $C01_ROOT/tensorboard --stages E pi_lora > $C01_ROOT/monitoring/export.log 2>&1"
tmux new-window -t g1d_c01_tb -n server \
  "$PYTHON -m tensorboard.main --logdir $C01_ROOT/tensorboard --bind_all --load_fast=false --port 6008 > $C01_ROOT/monitoring/server.log 2>&1"
```

A100全量FT使用`--stages pi_fullft_dp2_100k`和端口6009。启动命令用于新部署；当前服务已经运行，请不要重复启动。普通加载模式避开本环境快速加载的曲线请求超时。回填时同一步采用JSONL最后一条记录；后续发现回退/恢复训练时写入TensorBoard session start以清除该步之后的旧曲线。

## 在任一A100上查看文件传输

```bash
C01_ROOT=$HOME/HDD_POOL/yangky/g1d_c01_20261009
# 汇总状态，以及自动重试控制器心跳
cat "$C01_ROOT/ASSET_SYNC_STATUS.json" "$C01_ROOT/AUTO_SYNC_RETRY_STATUS.json"
# 连续日志（Ctrl+c只退出日志查看）
tail -f "$C01_ROOT/asset_sync_auto.log" "$C01_ROOT/sync_supervisor.log"
```

迁移脚本现已每5秒更新心跳，按实际完成顺序累计SHA256校验通过的文件，active_files记录正在写入的各文件、字节数和最后收到数据的时间。bytes仅表示已校验字节，written_bytes_including_unverified包含未校验的在途字节。下面每10秒显示汇总和正在写入的大文件大小（bytes），不依赖watch软件：

```bash
C01_ROOT=$HOME/HDD_POOL/yangky/g1d_c01_20261009
while true; do
  date
  cat "$C01_ROOT/ASSET_SYNC_STATUS.json"
  find "$C01_ROOT/assets" -name '*.transfer' -printf '%s %p\n'
  sleep 10
done
```

`.transfer`是未完成或未校验的临时文件，其大小增长说明正在写入，不代表SHA256校验通过。最终以`state: complete`为准。两rank状态与训练日志：

```bash
cat "$C01_ROOT/run/RANK0_STATUS.json" "$C01_ROOT/run/RANK1_STATUS.json"
tail -f "$C01_ROOT/run_rank0_auto.log" "$C01_ROOT/run_rank1_auto.log"
```

## 传输指标曲线

A100页面选择run `asset_transfer`。每5秒记录`transfer/verified_percent`、`transfer/inflight_GB`、`transfer/retry_attempt`和状态年龄等。训练曲线只有真正进入训练后才有；传输指标不是模型loss。脚本为`scripts/transfer_to_tensorboard.py`，部署在817588xt25的`g1d_c01_tb:transfer_metrics`窗口。

## 40k续训100k

A800同一6008入口选择`pi_lora_100k/train`与`pi_lora_100k/validation`；旧40k历史已回填，后续日志增加学习率。续训协议及自动评测见[RUN_CONTINUE_100K.md](experiments/g1d_c01_20261008/RUN_CONTINUE_100K.md)。A100已切换为policy全量FT100k，全局batch2，选择`pi_fullft_dp2_100k/train`与`pi_fullft_dp2_100k/validation`，验证每2500步记录。旧`pi_lora_dp2_40k`曲线保留用于追溯，已经停止训练。
