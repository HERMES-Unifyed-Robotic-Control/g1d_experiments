# A100双实例 C01 policy全量FT100k

A800继续独立的LoRA40k→100k；A100两个单GPU实例共同训练一个全量FT policy。两个实例各有一张A100 80GB，以JAX全局batch2进行数据并行。E固定为同一个E30k，不重新训练E。

## 固定协议

- 相同官方subtask训练/验证episode划分、E缓存、state归一化、实测16D关节响应监督；state16D补齐32D，E30×256，动作30×16。
- 从π0.5预训练参数和同一E30k decoder初始化；全量更新骨干、E投影和decoder。新建输出目录，不把LoRA checkpoint当作全量FT恢复点。
- 100000步，global batch2，每实例1样本；seed42，AdamW（mu bfloat16），无EMA。
- 学习率warmup500、峰值2.5e-5，cosine衰减到100000步的1e-6；decoder更新比例0.25。
- 复用原GT-AUX标定权重1.3227095743492492（原标定基于LoRA），gripper delta权重0.05，ramp10000。
- 每2500步保存和验证相同200个heldout观测；按已提交检查点的heldout composite loss选择，再比较baseline7500前30步。heldout用于选点，不是独立test。
- A800 LoRA是40k后以1e-6续训，global batch1；FT从头100k且batch2，因此训练样本量与学习率计划不同，不能称为严格等预算消融。

## 运行与查看

```bash
C01_ROOT=$HOME/HDD_POOL/yangky/g1d_c01_20261009
# 两个实例分别运行rank0、rank1（已有tmux任务时勿重复启动）
bash "$C01_ROOT/code/scripts/run_c01_fullft_distributed.sh" "$C01_ROOT" 0 192.168.60.48:29519
bash "$C01_ROOT/code/scripts/run_c01_fullft_distributed.sh" "$C01_ROOT" 1 192.168.60.48:29519
# 当前训练
$HOME/bin/tmux attach -t g1d_c01_fullft100k
# 查看日志
cat "$C01_ROOT/run/FULLFT_RANK0_STATUS.json" "$C01_ROOT/run/FULLFT_RANK1_STATUS.json"
tail -f "$C01_ROOT/run_fullft_rank0.log" "$C01_ROOT/run_fullft_rank1.log"
tail -n 1 "$C01_ROOT/run/pi_fullft_dp2_100k/training_metrics.jsonl"
```

TensorBoard：<http://100.74.87.113:6009>，选择`pi_fullft_dp2_100k/train`及`pi_fullft_dp2_100k/validation`。A800 LoRA续训：<http://100.74.87.113:6008>，选择`pi_lora_100k/train`。

输出`run/pi_fullft_dp2_100k`；原A100 LoRA输出`run/pi_lora_dp2_40k`停止并保留。现有C01 LoRA40000推理服务9021继续使用原检查点。
