# A800 C01 π LoRA40k → 100k续训

本轮续训π/policy，固定E30k及其标签缓存。从完整40000 checkpoint恢复模型、optimizer和sample RNG，累计到100000（额外60000）步；batch1、H30/native16、GT-AUX1.3227095743和gripperΔ0.05保持原配置。

原学习率是500步warmup、peak2.5e-5、40000步cosine衰减到1e-6。此次不重新warmup、不重新拉伸原曲线；40000之后保持连续的1e-6。decoder更新继续乘0.25。原40k推理服务继续使用原检查点，训练在独立pi_lora_100k目录。

## 目录与队列

A800根目录：`$HOME/HDD_POOL/yangky/g1d_c01_20261008`。

- 原40k：`run/pi_lora`。
- 续训：`run/pi_lora_100k`；40000 checkpoint通过硬链接复用，保留原40k结果。
- 元数据：`run/pi_lora_100k/CONTINUATION.json`。
- 训练与后续评测：`tmux attach -t g1d_c01_100k`。
- 日志：`tail -f "$C01_ROOT/run/logs/pi_lora_100k.log"`。

每2500步验证并保存。100k结束后在续训已提交checkpoint中按heldout composite loss选择，自动评测相同200观测/H30/native16，并与baseline7500前30步比较。holdout曾用于选择，并非独立测试集；不能把预算7500/100000不同的结果称为同预算架构对比。

## TensorBoard

- A800：http://100.74.87.113:6008，选择`pi_lora_100k/train`或`pi_lora_100k/validation`；前40k历史已回填，续训新增`train/lr`。
- A100：http://100.74.87.113:6009，选择`pi_lora_dp2_40k/train`；首次validation在2500步后出现。
- sys01转发：`tmux attach -t g1d_tb_forward`；Ctrl+b之后d保留后台。

A100当前训练π/policy LoRA40k，复用相同E30k、decoder初始化与G1D数据。全局batch2，每个实例一张A100、每卡1样本，同步梯度；未重新训练E。同样40k，其训练样本预算约为A800 batch1的两倍，因此也不是严格同预算对比。
