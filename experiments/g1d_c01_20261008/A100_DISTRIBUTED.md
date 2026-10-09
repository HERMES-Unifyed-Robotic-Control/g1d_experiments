# G1D C01 双实例 A100

两台实例 `pytorch-ng-8174759` 和 `pytorch-ng-817588xt25` 使用 `sysu_xdliang_4`。每个实例只暴露一张 A100-SXM4-80GB，GPU UUID 不同；不能在任意一台上直接启动两个本地 GPU 进程。

共享实验根目录：

```text
/HOME/sysu_xdliang/sysu_xdliang_4/HDD_POOL/yangky/g1d_c01_20261009
```

该目录解析到 `/XYAIFS00/HDD_POOL/sysu_xdliang/sysu_xdliang_4/yangky/g1d_c01_20261009`。两实例均成功写入并 fsync 32MiB；`Storage01` 的 16MiB探测在写入8MiB后报 EDQUOT，故本实验选用 HDD_POOL。32MiB探测不证明大规模配额充足，迁移状态另行记录。

## 训练协议

- 复用 A800 已完成的 E30000、E缓存、训练/验证划分、native16规范化和 GT-AUX 校准；不重训 E、不改变 subtask prompt。
- 原始状态为双臂14关节加2夹爪的16D；策略状态填充到32D；E为30×256；解码动作仍为30×16。
- JAX0.5.3、Flax0.10.2、Orbax0.11.13、Optax0.2.5；NGC系统PyTorch2.7.0a0仅用于读取数据等。额外依赖LeRobot0.3.3；旧 `requirements-baseline-sys01.txt` 的LeRobot0.1.0没有所需Dataset API，不能作为这台机器的完整锁文件。
- 两个进程进行数据并行，每进程一个样本，全局batch=2，π LoRA40k。A800保留batch=1的独立主实验。全局batch不同，40k的样本预算分别为80k/40k，不能把差异单独归因于硬件加速。
- 每2500步共同保存检查点，rank0写日志和cursor；E mmap缓存最多64个episode，启动软文件句柄上限65536。
- 最终按已提交检查点的heldout composite loss选择权重，以相同200观测、H30/native16动作误差比较baseline7500。不是独立测试集，也不代表真机成功率。

## 启动与评测入口

代码位于共享根目录 `code`。资产逐文件SHA256验证，`ASSET_SYNC_STATUS.json`必须为complete，且环境检查生成`ENVIRONMENT_READY.json`后，启动器才会进入训练。

在8174759运行：

```bash
C01_ROOT=$HOME/HDD_POOL/yangky/g1d_c01_20261009
bash "$C01_ROOT/code/scripts/run_c01_distributed.sh" "$C01_ROOT" 0 192.168.60.48:29573
```

在817588xt25运行同一命令，rank改为1。IP为本次实例的pod IP，作业重建后须重新查看 `hostname -I`，两进程使用同一协调地址。保留启动器日志，不能重复启动仍在运行的rank。

查看 `run/RANK{0,1}_STATUS.json`、`run_rank{0,1}.log`、`run/pi_lora_dp2_40k/training_metrics.jsonl`、`validation_metrics.jsonl`。rank0随后自动运行离线评测和baseline比较，结果在 `run/comparison/baseline7500_vs_c01_dp2.json`。

2026-10-09实测：Torch NCCL两GPU归约成功；JAX实际发现全局2GPU/每进程1GPU，分片输入和全局梯度更新成功（两rank均loss10、更新后weight1）。这属于通信和训练基础设施验证，尚不等于完整C01已完成训练。
