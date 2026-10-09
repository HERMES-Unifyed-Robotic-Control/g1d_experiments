# G1D C01 E/π训练与评测

2026-10-08。由本项目自行完成G1D适配和训练；当前baseline仍为7500。原生16D关节响应为首轮建议方案；它是C01的G1D适配变体，不是原Panda+Allegro FK目标的直接复用。

## 已执行

- 读取全部885 train/100 validation片段；原episode177/20，无交叉。
- 验证state/action都是16D，顺序一致，frame/timestamp连续且30Hz。
- 统计量只使用训练集；文件见statistics.json，指纹与实际维数见STATE_CONTRACT.json。

## 状态和阻塞

DINOv3已通过Meta邮件授权下载原生 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`，342860279 bytes，SHA256 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。不依赖HF gated访问。原生加载使用官方Meta源码commit `6876159a11b4df116f30f667f8c9888617df0751`，strict state_dict加载，ImageNet mean/std、224px、196×768 patch tokens。

A800运行目录：`/HOME/sysu_xdliang/sysu_xdliang_2/HDD_POOL/yangky/g1d_c01_20261008`。已完成全部885 train/100 validation真实DINO/PCA/world目标；E30k已开始训练并输出loss，π训练及最终比较尚未开始。首次发现A800 OpenCV未编译FFmpeg，已安装PyAV16.1.0并改用PyAV精确帧解码。顺序队列为E30k→export→GT-AUX calibration→π LoRA40k→固定holdout评测。查看 `run/PIPELINE_STATUS.json`、`run/logs/*.log`、`logs/pipeline.log`。过程尚未完成，不能宣称有新的C01成功率或优于baseline的结果。

新服务 `scripts/serve_c01.py` 与独立 `deploy/c01_client/` 使用H30/native16契约，保留baseline的高度、重力补偿、30Hz、异步预取/平滑配置。尚未执行真机动作，当前运行的baseline服务未改动。

## state与模型维数

|对象|维数|含义|
|---|---:|---|
|原始G1D state/action|16|左臂7，右臂7，左夹爪1，右夹爪1；绝对qpos|
|E action encoder|16→64|G1D物理动作经过train-only标准化|
|E robot encoder|16→64|相邻实测关节变化，单步Δqpos；非累计位移|
|E latent|30×256|C01 Query-ASV与TemporalEffect结构|
|E/π decoder输入state|16|当前未归一化G1D state，内部用train-only mean/std|
|π语言中的state|16|baseline的train quantile规范化后离散化，先tokenize后padding|
|π Observation.state|32|16D规范化state+16个padding；不是Dexjoco46D，也不是E256D|
|π flow输出|30×256|预测E；不直接当作机器人动作|
|机器人最终动作|30×16|同一joint decoder恢复物理qpos|
|原Dexjoco state/action|23/22或46/44|末端位姿+16指关节/手，不能直接映射|

升降和腰部不在16D内；夹爪电机q不是夹爪米制开口。位置/旋转矩阵损失不能套在G1D关节角切片上。G1D分组改为7/7/1/1；finger-Δ改为夹爪14、15两维。

## 运行顺序

在仓库根目录，使用相应Torch CUDA或JAX CUDA环境。Core版本见README；E准备还需要支持DINOv3的Transformers，建议与π JAX环境分开，避免NumPy/视频依赖冲突。下列路径按目标机器改为绝对路径。

```bash
export TRAIN=/absolute/path/g1d_subtasks_official_train_v1
export VALIDATION=/absolute/path/g1d_subtasks_official_validation_v1
export C01_RUN=/absolute/path/g1d_c01_run
export C01_DINO=/absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
export C01_DINO_SOURCE=/absolute/path/facebookresearch-dinov3-source
export PI05_BASE=/absolute/path/pi05_base_jax/params
export NORM_STATS=/absolute/path/baseline7500/assets/g1d_subtasks_train_v1/norm_stats.json
export PYTHONPATH="$PWD:$PWD/vendor/openpi/src:$PWD/vendor/openpi/src/openpi/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/prepare_c01_statistics.py --train "$TRAIN" --validation "$VALIDATION" --output "$C01_RUN/normalization"
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_c01_visual.py --train "$TRAIN" --validation "$VALIDATION" --dino "$C01_DINO" --dino-source "$C01_DINO_SOURCE" --output "$C01_RUN/evidence"
CUDA_VISIBLE_DEVICES=0 python scripts/train_c01_e.py --train "$TRAIN" --validation "$VALIDATION" --evidence "$C01_RUN/evidence" --statistics "$C01_RUN/normalization/statistics.json" --output "$C01_RUN/E" --steps 30000
# 恢复E：同一命令增加 --resume
CUDA_VISIBLE_DEVICES=0 python scripts/export_c01_e.py --checkpoint "$C01_RUN/E/best.pt" --train "$TRAIN" --validation "$VALIDATION" --evidence "$C01_RUN/evidence" --output "$C01_RUN/cache"

# 以下改用JAX环境，仍从官方pi05 base初始化，不使用7500作为base。
CUDA_VISIBLE_DEVICES=0 python scripts/train_c01_pi.py --cache "$C01_RUN/cache" --train "$TRAIN" --validation "$VALIDATION" --base-params "$PI05_BASE" --norm-stats "$NORM_STATS" --output "$C01_RUN/calibration_lora" --variant lora --calibrate-only
export GTAUX_WEIGHT=$(python -c 'import json,os; print(json.load(open(os.environ["C01_RUN"]+"/calibration_lora/calibration.json"))["chosen_gtaux_weight"])')
CUDA_VISIBLE_DEVICES=0 python scripts/train_c01_pi.py --cache "$C01_RUN/cache" --train "$TRAIN" --validation "$VALIDATION" --base-params "$PI05_BASE" --norm-stats "$NORM_STATS" --output "$C01_RUN/pi_lora" --variant lora --gtaux-weight "$GTAUX_WEIGHT" --steps 40000
# fullft：独立calibration_fullft与pi_fullft目录，--variant fullft，并使用对应校准权重。
# 恢复π：同一命令增加 --resume；保存params、optimizer、样本游标/RNG。
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_c01_pi.py --params "$C01_RUN/pi_lora/checkpoints/40000/params" --cache "$C01_RUN/cache" --validation "$VALIDATION" --norm-stats "$NORM_STATS" --variant lora --output "$C01_RUN/evaluation"
```

E采用seed42、batch32、30k、前20k1e-4后3e-5，AdamW beta2=.99，响应/视觉权重由.05到.2 ramp；π当前便携入口单GPUbatch1、LoRA或fullft、无EMA、decoder update scale=.25、GT-AUX ramp10k、gripper-Δ默认.05（待实测）。每2500步验证/保存π；E每1000步验证。训练指标写入JSONL，未启动TensorBoard进程。

E视觉目标当前只取egocentric，与上游单视图world协议对应；π输入仍为三路RGB。PCA/E normalization/robot normalization仅从train拟合。缺少机器人mask使用明确标记的legacy-none协议；不能声称有实测分割。

## 评测解释

C01离线入口推理前删除全部未来监督，只传当前图像、tokenized16D state、32D padded state、prompt。输出必须为H30×16D物理qpos。

holdout曾用于选checkpoint，不是独立测试集。C01是H30，baseline是H32；动作误差对比需统一anchor与前30步、噪声/去噪协议，不能直接将不同horizon的汇总MAE当作严格同条件结果。已加入独立C01 H30服务/client profile。训练完成后需先完成离线评测、server/client契约检查，再由用户启动真机执行。不能把C01输出接到H32 baseline服务。

## baseline7500 离线比较

`evaluate_c01_pi.py`直接读取仓库 `data/validation_samples.json` 的200个固定anchor；噪声采用与baseline相同的逐观测 `numpy.default_rng(20260924+i)` 标准正态协议，但latent维数不同，不能声称噪声张量相同。默认10次去噪。`compare_c01_baseline.py`检查固定anchor、task、物理GT逐值一致，只比较双方前30步，并按两臂、两夹爪及task汇总。

```bash
python scripts/compare_c01_baseline.py --baseline /path/to/baseline7500/heldout_predictions.npz --c01 "$C01_RUN/evaluation/predictions.npz" --output "$C01_RUN/comparison/baseline7500_vs_c01.json"
```

7500 baseline和C01 40k存在训练预算差异；后续可另加相同步数π0.5作为消融。原200观测的baseline产物来自 `pi05_official_holdout`，checkpoint路径以 `/7500` 结尾；复制时保留SHA256 provenance。

## 训练后的真机入口（尚未启动）

```bash
python scripts/serve_c01.py --checkpoint /path/to/pi_lora/checkpoints/SELECTED_STEP --cache "$C01_RUN/cache" --port 9021
```

服务启动成功后，将 `deploy/c01_client/`（包括更新后的profile JSON）同步到G1D独立目录，并使用同一版本openpi-client图像处理代码。保持已有baseline目录作为对照。首先执行 `./run_client_on_g1d.sh --check-server-only`，成功后再使用用户已有的环境变量启动格式和 `--execute`；C01端口为9021。profile中的checkpoint占位在实际服务加载完成前不可用于执行。

## A800 数据一致性修复

首次视觉准备已完成全部885个train episode；validation episode36解码失败。对全部3952个视频/parquet/meta文件进行SHA256核对，仅validation36 egocentric视频存在传输截断：A8002588672 bytes，本地完整3208496 bytes。完整原文件SHA256为 `1cbeb2bd68106bba514855cdd63b8cbd891b5db6fd7afd82070ece7ffefb782d`。

已建立运行目录下独立 `data/` 硬链接副本；修复文件通过原子替换解除共享inode，未修改baseline数据目录。其余3951文件SHA256一致，修复文件SHA256已复核。原统计量SHA256也与sys01完全一致。运行目录保留 `INPUT_CHECK_BEFORE_REPAIR.json`、`INPUT_CHECK.json` 和 `DATA_RELOCATION.json`，明确记录缓存复用和路径迁移依据。

## 2026-10-09 进度检查

E30k于10月8日22:50（Asia/Shanghai）完成，最佳step30000。200观测验证loss=0.1000007024，action=0.0003459838，response=0.0014352506，world=0.4968383479。这是E的归一化重建/监督损失，不是π闭环成功率。E缓存于22:54完整导出。

队列于22:55因π环境缺augmax停止，π当时没有开始训练。10月9日补齐实际入口需要的augmax、beartype、Orbax辅助依赖、Transformers/tokenizers、numpydantic/Pydantic等，并恢复校准；不会重训E。PaliGemma tokenizer沿用已有本地模型，SHA256与sys01 baseline一致：`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。启动与恢复脚本保存在A800运行根目录 `resume_pipeline.sh`，失败阶段会写入 `PIPELINE_STATUS.json`，不能把队列已提交误认为π已训练完成。

2026-10-09 09:38（Asia/Shanghai）：GT-AUX 校准已通过，16个训练样本输出梯度比的中位数为0.0756023861，所选权重为1.3227095743；校准不执行优化器更新。队列已进入π LoRA 40000步阶段，模型已加载，正在等待首次编译/实际训练指标。最终holdout评测和baseline7500对比尚未完成。

2026-10-09：A100 pytorch-ng-992446有240GiB RAM（available232GiB）、2×A10080GB，容器盘160GB空闲且HOME/workspace实写12MiB成功；指定Storage01持久化目录仍报EDQUOT，按用户指示继续在原A800运行。A800 π首次达到2500，200观测validation loss1.3847066134，但Orbax0.11.1缺CommitFutureAwaitingContractedSignals导致首个checkpoint未提交，无可可靠续训的cursor。已安装与sys01一致的Orbax0.11.13，失败产物保留为run/pi_lora_failed_save_20261009_2500，重新开始π；E和缓存、校准结果全部复用。最终评测及真机部署尚未执行。
