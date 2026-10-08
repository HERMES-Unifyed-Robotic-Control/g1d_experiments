# G1D C01 E/π训练与评测

2026-10-08。由本项目自行完成G1D适配和训练；当前baseline仍为7500。原生16D关节响应为首轮建议方案；它是C01的G1D适配变体，不是原Panda+Allegro FK目标的直接复用。

## 已执行

- 读取全部885 train/100 validation片段；原episode177/20，无交叉。
- 验证state/action都是16D，顺序一致，frame/timestamp连续且30Hz。
- 统计量只使用训练集；文件见statistics.json，指纹与实际维数见STATE_CONTRACT.json。

## 状态和阻塞

DINOv3 HF源`facebook/dinov3-vitb16-pretrain-lvd1689m`为gated repo。当前账号获取config返回403，本机只找到refs索引，没有实际权重。需要已有权重目录或授予访问权限，才能准备真实视觉目标。未用随机DINO、其他模型或伪世界特征代替。

E/π代码入口已实现但尚未执行；GPU依赖、JAX decoder数值匹配、恢复和新模型的真机能力均未验证。没有新C01 loss或成功率结果。当前两张4090由推理服务占用；正式训练需可用GPU（建议A800），不自动停止现有服务。

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
export C01_DINO=/absolute/path/dinov3-vitb16
export PI05_BASE=/absolute/path/pi05_base_jax/params
export NORM_STATS=/absolute/path/baseline7500/assets/g1d_subtasks_train_v1/norm_stats.json
export PYTHONPATH="$PWD:$PWD/vendor/openpi/src:$PWD/vendor/openpi/src/openpi/openpi-client/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python scripts/prepare_c01_statistics.py --train "$TRAIN" --validation "$VALIDATION" --output "$C01_RUN/normalization"
CUDA_VISIBLE_DEVICES=0 python scripts/prepare_c01_visual.py --train "$TRAIN" --validation "$VALIDATION" --dino "$C01_DINO" --output "$C01_RUN/evidence"
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

holdout曾用于选checkpoint，不是独立测试集。C01是H30，baseline是H32；动作误差对比需统一anchor与前30步、噪声/去噪协议，不能直接将不同horizon的汇总MAE当作严格同条件结果。真机评测还需为C01新契约增加client/server profile和H30接入；在此之前不能把C01输出接到H32 baseline服务。不得自动执行真机动作。
