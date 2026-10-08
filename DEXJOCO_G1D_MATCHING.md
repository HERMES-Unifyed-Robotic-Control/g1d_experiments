# G1D ↔ Dexjoco C01 single-decoder / finger-Δ

用户于2026-10-08更正：模型参考为Dexjoco指定分支，而非EFT。

- 仓库：`HERMES-Unifyed-Robotic-Control/Dexjoco`
- 分支：`zbk/c01-single-decoder-finger-delta`
- 固定commit：`90fa71f83c25b1ac0c5c4ba9676464015ba069b8`
- 原流水线：`third_party/dexjoco/experiments/c01_single_finger/README.md`
- 当前真机baseline：官方OpenPI pi05 LoRA step7500，独立于C01。

## 不能直接复用的接口

C01 `afce_all11/codec.py` 的ActionBranch/ActionDecoder按22/44D建模，state为23/46D。G1D不是它的双臂44D动作空间：G1D顺序是left_arm7,right_arm7,left_gripper,right_gripper，全部为绝对qpos。末端位姿与臂关节角之间需要机器人运动学，16指关节与单夹爪电机没有一一映射。原C01 decoder输出层和checkpoint不兼容原生16D。

`prepare.py` 还要求每task恰好100episode、指定仿真camera和v3 meta/episodes parquet；我们的LeRobot v2.1使用episodes.jsonl，共197原episode，并已按原episode划分177/20。原数据准备命令不能原样用于这套G1D数据。

C01机器人响应的FK根位姿/四指尖/16指关节不能由本次G1D的16D记录直接冒充。已提供实测joint_response=`state[t+1:t+31]-state[t]`作为明确命名的G1D响应变体。若研究要求严格保留几何响应，需另外定义并校验G1D FK、基座/升降姿态及夹爪运动学；当前不输出伪指尖或伪触觉标签。

## 本次已交付的组件

|组件|实现|状态|
|---|---|---|
|数据|`g1d_experiments/c01_dataset.py`|原生16D，H30，不跨subtask边界；未来监督与当前观测分离|
|单decoder|`g1d_experiments/c01_decoder.py`|复用上游ActionDecoder架构，state/action16D，E维256，新增头需训练|
|GT-AUX|同上|按左臂7/右臂7/左夹爪1/右夹爪1分组normalized Huber|
|gripper-Δ|同上|仅14/15维；相邻valid掩码；target RMS scale floor0.1；target/scale停止梯度|
|配置与来源|`configs/c01_g1d.yaml`、PROVENANCE|固定参考commit，明确响应变体与未完成环节|
|baseline/holdout|下载、服务、evaluate_subtask脚本|独立OpenPI7500；不将baseline训练结果标为C01|

Decoder构造要求训练集产生的state_mean/std、action_mean/std；不接受上游Dexjoco统计量。输出恢复到G1D物理动作单位。auxiliary_losses输出逐样本GT-AUX和gripper-Δ，外层仍须按原C01的time_weight与ramp组合到FM(E)；本组件不生成FM(E)，也不提供已训练π。

## 完整G1D C01训练仍需完成

1. 用G1D训练集真实图像提取DINO特征、世界变化目标；PCA及所有统计只从train拟合。相机/机器人掩码缺失要标记缺失，不能当成有效零值。
2. 确定G1D响应协议：本交付的joint-response变体，或补齐有依据的FK几何响应。修改C01 E的action/robot/response分支到原生16D协议，并重新训练G1D E。
3. 导出G1D E cache与新decoder；保留原episode split，记录所有资产哈希。
4. 将`pi_bridge_joint.py`、`readout_alignment.py`、`single_finger_alignment.py`里的22/44、23/46、finger slices与双头selection改为单一G1D16D。JAX导出/推理头必须与训练decoder一致。
5. 用G1D数据重新校准GT-AUX，不把上游0.25628781345139295当成已验证的G1D权重；gripper-Δ的0.05也须评估。
6. 训练G1D C01 π；在相同holdout和真实起始条件下与7500baseline比较，记录人工切换/介入、progress/10与成功率。

本次完成数据和decoder/损失组件交付，没有运行上述新E/π训练、组件测试或真机评测。上游11任务51.64%等结果属于仿真任务与原检查点，不是G1D结果。

图像适配保留RGB、224×224 PIL bilinear resize-with-pad，与当前OpenPI客户端一致；C01 backbone自身normalization仍需按其协议执行。

## 2026-10-08后续实现

由本项目自行完成适配、E/π训练与评测。新增native16 codec、单步实测关节响应、视觉目标准备、E训练与导出、JAX单decoder、π联合训练/校准和holdout动作评估代码。上述“仍需完成”清单中的代码入口已补齐；训练执行、GPU数值一致性与真实评测尚未完成。实际命令及DINOv3资产阻塞见 `experiments/g1d_c01_20261008/RUN_C01.md`。
