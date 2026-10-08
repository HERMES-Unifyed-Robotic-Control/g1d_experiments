# G1D 数采、推理几何与 subtask 切换审计（2026-10-07）

本次只读读取 G1D 文件和本地源码，复制已有 rollout 并做离线分析。没有启动机器人控制、改变高度/腰部目标、修改部署程序、重启服务或训练模型。当前机器人控制器、数采入口和 IK 文件与 sys01 对应源码 SHA256 一致。历史数采时使用的完整源码版本未在 episode 内保存，不能将当前源码一致视为历史版本已核实。

## 后续实施更新（2026-10-07）

以下审计正文保留原始运行状态，描述的是实施之前的客户端。按用户后续要求，已另建并部署实测高度、数采重力补偿和异步平滑客户端，恢复 9/26 step7500 基准到 9016；9019 原 197 模型仍在运行，9020 QC 暂停。当前只读高度为 0.065664 m；180 步无动作 dry-run 为 28.45 Hz，没有执行真机动作。重力补偿模型与数采缓存在 128 组姿态的力矩差为 0 Nm。菊花默认维持示教对应的 0.065 m，0.1 m 尚需对应的启动/停止权限。具体配置、证据与启动命令见 [RUN_ALIGNED.md](../g1d_client_alignment_20261007/RUN_ALIGNED.md)。

## 主要结论

目前可以确认模型接口与数据表示大体对齐，但物理工作空间与控制方式尚未全面对齐。统一使用 0.065 m 的高度只与菊花数据的数值相符；红枣、柠檬数采高度明显更高。数采发送重力补偿前馈力矩，推理发送零前馈力矩。今天的两个 joint rollout 都是菊花，所以不能只用其他两类数据的高度差解释首步失败。

9/26 对照记录确实包含机器人夹持并把黑杯从右侧移至中间的过程；它与今天运行的训练方式、prompt 和高度锁定配置不同。旧日志缺少实测升降高度，不能直接断言成功来自更高的躯干。当前日志也没有实际高度、腰部姿态、真实桌高/末端离桌高度，不能把配置目标视为已测量的物理状态。

## 核查的运行

证据：各目录下 `resolved_config.json`、`events.jsonl`、`rollout_actions.npz`、输入视频。完整数值见 `audit_metrics.json`。

|运行|模型|prompt|高度配置|实际动作 waypoint 平均频率|
|---|---|---|---|---:|
|9/26 11:39:04|9016，7-subtask JAX LoRA，step 7500|subtask，起始 index 0，手动切换|未记录高度锁参数|16.46 Hz|
|9/26 16:04:52|同上|subtask，起始 index 0|未记录高度锁参数|4.37 Hz，存在很长推理等待|
|9/27 17:17:16|同上|subtask|0.065 m|27.70 Hz，async_prefetch 开启、额外动作平滑开启|
|9/30 23:30:22|9018，菊花 subtask full FT，step 36500|subtask|0.065 m|15.49 Hz|
|10/7 15:48:36|9019，197 episode joint full FT，step 36000|whole|0.065 m，高度锁开启|16.53 Hz|
|10/7 15:56:03|9020，QC-balanced 153 episode joint full FT，step 31000|whole|0.065 m，高度锁开启|16.61 Hz|

9/26 11:39 与今天两次的 `execute_chunk_steps=8`、`async_prefetch=false`、`disable_action_smoothing=true`、`max_arm_delta=0.3`、`max_gripper_delta=1.0`、`initial_state=current`、`go_ready_before_start=true`、`fps=30`、`network_interface=eth0` 相同。9/26 原始配置没有 `interpolation_substeps`、`chunk_blend_steps`、`fixed_height_m` 字段，`prefetch_remaining_steps=4`；9/27 优化版本才记录 4 个插值子步、2 步边界混合、预取 8 步、高度 0.065 m。9/26 客户端 SHA 为 9d7695ce...，9/27 为 465313b9...，今天 joint 客户端为 f9d5f33e...。2026-10-07 补查时更正了之前把部分 9/27 参数误写成 9/26 参数的表述。

197 是总 episode 数，训练/验证为 157/40；QC 153 是总 episode 数，训练/验证为 123/30。QC checkpoint 编号 0–152 对应 153 个 episode。

## 数采高度与物理控制

以下高度来自已有 raw 数据审计 JSON，并非本次重新读取原始 raw episode（原路径已不在 sys01）。本次还核对了每个来源的 `meta/tasks.jsonl`：1 是菊花，2 是红枣，3 是柠檬。

|来源/任务|数采升降高度 min–max|数采平均高度|今天统一目标与均值之差|
|---|---:|---:|---:|
|food_sorting_1 / 菊花|0.06422–0.06651 m|0.06509 m|-0.00009 m|
|food_sorting_2 / 红枣|0.14888–0.15260 m|0.14966 m|-0.08466 m|
|food_sorting_3 / 柠檬|0.14042–0.16149 m|0.14506 m|-0.08006 m|

数据里的升降速度几乎为零，表示各个 episode 多数保持高度；不表示各批次的高度相同。`0.065 m` 是升降机构状态/目标的米制数值，不是手臂或夹爪离桌 6.5 cm。末端离桌高度还取决于桌面、躯干、腰部、关节角与杯子摆放。

LeRobot 转换只提取左右臂 qpos 与左右 Dex1 qpos，共 16D；升降、腰部和底盘状态没有进入模型 state。图像提供几何线索，但没有显式高度状态或高度动作供模型控制升降。因此不能假定模型会稳定补偿 8 cm 的物理工作空间差异。

当前推理代码 `main()` 中“official demonstrations 都是 65 mm”的注释与 raw 数据审计不符。`launch_g1d_client.py` 对所有 official 任务设置 0.065 m，并没有按 TASK 区分。它也没有实现 `FIXED_HEIGHT_M` 环境变量覆盖；只在命令前增加这个环境变量不会改变当前启动器的高度参数。

数采链路：`teleop_hand_and_arm.py` 调用 `arm_ik.solve_ik()` 返回 `sol_q, sol_tauff`，随后 `ctrl_dual_arm(sol_q, sol_tauff)`。`robot_arm_ik.py` 使用 `pin.rnea(model, data, q, 0, 0)`，是静态重力补偿。

推理链路：ready pose 和每个动作插值 waypoint 都调用 `ctrl_dual_arm(q, np.zeros(14))`。因此相同关节目标也可能产生不同的实际跟踪偏差。历史部署备份亦使用零前馈，所以这一差异是数采与部署的共同问题，不能作为新 joint 模型退化的独立证明。

今天两次在下个 chunk 开始时，相对上一条已发送目标，右肩 pitch 的平均绝对误差约 0.035/0.042 rad，右肘约 0.010/0.024 rad。这个测量发生在推理等待之后，较即刻记录更接近保持目标时的误差，但仍不是严格静态实验。它说明有偏差，也说明不能把整个轨迹异常都归因于电机完全不跟随。转换成末端厘米偏差需要 FK 与实际桌面几何。

`FixedHeightLock.start()` 只检查控制子进程没有立即退出，没有等待实测高度进入容差。腰 yaw 启动检查容差 0.03 rad、超时 3 s，超时仅打印 warning；这些均不代表任务完成判据。

## 已对齐及仍有差异的配置

|项目|检查结果|
|---|---|
|state/action|两侧都是 left_arm_7,right_arm_7,left_gripper,right_gripper，16D 物理量，绝对 qpos，单位 rad；模型填充到 32D 后输出裁回 16D|
|归一化/权重|今天两个 rollout 各自 server checkpoint、stats、profile、客户端源码 hash 对应；未看到选错端口或旧 norm_stats 混用的证据|
|相机输入|训练采用 color_0/color_2/color_3，推理采用 head 的左目、配置中的 left/right wrist；RGB、PIL resize_with_pad 到 224，image_tools hash 一致。样例图的杯子所在视图对应，没有看到直接对调的证据；相机外参/桌高没有标定复核|
|初始化|固定 READY_ARM_Q/READY_GRIPPER_Q 后读 current 实测状态；today 实测臂关节与固定 ready 最大差约 0.040 rad。ready 是历史分布中的代表值，不是每个任务、每个 episode 的真实统一起点|
|腰部|数采记录 pitch 状态接近 0，推理腰 pitch 目标初始化为 0、yaw 固定为 0；实际整段腰姿态未进入 rollout 记录|
|前馈力矩|数采有重力补偿，推理为 0，未对齐|
|频率|训练/数采约 30 Hz，今天执行 chunk 内按 30 Hz，但同步等待使整体约 16.6 Hz，未连续对齐|
|动作处理|额外 EMA smoothing 禁用，但 chunk 边界前两步混合、4 个插值子步、底层 Dex1 滤波/限速仍存在；“disable_action_smoothing”不表示完全没有动作处理|
|高度|菊花目标符合 raw 数值；红枣/柠檬统一 0.065 m 不符合 raw 数值；所有任务都缺少实测高度证据|

训练示教初始、9/26、今天相机对照见 `initial_camera_comparison.jpg`，采用与部署相同的 resize_with_pad。图像可以看到杯子位置与腕部视角差异，但不能仅据单张图推算升降高度或判定它们是失败原因。

## 首步抓杯的直接证据

9/26 11:39:04：右夹爪目标在约 29.83 s 开始小于 4.5 rad，视频约 30–38 s 出现夹持、黑杯从右向中间移动、松开，随后 41.66 s 按 N 切至放菊花，110.26 s 再按 N 切至左手拿白杯。见该目录的 `cup_transition.jpg`。这里没有末端离桌净高度测量，不能给出提升高度厘米数。

今天 197：前 30 s 右夹爪目标 5.2009–5.4 rad，接近张开；约 32.85 s 才开始小于 4.5 rad。后续视频中杯子位置有变化，但这不能证明达到成功夹起并稳定搬运的标准。

今天 QC：前 30 s 右夹爪目标 5.2086–5.4 rad，接近张开；约 110.92 s 才开始小于 4.5 rad，明显晚于旧 subtask 对照。前 30 s 记录中的 policy_action 与最终 cmd_action 的右夹爪平均绝对差仅约 0.00069 rad，197 约 0.00222 rad。因此这些阶段缺少闭合主要已存在于模型输出经过 chunk 混合后的序列中，并非额外 clip 把明显的闭合指令消除了。

4.5 rad 只是本报告用来描述“何时开始明显离开张开状态”的统一分析标记，不是部署中的成功阈值。夹爪完全闭到 0 可能意味着空夹，物体挡住夹爪时实际角度反而较大，不能仅凭角度判定抓取成功。

日志字段 `policy_action` 是已经经过 `blend_chunk_boundary()` 的序列，不是未经客户端处理的 server 原始动作。当前 recorder 没有同时保存全部原始 32 步输出，所以本次不能精确剥离前两步边界混合。

首阶段姿态轨迹亦明显不同：9/26 11:39 的右肘目标在接近杯子时约为 -0.3 rad，而今天两个 joint 运行很快走向约 +1.3 rad；右肩 pitch 和右腕 pitch 的目标路径也不同。实测多数跟随相应目标。因此“动作路径本身改变”是已观察事实，零前馈造成的跟踪偏差不能解释全部新旧目标路径差异。关节角正负本身不能直接换算成离桌高低，需进一步 FK/桌面几何测量。

频率也需要正确解释：9/26 成功对照同样只有约 16.46 Hz，今天约 16.6 Hz。同步停顿是部署与数采之间的共性问题，不足以单独解释今天退化。9/27 async 样本 27.7 Hz 不是用户指出的成功样本，不能用它替代 9/26 做因果比较。

## 完整 episode 与 subtask 训练的区别

“full FT / LoRA”描述哪些参数更新；“whole / subtask”描述训练样本的 prompt 和片段边界。这是两个不同维度。9/26 是 subtask + LoRA 7500；今天是 whole + full FT 31000/36000。另有菊花 subtask + full FT 36500，不能把所有 subtask 都称作 LoRA。

项目中历史 subtask 训练通常是一个共享模型，同时训练七种 subtask prompt，不是七套独立权重：搬黑杯、放菊花、左手拿白杯、右手扶黑杯、倒水归位、放红枣、放柠檬。部署时按任务选取五步顺序，更换 prompt，继续调用同一 checkpoint。

`make_subtask_lerobot_dataset.py` 按标注起止帧把原 episode 切成多个片段，每个片段使用当前子任务文本、独立时间索引和 episode 边界。whole 数据则保留原 episode 的完整帧序列，全过程使用该任务完整文本；训练动作窗口可以跨内部阶段边界。

两种模型训练和推理都以当前三路图像、当前关节 state、prompt 预测未来 32 步动作块。whole 并不是一次生成整条数十秒示教，也没有一个自动管理五个阶段的程序。当前客户端每次执行前 8 步，再取观测；whole 模型需要根据观测隐式判断下一步，subtask prompt 则提供显式阶段条件。当前默认政策不使用上一帧历史、显式阶段 token 或完成状态机。

whole 对阶段区分、接触动作与错误恢复提出更高要求；它的收益是有机会学到跨阶段连续衔接。不能保证平均 validation loss 更低就能更可靠地抓杯。此次 comparison 同时改变了参数更新方式、训练数据/划分、stats、prompt、客户端版本、高度锁配置，不能单凭这几次 rollout 证明 full FT 或 whole 训练本身一定更差。

## 当前 subtask 切换究竟如何工作

源码位置：QC 客户端 `PromptSchedule`、`NextSubtaskKey`、`run_sync_rollout` 与 `run_async_rollout`。

- `prompt_mode=subtask, subtask_schedule=fixed`：持续执行当前短 prompt；按 N 排队，在动作 chunk 边界切到下一步。切的是 prompt，不切权重；最后一步保持最后一步。
- `subtask_schedule=timed`：按执行 step 数占总 STEPS 的比例切换，默认占比 0.220/0.355/0.134/0.089/0.202；不检查上一步有没有完成。STEPS=3600 时约在 step 792/2070/2552/2873 切换；网络停顿会使真实墙钟切换时刻变晚。
- `prompt_mode=whole`：`prompt_for_step()` 始终返回同一完整 prompt。即使 terminal N handler 被创建并出现切换提示，实际 whole prompt 也不会变，不能借 N 把 joint 模型切成短 prompt 模式。

没有当前子任务完成分类器、物体检测/跟踪成功门槛、夹持成功阈值、done 输出或“完成就自动推进”的状态机。parquet 的 `next.done` 是片段/episode 结束元数据，G1D RepackTransform 没把它作为监督输出，policy.infer 返回动作与 timing；部署不消费它。

自动切换可以另行实现，但应基于可验证的成功事件。第一步可要求杯子与夹爪共同运动、杯子确实离桌并最终位于中间、释放后稳定，结合夹爪/电机状态且持续多帧成立；不能只用“夹爪角度小于某个数”或“执行时间已到”。阈值要从真实成功/失败录像标定，超时应停留/重试/交给人处理，不能视为成功直接推进。

## 建议的下一步顺序

1. 先记录实测升降高度、腰 yaw/pitch、相机初始帧与桌面/杯子几何，建立与菊花示教同条件的起点。9/26 的实际高度未知，应恢复测量能力，而非猜测提高到 0.15 m。
2. 将数采与部署的重力补偿方式对齐，并先检查 URDF 关节顺序、计算结果与控制器限幅。此审计只指出差异，没有擅自修改真机控制行为。
3. 保存未混合的完整模型动作块、最终控制目标和硬件实测状态，分别检查模型是否发出夹杯/提杯轨迹和电机是否跟随。
4. 用同一初始场景、同一高度/控制参数重复比较 9/26 checkpoint 与 joint 197/QC，优先评估第一步夹持和离桌提升。现在的证据不足以将 rollout 简化为一个“平均 MAE 好不好”的结论。
5. 若重新比较训练方案，应固定同一来源 episode 划分、归一化、优化方式与执行链路，分别做 subtask/whole，避免同时改变多项。当前更适合把首步可复现成功的 subtask 作为诊断基准，先补齐阶段完成检测，再评估全任务自主衔接。

## 文件与复现

- `audit_alignment.py`：离线分析脚本，不导入机器人控制器，不发送 DDS。
- `audit_metrics.json`：高度审计摘录、训练初始分布和六个 rollout 的配置与数值。
- `first40s_tracking.png`：9/26 11:39 subtask、197、QC 的右臂与夹爪目标/实测对照，旧运行在这段之后才按 N 切换；绿色仅为本地 197 数据中菊花训练 episode 首阶段的 state 1–99% 范围，不是安全边界或 QC 独立统计。
- `initial_camera_comparison.jpg`：三个数采任务和旧/新 rollout 初始三路图像。
- `subtask_0926_113904/cup_transition.jpg`：9/26 抓杯搬运段。
- 各运行目录：原始复制配置、事件、NPZ 和输入视频。输入视频写入速率为 fps/32、每推理块写一帧，而实际执行块为 8 步，因此视频播放秒数不等于真实运行秒数；本报告对照帧标注使用日志的真实 time_s。

复现命令：

```bash
/mnt/data/yangky/test/Psi0/.venv-openpi/bin/python \
  /mnt/data2/yangky/test/inference_alignment_20261007/audit_alignment.py
```

/mnt/data 当前无可用空间，本次新报告与数据复制全部落在 /mnt/data2，未删除原文件。

## 16:56 真机结果与插值修正

用户反馈重力补偿后离桌状态改善，但运动往返且不够平滑。该运行实际 25.96 Hz；检查发现策略目标本身有往返，以及插值每步从滞后实测状态起步产生的锯齿。已部署连续命令插值修正，同步刷新 9016 契约。尚未执行修正后的真机 rollout。见 [晃动诊断报告](../oscillation_audit_20261007/REPORT.md)。

## 17:16 反馈后的预取调整

确认最新真机使用了连续插值修正，但仍大多消费新块第 8–15 步。已将预取延后至剩余 4 步，并改为按观测后实际执行步数对齐，新增详细时序日志。无动作联调后续块消费 offset=3，约 25.42 Hz；真机效果未确认。见 [预取时序报告](../prefetch_tuning_20261007/REPORT.md)。

## 三个完整任务部署更新

按用户要求，默认改用 QC153 step31000 的 whole prompt（9020），197 step36000（9019）也同步最新版高度实测、重力补偿、连续插值和预取对齐。9016 subtask 基准暂停。模型参数未修改，无新增真机执行。红枣/柠檬物理高度的精确 root 权限安装脚本已上传，待操作员 sudo 安装。见 [RUN_WHOLE.md](../whole_task_alignment_20261007/RUN_WHOLE.md)。

## 三任务高度权限安装完成（17:42）

使用操作员授权，在 G1D 安装 0.150/0.145 的精确免密启动规则并更新三种高度的停止 helper；0.065 原有规则保留。权限查询均通过，没有执行高度控制、没有发机器人动作。当前三任务可由 TASK 自动选择对应高度，凭据未写入项目文件。

## 恢复 subtask 原子验证

用户报告 whole 首步抓杯仍不可靠，按要求默认恢复 9/26 subtask LoRA step7500/9016，保留全部控制修正；QC/9020 暂停、197 whole/9019 保留。原子验证固定单个 prompt，不按 N，后续阶段从匹配示教的状态独立验证。最新 whole 目录虽标 chrysanthemum，实际 TASK=red_dates、目标高度 0.150 m；本次仅核查配置，未独立诊断视频失败原因。启动和能力表见 [RUN_ATOMIC.md](../subtask_atomic_20261007/RUN_ATOMIC.md)。

## 启动 matched subtask 40k 对照

用户授权启动七 subtask 的 40k LoRA 与全量 FT。核查发现历史 full FT 已到40k、LoRA约13.5k，但 batch/EMA/优化器不同。另建同划分、同统计、batch1、无EMA、相同AdamW一阶动量精度和学习率的两项 fresh 训练，A800 单卡顺序队列。保留7500/10000/20000/40000等检查点，当前9016推理基准不替换。细节见 [RUN_COMPARE.md](../subtask_compare40k_20261007/RUN_COMPARE.md)。

## 对照训练进度与 TensorBoard

40k LoRA 正常进入训练，本次查询step5420（13.55%），最近step5000验证loss0.0202062；full FT仍排队。两项实时TensorBoard已建立，每5秒同步并回填历史指标，访问 http://100.122.105.65:6024/#scalars 。本轮未改训练参数或真机部署。详情见 subtask_compare40k_20261007/RUN_COMPARE.md。

## 2026-10-08：40k 对照完成

LoRA与全量FT均完成40000，验证loss分别0.0126283/0.0122344，best_checkpoint均40000。首步抓杯subtask0分别0.0141312/0.0142837，尚无新权重真机成功率。TensorBoard远端正常，掉线为跳板6024监听消失；已恢复并加断线重连脚本，HTTP200及scalar API已核实。当前访问 http://100.122.105.65:6024/#scalars 。本次没有更换部署权重。

## 2026-10-08：subtask 对照续训100k

按要求从两项40k完整训练状态续到总计100k，各增加60k；保持原40k日程末值LR1e-6，避免扩展cosine日程引起LR跳升。单卡LoRA先行、FT排队，原40k权重保留。TensorBoard新增对应to100k曲线，真机部署未更换。详情见 subtask_compare40k_20261007/CONTINUE_100K.md。
