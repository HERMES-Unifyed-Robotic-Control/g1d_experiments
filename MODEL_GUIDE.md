# G1D C01模型说明

当前实现来自Dexjoco C01分支的G1D native16适配，策略骨干仍复用OpenPI π0.5。E不是奖励或任务完成分数，π是策略符号，pai是它的读音。当前G1D变体用相邻时刻实测16D关节变化作为响应监督，不是原Dexjoco的末端/四指尖几何监督。

## 与baseline的区别

baseline：当前三路图像、state和prompt → π0.5 → 动作chunk。
C01：当前三路图像、state和prompt → 修改后的π0.5 → E30×256 → 单decoder结合当前raw state16 → 动作30×16。

16D顺序是左臂7、右臂7、左夹爪1、右夹爪1，输出绝对关节/夹爪目标位置。π骨干输入state填充32D，decoder接收raw16D。E256是学习出的隐藏维数，不是256个关节。

## E训练：学习示范的动作和后果

冻结DINOv3 ViT-B/16，从示范主相机视频每5帧提取特征；训练集拟合PCA64并构造局部视觉变化目标。E codec输入未来30步示范动作A、实测相邻关节变化S、视觉变化V，分别编码后由query fusion和时序模块生成30×256的E。

codec通过三个头重建动作、关节响应和视觉变化，避免E只保存某一种信号。其目标为动作重建loss加响应/world重建loss；两后者权重从0.05逐渐升至0.2。当前E训练30k、batch32，学习率20k前1e-4、之后3e-5，使用验证最优checkpoint导出E标签缓存与decoder初始化。

这里的未来信息只用于离线教师标签，真机推理不需要未来图像或未来state。视觉阶段目前没有有效机器人/物体掩码，采用明确记录的legacy-none协议，不能把这些视觉变化当成完美分离的物体运动。

## π训练：从当前观测预测E

π读取当前主相机、左右腕相机图像、当前关节状态和subtask文本。保留π0.5预训练骨干，替换输入/输出投影以处理256D的E token，采用flow matching学习从噪声生成标准化E；不直接在16D动作空间做flow matching。

训练时从含噪E估计干净E，再通过同一个动作decoder得到动作，并加两类辅助监督：

- GT-AUX：预测动作与示范动作的normalized Huber误差，按左臂/右臂/左夹爪/右夹爪四组平均。
- gripper-Δ：两个夹爪的相邻动作变化与示范变化一致，不是单纯惩罚动作变化；它不保证整条手臂平滑。

总目标为FM(E)加带time gate和ramp的辅助项。辅助项只在噪声时间0.2～0.7区间参与，ramp于10k步达到1。G1D训练集梯度校准得到GT-AUX权重1.3227095743，夹爪Δ权重0.05，decoder更新乘0.25。

当前主分支是π LoRA40k，同时训练新的E投影/decoder；支持fullft入口，但不能将该主分支称为C01全量FT。A800 batch1；A100队列global batch2、每卡1样本，双实例同步梯度，因此同样40k并非相同样本预算。

## 数据与评测

原197个episode按原episode划分177训练/20验证，再切subtask，得到885训练片段/100验证片段；同一原episode不能跨split。窗口H30不跨subtask边界，统计/PCA/校准只用训练集。

π每2500步在固定200个heldout观测上验证并保存checkpoint。TensorBoard的effect_loss是E空间flow matching误差，不能与baseline动作空间loss直接比较。公平比较要用相同观测、prompt、H30/native16动作指标，之后做相同初始布局、控制参数的真机progress/10、成功率及介入次数比较。模型训练loss下降不能直接证明抓杯成功或长程能力。

## 部署

A800/A100服务器加载训练好的π checkpoint及decoder、E统计和state统计；WebSocket接收当前观测，默认10次去噪生成30×256 E，decoder恢复30×16物理动作并返回G1D客户端。训练结束并不自动执行机器人动作。

G1D客户端负责读取三路图像和关节状态、发送prompt、异步预取/执行动作队列，以及高度、重力补偿、限幅和配置的平滑。30帧在数据定义上约1秒；推理延迟和实际控制频率需要用真机rollout验证。E编码器、DINO视觉后果分支和未来数据不在在线推理链路中。

当前服务metadata为subtask模式，没有完成检测、阶段阈值或自动prompt规划器；仍需阶段prompt/外部切换。因此这一轮主要验证原子能力，对长程任务完成率尚无结论。baseline仍是实机已有经验的官方π0.5 LoRA7500，C01真机效果尚待实测。
