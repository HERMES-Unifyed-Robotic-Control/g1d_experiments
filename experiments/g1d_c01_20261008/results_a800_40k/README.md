# C01 A800 LoRA40k离线结果

E30k已完成，π LoRA40k已完成；验证最优为step40000，composite loss0.3629367731。

|指标|π0.5 baseline7500|C01 LoRA40000|
|---|---:|---:|
|mae|0.056769|0.050182|
|rmse|0.126825|0.106358|
|left_arm_mae|0.038898|0.034840|
|right_arm_mae|0.052340|0.047218|
|left_gripper_mae|0.106647|0.088524|
|right_gripper_mae|0.162994|0.139993|

整体MAE下降11.60%，RMSE下降16.14%。7个subtask聚合MAE均下降，但task1的RMSE由0.07238上升至0.09234、右夹爪MAE由0.06719升至0.14397；不能声称每个关节/阶段均改善。

相同200个holdout观测，比较baseline H32前30步与C01 H30原生16D动作。holdout曾用于选checkpoint；训练步数7500/40000不同，latent/action噪声维数不同。该结果不等同于独立测试或真机成功率。

## 部署入口

A800：tmux g1d_c01_serve，checkpoint40000，WebSocket9021；sys01：tmux g1d_c01_forward。G1D独立目录/home/unitree/pi05_g1d_deploy_c01_40k_20261009，服务与client契约H30/native16。

```bash
cd /home/unitree/pi05_g1d_deploy_c01_40k_20261009
G1D_PROFILE=official POLICY_HOST=192.168.10.116 POLICY_PORT=9021 TASK=chrysanthemum SUBTASK_INDEX=0 STEPS=3600 GO_READY_BEFORE_START=1 RECORD_DIR=/home/unitree/pi05_g1d_deploy_c01_40k_20261009/rollouts/c01_lora40k_atomic0_$(date +%Y%m%d_%H%M%S) NETWORK_INTERFACE=eth0 ./run_client_on_g1d.sh --execute
```

客户端与控制辅助文件、profile、图像预处理哈希已对齐服务元数据。配置沿用已部署baseline的高度/重力补偿/异步执行/平滑参数；启动命令由用户在真机执行。当前仍为固定subtask原子评估，不是全任务自动切换。

## 服务准备完成

module_jit启动预热后，两次合成观测请求均返回有限30×16动作；服务端推理约90～114ms，经跳板端到端约0.38～0.44秒。G1D连接sys01 192.168.10.116:9021并读取40000检查点元数据成功。该检查没有发送机器人控制命令；实际30Hz控制与异步chunk供应情况需在rollout中记录。
