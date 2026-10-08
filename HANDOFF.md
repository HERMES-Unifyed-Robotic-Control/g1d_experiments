# 评测入口

## 1. 查看HF数据与权重

在HF点 **Files and versions**，进入以下目录：

|内容|直接浏览链接|
|---|---|
|原始food sorting|[food_sorting.tar](https://huggingface.co/datasets/yky666/humanoid_long_horizon_tasks/blob/main/food_sorting.tar)|
|标注与划分|[annotations](https://huggingface.co/datasets/yky666/humanoid_long_horizon_tasks/tree/main/g1d_food_sorting/annotations)|
|四个LeRobot视频数据包|[processed](https://huggingface.co/datasets/yky666/humanoid_long_horizon_tasks/tree/main/g1d_food_sorting/processed)|
|7500baseline检查点|[7500](https://huggingface.co/yky666/g1d-pi05-subtask-baseline/tree/main/7500)|

截图中的`water_bottle_pick_and_insert.rar`是另一任务。返回Files根目录后进入`g1d_food_sorting`即可找到本次数据。tar和检查点分片太大，HF不能直接预览；点Download或用下面的下载脚本。权重为私有，需登录拥有权限的账号，否则链接可能显示404。

## 2. 获取代码与数据

```bash
git clone --recurse-submodules https://github.com/HERMES-Unifyed-Robotic-Control/g1d_experiments.git
cd g1d_experiments
python -m pip install -r requirements-data.txt
# 使用具有私有模型访问权限的HF账号登录
python scripts/download_artifacts.py --dataset subtasks_train --dataset subtasks_validation --baseline
```

原始数据在HF dataset `food_sorting.tar`；时间标注与原episode分组划分在 `data/`；需要完整任务时另下载 `--dataset whole_197`，所有切分片段另下载 `--dataset subtasks_all`。下载脚本只接受ARTIFACTS.json中已完成并固定revision的条目。

## 3. 跑当前baseline

当前baseline是官方OpenPI pi05 LoRA step7500，尚不是G1D C01 policy。安装环境见README，固定200条holdout观测的离线动作评估：

```bash
python scripts/evaluate_subtask.py --checkpoint artifacts/baseline/7500 --variant lora
```

换full FT权重需同时改 `--variant fullft`。离线动作误差不能当作真机成功率。原数据划分曾用于选ckpt，不是独立测试集。

真机服务入口 `scripts/serve_baseline.py`，客户端 `deploy/client/`。先核对server/client契约，再按真实场景准备原子起点。当前五阶段仍人工切换，没有自动完成标志。Baseline现场的重力补偿、高度与异步预取配置已保留。

## 4. G1D C01 E/π训练与评测

本项目由我们自行完成G1D适配、E训练、π训练和评测。原Dexjoco检查点作为架构来源，不能直接当作G1D检查点。

参考固定到指定分支与commit，见PROVENANCE.json和DEXJOCO_G1D_MATCHING.md。接口入口为 `g1d_experiments.c01_dataset.G1DC01Windows` / `collate_c01_windows`，16D decoder为 `g1d_experiments.c01_decoder.G1DC01Decoder`。

`observation`: RGB `[B,3,3,224,224]`（view,channel）、state `[B,16]`、prompt list。
`supervision`: action/state_future/joint_response `[B,30,16]`、valid `[B,30]`。
RGB uint8；state/action为物理单位。H30@30Hz；三视图按egocentric,left_wrist,right_wrist排列。未来数据只能用于E/损失监督，不进入π推理。

原22/44D末端+灵巧手动作不等于G1D的双臂14关节+2夹爪。不能靠补零、切片或转换名称来直接部署原C01权重。已提供原生16D数据/decoder/GT-AUX和gripper-Δ组件；已加入G1D视觉准备、E训练/cache导出、原生16D JAX/pi桥接、GT-AUX校准、π训练与动作评估入口；运行顺序见 [RUN_C01.md](experiments/g1d_c01_20261008/RUN_C01.md)。DINOv3原生权重已就绪，A800已开始真实视觉特征准备，队列随后运行E30k与π LoRA40k；没有完成的C01评测结果。

## 5. 权重与证据

模型repo：`yky666/g1d-pi05-subtask-baseline`（私有）。`7500/`含完整baseline训练状态；`comparisons/`仅含推理params/assets。具体已发布权重和不可变revision见ARTIFACTS.json。

代码在GitHub；大数据/权重在HF；三者权限独立。实验报告为历史现场证据，路径/进度不是可迁移配置。评测应记录prompt、原子起点、物料抓取条件、人工介入次数、progress/10与闭环成功率。
