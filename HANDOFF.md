# 师兄评测交付入口

## 1. 获取代码与数据

```bash
git clone --recurse-submodules https://github.com/HERMES-Unifyed-Robotic-Control/g1d_experiments.git
cd g1d_experiments
python -m pip install -r requirements-data.txt
# 使用具有私有模型访问权限的HF账号登录
python scripts/download_artifacts.py --dataset subtasks_train --dataset subtasks_validation --baseline
```

原始数据在HF dataset `food_sorting.tar`；时间标注与原episode分组划分在 `data/`；需要完整任务时另下载 `--dataset whole_197`，所有切分片段另下载 `--dataset subtasks_all`。下载脚本只接受ARTIFACTS.json中已完成并固定revision的条目。

## 2. 跑当前baseline

当前baseline是官方OpenPI pi05 LoRA step7500，不是EFT。安装环境见README，固定200条holdout观测的离线动作评估：

```bash
python scripts/evaluate_subtask.py --checkpoint artifacts/baseline/7500 --variant lora
```

换full FT权重需同时改 `--variant fullft`。离线动作误差不能当作真机成功率。原数据划分曾用于选ckpt，不是独立测试集。

真机服务入口 `scripts/serve_baseline.py`，客户端 `deploy/client/`。先核对server/client契约，再按真实场景准备原子起点。当前五阶段仍人工切换，没有自动完成标志。Baseline现场的重力补偿、高度与异步预取配置已保留。

## 3. 接入师兄的EFT模型

```python
from torch.utils.data import DataLoader
from g1d_experiments.eft_dataset import G1DPredictiveWindowDataset, collate_g1d_windows
loader = DataLoader(G1DPredictiveWindowDataset('artifacts/data/g1d_subtasks_official_train_v1'),
                    batch_size=1, collate_fn=collate_g1d_windows)
```

`inputs`: RGB历史 `[B,4,3,3,224,224]`（time,view,channel）、状态 `[B,4,16]`、语言list。
`targets`: RGB未来 `[B,45,3,3,224,224]`、未来状态/动作 `[B,45,16]`。
RGB uint8；state/action为物理单位。历史0.5s，未来1.5s，30Hz；三视图按egocentric,left_wrist,right_wrist排列。未来数据只能进入监督损失，不能输入U predictor。

公开EFT上游当前仍是未实现的骨架，本交付提供数据匹配，没有EFT训练权重。真实模型评测需要师兄提供实现分支、检查点与normalization/动作头协议。

## 4. 权重与证据

模型repo：`yky666/g1d-pi05-subtask-baseline`（私有）。`7500/`含完整baseline训练状态；`comparisons/`仅含推理params/assets。具体已发布权重和不可变revision见ARTIFACTS.json。

代码在GitHub；大数据/权重在HF；三者权限独立。实验报告为历史现场证据，路径/进度不是可迁移配置。评测应记录prompt、原子起点、物料抓取条件、人工介入次数、progress/10与闭环成功率。
