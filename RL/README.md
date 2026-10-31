# 学习控制代码导航

2026-10-31 按学习阶段整理。第一版已由用户核对，确认为 **4 输入、6 输出**，不是 8 输入。此次只调整目录、导入和存储路径，没有修改网络结构、控制算法或重新训练。

| 目录 | 输入 → 输出 | 控制方式 | 当前状态 |
| --- | --- | --- | --- |
| `v1_standing/` | 4 → 6 | 最初的单步 Actor–Critic；无位置外环 | 教学版本，尚未学会可靠站立 |
| `v2_outer/` | 24 → 6 | 手写位置外环生成运动指令，网络输出力矩 | 已有平面到点停稳验证，范围见报告 |
| `v3_direct/` | 27 → 6 | 目标相对位置、机器人状态和上一动作直接输入网络 | 模仿学习及 PPO 实验，尚未可靠到点停稳 |
| [v4_rsl_pd/](v4_rsl_pd/README.md) | 18 → 6 | 差速目标＋PPO 平衡修正＋PD，坐标外环 | 8 个固定平地目标均到点停稳 |

## 现在主要看哪些文件

- 第一版：[网络定义](v1_standing/networks.py)、[训练入口](v1_standing/observe_robot.py)。4 个输入为俯仰角、俯仰角速度、左右轮速度；原代码含控制模式选择，策略模式不使用位置外环。
- 第二版：[位置外环](v2_outer/planar_control.py)、[模仿学习和评估](v2_outer/planar_navigation.py)、[PPO 微调](v2_outer/planar_ppo.py)、[详细说明](v2_outer/PLANAR_NAVIGATION.md)。
- 第三版：[模仿学习和评估](v3_direct/direct_navigation.py)、[PPO 微调](v3_direct/direct_ppo.py)、[详细说明](v3_direct/DIRECT_NAVIGATION.md)。模仿训练时仍由教师生成标签；网络独立评估时不调用位置外环。
- `common/` 是共用代码：网络定义、20 维基础观测、存档、训练统计和 LQR 教师。第二、三版共享部分 PPO 更新函数，不代表第三版执行位置外环。
- [archive/](archive/README.md) 保存 20 维站立、22 维 x 轴控制、激活函数比较及其他端到端变体。
- [训练日志](docs/TRAINING_LOG.md) 和 [训练历程](docs/TRAINING_CAMPAIGN_20261029.md) 保留全部历史结论。

## 训练数据在哪里

每个版本的 `data/` 与代码放在一起：`results/` 是选定模型和报告，`experiments/` 是完整实验数据及候选模型。第一版使用 `data/checkpoints/` 和 `data/training_logs/`。其他实验的数据统一放在 `archive/data/`，其中每次实验仍有独立目录。

完整数据和训练存档仅在本机保存，不自动加入 Git；源码、说明和选定结果可以提交。历史 JSON 报告内部的 Windows 路径不改写，它们记录当时的数据来源。文件新旧位置见 [迁移清单](reorganization_manifest.json)，其中保存了移动前的文件 SHA256。

## Linux 启动

先在终端进入项目并激活环境：

```bash
cd /home/melin/Projects/Wheel_Leg
source .venv-linux/bin/activate
unset PYTHONPATH
```

查看第二版现有模型（只评估，不训练）：

```bash
python RL/v2_outer/planar_navigation.py --evaluate --checkpoint RL/v2_outer/data/results/planar_policy.pt --target-x 0.25 --target-y 0 --seconds 30 --viewer
```

查看第三版现有模型（这是未成功的实验基线）：

```bash
python RL/v3_direct/direct_navigation.py --evaluate --checkpoint RL/v3_direct/data/results/direct_imitation_candidate.pt --target-x 0.25 --target-y 0 --seconds 30 --viewer
```

去掉 `--viewer` 可无窗口评估。这些入口也支持 `python -m RL.v2_outer.planar_navigation ...` 等模块形式运行。

第一版启动命令如下。注意：它会打开仿真窗口，默认加载本版最近存档，执行训练并保存，不是只查看模型。

```bash
python RL/v1_standing/observe_robot.py
```

第二、三版重新做模仿训练分别使用 `python RL/v2_outer/planar_navigation.py --train` 和 `python RL/v3_direct/direct_navigation.py --train`；需要较长时间，本次没有执行。第二版初始化仍会读取归档内的 22 维模型；第三版初始化读取第二版模型，因此不要单独删除 `archive/data/results/` 或 `common/`。
