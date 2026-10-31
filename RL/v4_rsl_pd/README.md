# 第四版：平地坐标导航

从项目根目录运行，使用项目的 Linux 虚拟环境：

```bash
source .venv-linux/bin/activate
unset PYTHONPATH
python -m RL.v4_rsl_pd.navigate 0.18 -0.18 --viewer
```

目标是世界坐标（米），初始朝向为 +x。去掉 `--viewer` 可无窗口运行；加 `--lqr` 可用同一坐标外环运行传统控制器。到达判据为连续 2 秒位置误差 ≤ 3 cm、速度 ≤ 0.03 m/s、转向速度 ≤ 0.1 rad/s。

| 文件 | 作用 |
|---|---|
| navigate.py | 从坐标误差计算速度指令，判断到达和停稳 |
| velocity_control.py | 将线速度、角速度映射成左右轮速，再计算电机力矩 |
| environment.py、movement.py | MuJoCo 单机环境、观测、速度奖励、策略修正 |
| parallel_env.py、config.py | 八个独立仿真进程、网络及 PPO 参数 |
| train.py、evaluate.py | 训练最终方案、测试六种速度指令 |

[选定模型](data/results/planar_policy.pt)来自 5,000 轮实验的最后一次更新。原始运行名为 `residual8-20261031-223329/model_4999.pt`。[速度评估](data/results/velocity-evaluation.json)与[导航对比](data/results/navigation-comparison.json)记录其验证范围。代码依赖项目内的机器人模型、共用传感器和传统控制器。

```bash
python -m RL.v4_rsl_pd.evaluate
python -m RL.v4_rsl_pd.train --iterations 5000
```

训练默认写入独立的 `data/experiments/` 子目录。早期站立、直接轮速实验和历次运行的原始数据见[历史归档](archive/README.md)。
