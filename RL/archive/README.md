# 其他实验归档

`code/` 保存其他实验版本，`data/experiments/` 按原实验名称分别保存数据、候选模型和报告，`data/results/` 保存原来已挑选的结果，`docs/` 保存对应诊断说明。

- `observe_robot.py`：20 维单步奖励训练；不是保留的第一版 4 维入口。
- `lqr_dagger.py`、`x_position_imitation.py`：站立数据聚合与 22 维 x 轴控制。20 维模仿学习入口在 `../common/lqr_imitation.py`，其教师工具仍被主要版本共用。
- `compare_policy_activations.py`：SiLU/ReLU 比较。
- `direct_*`：时间信息、价值校准、残差、大网络、历史动作等端到端变体。
- `diagnose_*`、`check_action_*`、`verify_expanded.py`：训练诊断及候选验证。它们是实验记录的一部分，因此归档保留；独立的 `test_*.py` 自动测试脚本已删除。
- `actor_critic_demo.py`：最早的梯度更新教学演示。

脚本导入和数据路径已随目录更新。部分实验脚本直接运行就开始计算或训练，阅读文件入口后再运行。不要把候选模型覆盖到第二、三版的基线模型上。所有结果维持原有结论，整理目录不代表实验通过。
