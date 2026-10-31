# 第四版历史归档

原顶层的 `runs/` 已整体移动到这里的 `runs/`。所有原始检查点、TensorBoard 事件、评估报告和日志保留在本机。按实验名前缀查找：

- `standing8-cuda-01` 和 `standing8-20k-*`：站立试验和续训。
- `movement8-*`：从站立权重迁移，直接输出轮速的试验。
- `residual8-*`：差速目标加策略修正，最终模型来自这批训练。

旧入口按模块运行，例如 `python -m RL.v4_rsl_pd.archive.train_standing --help` 和 `python -m RL.v4_rsl_pd.archive.train_movement --help`。旧版评估分别是 `evaluate_standing.py` 和 `evaluate_movement.py`。
