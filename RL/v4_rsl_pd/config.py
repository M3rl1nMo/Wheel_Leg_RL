"""第四版共享网络配置（rsl_rl 5.5.2）。"""

from pathlib import Path
from copy import deepcopy

CHECKPOINT = Path(__file__).parent / "data/results/planar_policy.pt"

TRAIN_CFG = {
    "num_steps_per_env": 64,
    "save_interval": 50,
    # policy 是环境返回的观测组名，Actor 和 Critic 都读取这组观测。
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "actor": {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": [32, 32],
        "activation": "swish",  # 本版本中 SiLU 的配置名
        "obs_normalization": True,
        "distribution_cfg": {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": 0.3,
            "std_type": "log",
        },
    },
    "critic": {
        "class_name": "rsl_rl.models:MLPModel",
        "hidden_dims": [32, 32],
        "activation": "swish",
        "obs_normalization": True,
    },
    "algorithm": {
        "class_name": "rsl_rl.algorithms:PPO",
        "learning_rate": 0.0003,
        "schedule": "fixed",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "max_grad_norm": 1.0,
    },
}

def movement_train_cfg():
    """成功的 5,000 轮实验使用的实际 PPO 参数。"""
    cfg = deepcopy(TRAIN_CFG)
    cfg["algorithm"].update(learning_rate=.00015, entropy_coef=.0003)
    cfg["actor"]["distribution_cfg"].update(init_std=.6, std_range=(.05, 1.5))
    cfg["save_interval"] = 250
    return cfg
