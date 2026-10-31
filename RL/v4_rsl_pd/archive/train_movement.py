"""从第四版站立策略迁移到速度指令任务。"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from ..config import TRAIN_CFG
from ..parallel_env import ParallelStandingEnv

STANDING = Path(__file__).parent / "runs/standing8-20k-20261031-205443/model_19999.pt"


def warm_start(runner):
    original = torch.load(STANDING, map_location="cpu", weights_only=False)
    for key, model in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
        state = model.state_dict()
        for name, value in original[key + "_state_dict"].items():
            if name == "distribution.log_std_param":
                continue  # 使用新任务的探索尺度，避免站立末期的大方差
            if name == "obs_normalizer.count":
                state[name].fill_(1024)
            elif state[name].shape == value.shape:
                state[name].copy_(value)
            else:
                if name == "mlp.0.weight":
                    state[name].zero_()
                state[name][:, :15].copy_(value)
        model.load_state_dict(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--residual", action="store_true", help="在差速轮速基础上学习修正")
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(43)
    torch.set_num_threads(1)
    cfg = deepcopy(TRAIN_CFG)
    cfg["algorithm"].update(learning_rate=.00015, entropy_coef=.0003)
    cfg["actor"]["distribution_cfg"].update(init_std=.6, std_range=(.05, 1.5))
    cfg["save_interval"] = 250
    env = ParallelStandingEnv(8, movement="residual" if args.residual else True)
    try:
        runner = OnPolicyRunner(env, cfg, log_dir=str(args.log_dir), device="cuda")
        if args.residual:
            # 零输出初始策略精确对应手写速度接口。
            with torch.no_grad():
                runner.alg.actor.mlp[-1].weight.zero_()
                runner.alg.actor.mlp[-1].bias.zero_()
        else:
            warm_start(runner)
        runner.save(str(args.log_dir / "initial.pt"))
        runner.learn(args.iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
