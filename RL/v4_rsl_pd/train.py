"""第四版主训练入口：差速轮速 + PPO 平衡修正。"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from .config import movement_train_cfg
from .parallel_env import ParallelStandingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations 必须大于零")
    folder = args.log_dir or (Path(__file__).parent / "data/experiments" /
                              datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(43)
    torch.set_num_threads(1)
    env = ParallelStandingEnv(8, movement="residual")
    try:
        runner = OnPolicyRunner(env, movement_train_cfg(), log_dir=str(folder),
                                device="cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            runner.alg.actor.mlp[-1].weight.zero_()
            runner.alg.actor.mlp[-1].bias.zero_()
        runner.save(str(folder / "initial.pt"))
        runner.learn(args.iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
