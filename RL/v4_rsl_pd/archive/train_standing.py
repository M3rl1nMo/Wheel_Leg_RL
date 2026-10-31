"""项目根目录运行 python -m RL.v4_rsl_pd.archive.train_standing --iterations 1。"""

import argparse
from datetime import datetime
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from ..config import TRAIN_CFG
from ..parallel_env import ParallelStandingEnv


def main():
    parser = argparse.ArgumentParser(description="使用 rsl_rl 训练平地站立策略")
    parser.add_argument("--iterations", type=int, required=True, help="目标总轮数（含检查点已完成轮数）")
    parser.add_argument("--log-dir", type=Path, help="保存本次训练的新目录")
    parser.add_argument("--resume", type=Path, help="恢复已有 PPO 检查点")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations 必须大于零")
    folder = args.log_dir or (Path(__file__).parent / "runs" /
                              datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    env = ParallelStandingEnv(num_envs=8)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PPO 设备: {device}, 并行环境: {env.num_envs}", flush=True)
        runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=str(folder), device=device)
        if args.resume:
            runner.load(str(args.resume), map_location=device)
            runner.current_learning_iteration += 1  # 存档记录的是已完成的零基编号
        remaining = args.iterations - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError("目标轮数必须大于检查点已完成轮数")
        print(f"已完成 {runner.current_learning_iteration} 轮，剩余 {remaining} 轮", flush=True)
        runner.learn(num_learning_iterations=remaining)
    finally:
        env.close()
    print("模型和日志目录：", folder)


if __name__ == "__main__":
    main()
