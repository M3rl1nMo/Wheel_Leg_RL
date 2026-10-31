"""每个进程独占一台 MuJoCo 仿真；主进程只传动作和观测，不传模型。"""
import multiprocessing as mp
import traceback

import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .environment import StandingEnv


def _worker(pipe, movement, index):
    try:
        torch.set_num_threads(1)
        if movement:
            from .movement import MovementEnv, ResidualMovementEnv
            cls = ResidualMovementEnv if movement == "residual" else MovementEnv
            env = cls(seed=1000 + index)
        else:
            env = StandingEnv()
        pipe.send(("ready", (env.get_observations()["policy"][0].numpy(),
                             env.max_episode_length, env.cfg)))
        while True:
            action = pipe.recv()
            if action is None:
                break
            obs, reward, done, extras = env.step(torch.from_numpy(action[None]))
            pipe.send(("step", (obs["policy"][0].numpy(), reward.item(), done.item(),
                                extras["time_outs"].item(), env.episode_length_buf.item())))
    except EOFError:
        pass
    except Exception:
        pipe.send(("error", traceback.format_exc()))
    finally:
        pipe.close()


class ParallelStandingEnv(VecEnv):
    num_actions = 6
    device = "cpu"

    def __init__(self, num_envs=8, movement=False):
        self.num_envs = num_envs
        self.pipes = []
        self.workers = []
        # Linux forkserver 先加载 CPU 仿真代码，CUDA 仅在 PPO 主进程使用。
        ctx = mp.get_context("forkserver")
        mp.set_forkserver_preload(["RL.v4_rsl_pd.environment"])
        try:
            for index in range(num_envs):
                parent, child = ctx.Pipe()
                worker = ctx.Process(target=_worker, args=(child, movement, index), daemon=True)
                worker.start()
                child.close()
                self.workers.append(worker)
                self.pipes.append(parent)
            initial = self._receive("ready")
            self._obs = torch.from_numpy(np.stack([row[0] for row in initial]))
            self.max_episode_length, self.cfg = initial[0][1:]
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        except BaseException:
            self.close()
            raise

    def _receive(self, expected):
        rows = []
        for pipe in self.pipes:
            tag, value = pipe.recv()
            if tag != expected:
                raise RuntimeError(value)
            rows.append(value)
        return rows

    def get_observations(self):
        return TensorDict({"policy": self._obs}, batch_size=[self.num_envs])

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError("动作形状必须是 [环境数, 6]")
        for pipe, action in zip(self.pipes, actions.detach().cpu().numpy()):
            pipe.send(action)
        # 所有动作均已派发，各进程并行完成十个物理子步。
        obs, rewards, dones, timeouts, lengths = zip(*self._receive("step"))
        self._obs = torch.from_numpy(np.stack(obs))
        self.episode_length_buf = torch.tensor(lengths)
        return (self.get_observations(), torch.tensor(rewards), torch.tensor(dones),
                {"time_outs": torch.tensor(timeouts)})

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            pipe.close()
        for worker in self.workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join()
