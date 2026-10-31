"""第四版速度指令训练：机身前向速度 m/s，偏航角速度 rad/s。"""
import math

import numpy as np
import torch
from tensordict import TensorDict

from .environment import StandingEnv


class MovementEnv(StandingEnv):
    def __init__(self, seed=42, sample_commands=True):
        self.rng = np.random.default_rng(seed)
        self.sample_commands = sample_commands
        self.command = np.zeros(2, dtype=np.float32)
        super().__init__()

    def new_command(self):
        v, w = self.rng.uniform([-.20, -.65], [.20, .65])
        mode = self.rng.integers(5)
        self.command[:] = [v if mode in (1, 3, 4) else 0.,
                           w if mode in (2, 3, 4) else 0.]

    def reset(self):
        super().reset()
        if self.sample_commands:
            self.new_command()

    def get_observations(self):
        base = super().get_observations()["policy"]
        vx = self.data.sensor("linear_vel").data[0]
        extra = torch.tensor([[vx, *self.command]], dtype=torch.float32)
        return TensorDict({"policy": torch.cat((base, extra), dim=1)}, batch_size=[1])

    def reward_and_advance_command(self, upright, speed_xy, fallen):
        vx, vy = self.data.sensor("linear_vel").data[:2]
        wz = self.data.sensor("angular_vel").data[2]
        v, w = self.command
        reward = self.dt * (upright + 2 * math.exp(-((vx-v)/.09)**2)
                            + math.exp(-((wz-w)/.33)**2) - .5 * vy**2) - float(fallen)
        if self.sample_commands and self.episode_length_buf.item() % 150 == 0:
            self.new_command()
        return reward


class ResidualMovementEnv(MovementEnv):
    """差速轮速作为基础目标，策略只输出有限的平衡修正。"""
    def __init__(self, seed=42, sample_commands=True):
        super().__init__(seed, sample_commands)
        from .velocity_control import BodyVelocityController
        self.controller = BodyVelocityController(self)

    def action_to_targets(self, actions):
        q, _ = super().action_to_targets(actions)
        correction = 4.0 * np.tanh(actions.detach().cpu().numpy()[0, 4:])
        return q, self.controller.motor_speed_targets(*self.command)[4:] + correction

    def motor_torques(self, q_target, wheel_target):
        return self.controller.target_torques(q_target, wheel_target)

    def reward_and_advance_command(self, upright, speed_xy, fallen):
        # 保留速度跟踪奖励，并让大误差区间的减速也有收益。
        wz_error = self.data.sensor("angular_vel").data[2] - self.command[1]
        vx_error = self.data.sensor("linear_vel").data[0] - self.command[0]
        penalty = self.dt * (.35 * abs(wz_error) + 1.5 * abs(vx_error))
        return super().reward_and_advance_command(upright, speed_xy, fallen) - penalty
