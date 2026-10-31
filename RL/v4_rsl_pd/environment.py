"""单台轮腿机器人在平地上站立；策略输出关节角度偏移和轮速指令。"""

import math

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from RL.v3_direct.direct_navigation import make_model, robot_info_update

LEGS = ("right1", "right2", "left1", "left2")
WHEELS = ("rightwheel", "leftwheel")


class StandingEnv(VecEnv):
    num_envs = 1
    num_actions = 6
    device = "cpu"

    def __init__(self):
        self.model = make_model()  # 复用第三版的平地模型，关闭坡面碰撞
        self.data = mujoco.MjData(self.model)
        self.torso = self.model.body("torso").id
        # 起始实验参数；角度和转速都使用电机编码器自己的坐标方向。
        self.cfg = dict(substeps=10, episode_seconds=10, leg_scale=0.2,
                        wheel_scale=5.0, kp=5.0, kd=0.2, kv=0.5)
        self.dt = self.model.opt.timestep * self.cfg["substeps"]
        self.max_episode_length = round(self.cfg["episode_seconds"] / self.dt)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.reset()
        self.q_nominal = self.get_observations()["policy"][0, 5:9].numpy().copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.episode_length_buf.zero_()

    def get_observations(self):
        info = robot_info_update(self.data)
        values = [
            info["euler"]["pitch"], info["euler"]["roll"],
            *self.data.sensor("angular_vel").data,
            *[info["pos"][name] for name in LEGS],
            *[info["vel"][name] for name in LEGS + WHEELS],
        ]
        return TensorDict({"policy": torch.tensor([values], dtype=torch.float32)},
                          batch_size=[1])

    def action_to_targets(self, actions):
        if actions.shape != (1, 6) or not torch.isfinite(actions).all():
            raise ValueError("期望一台机器人的六个有限动作，形状为 [1, 6]")
        raw = actions.detach().cpu().numpy()[0]
        q_target = self.q_nominal + self.cfg["leg_scale"] * np.tanh(raw[:4])
        wheel_target = self.cfg["wheel_scale"] * raw[4:]
        return q_target, wheel_target

    def leg_torques(self, q_target, info):
        q = np.array([info["pos"][name] for name in LEGS])
        dq = np.array([info["vel"][name] for name in LEGS])
        leg = self.cfg["kp"] * (q_target - q) - self.cfg["kd"] * dq
        return np.clip(leg, -1.8, 1.8)

    def motor_torques(self, q_target, wheel_target):
        info = robot_info_update(self.data)
        omega = np.array([info["vel"][name] for name in WHEELS])
        wheel = self.cfg["kv"] * (wheel_target - omega)
        return np.concatenate([self.leg_torques(q_target, info), np.clip(wheel, -1.5, 1.5)])

    def reward_and_advance_command(self, upright, speed_xy, fallen):
        return self.dt * (upright - 0.5 * speed_xy**2) - float(fallen)

    def step(self, actions):
        q_target, wheel_target = self.action_to_targets(actions)
        for _ in range(self.cfg["substeps"]):
            self.data.ctrl[:] = self.motor_torques(q_target, wheel_target)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)

        self.episode_length_buf += 1
        obs = self.get_observations()
        pitch, roll = obs["policy"][0, :2].tolist()
        height = self.data.xpos[self.torso, 2]
        speed_xy = np.linalg.norm(self.data.sensor("linear_vel").data[:2])
        fallen = bool(abs(pitch) > 0.8 or abs(roll) > 0.8 or height < 0.12)
        timeout = self.episode_length_buf.item() >= self.max_episode_length and not fallen

        # 每秒最多得 1 分；倾斜或平移会减分，摔倒一次再扣 1 分。
        upright = math.exp(-4 * (pitch**2 + roll**2))
        reward = self.reward_and_advance_command(upright, speed_xy, fallen)
        done = fallen or timeout
        if done:
            self.reset()
        obs = self.get_observations()
        return (obs, torch.tensor([reward], dtype=torch.float32),
                torch.tensor([done]), {"time_outs": torch.tensor([timeout])})
