"""站立实验的 20 维观测；动作前后必须使用同一个构造函数。"""
import math
import torch

OBSERVATION_ORDER = [
    "pitch", "roll", "gyro_x", "gyro_y", "gyro_z", "torso_height",
    "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
    "right1_position", "right2_position", "left1_position", "left2_position",
    "right1_velocity", "right2_velocity", "left1_velocity", "left2_velocity",
    "rightwheel_velocity", "leftwheel_velocity", "remaining_time_fraction",
]
# 每项除以对应尺度。角度单位 rad，角速度 rad/s，高度 m，线速度 m/s。
# 这些是固定缩放值，不是测得的最大值；不会把观测截断到 [-1, 1]。
OBSERVATION_SCALES = [math.pi, math.pi, 10., 10., 10., .25,
                      1., 1., 1., math.pi, math.pi, math.pi, math.pi,
                      10., 10., 10., 10., 10., 10., 1.]
OBSERVATION_DIM = len(OBSERVATION_ORDER)


def build_observation(info, data, torso_id, episode_time_limit):
    # 直接使用 IMU 局部坐标系的 x/y/z，避免旧辅助代码的轴名称混淆。
    gyro = data.sensor("angular_vel").data
    velocity = data.sensor("linear_vel").data
    remaining = max(0., 1. - float(data.time) / episode_time_limit)
    values = [
        info["euler"]["pitch"], info["euler"]["roll"],
        *gyro, float(data.xpos[torso_id, 2]), *velocity,
        *[info["pos"][name] for name in ("right1", "right2", "left1", "left2")],
        *[info["vel"][name] for name in
          ("right1", "right2", "left1", "left2", "rightwheel", "leftwheel")],
        remaining,
    ]
    # 外层 [] 表示只有一条观测，输出形状为 [1, 20]。
    obs = torch.tensor([values], dtype=torch.float32)
    obs = obs / torch.tensor(OBSERVATION_SCALES, dtype=torch.float32)
    if not torch.isfinite(obs).all():
        raise ValueError("观测出现 NaN 或无穷大，停止训练。")
    return obs
