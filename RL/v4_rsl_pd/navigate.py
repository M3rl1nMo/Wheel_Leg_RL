"""第四版平地坐标导航。目标是世界坐标，单位 m；初始朝向沿 +x。"""
import argparse
from contextlib import nullcontext
import json
import math
import time

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import mujoco
from RL.common.lqr_imitation import LIMITS, robot_info_update
from RL.v2_outer.planar_control import PlanarLQRTeacher, wrap_angle
from .config import CHECKPOINT, movement_train_cfg
from .movement import ResidualMovementEnv



@torch.inference_mode()
def navigate(env, actor, target, seconds=40., viewer=False):
    env.reset()
    steps = math.ceil(seconds / env.dt)
    env.max_episode_length = steps + 1  # 不触发训练时的十秒自动重置
    teacher = PlanarLQRTeacher(env.model.opt.timestep) if actor is None else None
    hold_yaw = None
    streak = 0
    if viewer:
        from mujoco import viewer as mjviewer
        window = mjviewer.launch_passive(env.model, env.data)
    else:
        window = nullcontext(None)
    status = "timeout"
    xy = env.data.xpos[env.torso, :2].copy()
    elapsed = 0.
    with window as win:
        for k in range(steps):
            if win is not None and not win.is_running():
                status = "closed"
                break
            start = time.monotonic()
            before = xy.copy()
            rotation = env.data.xmat[env.torso].reshape(3, 3)
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            dx, dy = target - xy
            distance = math.hypot(dx, dy)
            if distance < 0.022 and hold_yaw is None:
                hold_yaw = yaw
            elif distance > 0.032:
                hold_yaw = None
            heading = math.atan2(dy, dx) if hold_yaw is None else hold_yaw
            angle = wrap_angle(heading - yaw)
            # 后方目标采用倒车，避免在近目标处为误差反复转半圈。
            if hold_yaw is None and abs(angle) > math.pi / 2:
                angle = wrap_angle(angle - math.pi)
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            v = np.clip(.85 * forward * max(0., math.cos(angle)), -.15, .15) - .012
            w = np.clip(1.6 * angle, -.55, .55)
            if k * env.dt < 1.:
                v, w = 0., 0.
            env.command[:] = (v, w)
            if teacher is None:
                _, _, done, _ = env.step(actor(env.get_observations()))
                fallen = done.item()
            else:
                for _ in range(env.cfg["substeps"]):
                    info = robot_info_update(env.data)
                    torque = teacher.action(info, env.data.sensor("angular_vel").data[2], v, w)
                    env.data.ctrl[:] = np.clip(torque, -LIMITS.numpy(), LIMITS.numpy())
                    mujoco.mj_step(env.model, env.data)
                    mujoco.mj_forward(env.model, env.data)
                angles = robot_info_update(env.data)["euler"]
                fallen = (max(abs(angles["pitch"]), abs(angles["roll"])) > .8
                          or env.data.xpos[env.torso, 2] < .12)
            elapsed = (k + 1) * env.dt
            if fallen:
                status = "fallen"
                xy = None  # 训练环境已重置，不把重置后的原点当作终点
                break
            xy = env.data.xpos[env.torso, :2].copy()
            speed = np.linalg.norm(xy - before) / env.dt
            distance = np.linalg.norm(xy - target)
            yaw_rate = abs(env.data.sensor("angular_vel").data[2])
            inside = distance <= .03 and speed <= .03 and yaw_rate <= .1
            streak = streak + 1 if inside else 0
            if win is not None:
                win.sync()
                time.sleep(max(0., env.dt - (time.monotonic() - start)))
            if streak * env.dt >= 2.:
                status = "settled"
                break
    return dict(target=list(target), status=status, seconds=round(elapsed, 2),
                final_xy=None if xy is None else xy.tolist(),
                error_m=None if xy is None else float(np.linalg.norm(xy - target)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("--seconds", type=float, default=40.)
    parser.add_argument("--lqr", action="store_true", help="同一外环使用传统 LQR/VMC")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if not np.isfinite([args.x, args.y, args.seconds]).all() or args.seconds <= 0:
        parser.error("坐标须有限，时长须为正")
    torch.set_num_threads(1)
    env = ResidualMovementEnv(sample_commands=False)
    actor = None
    if not args.lqr:
        runner = OnPolicyRunner(env, movement_train_cfg(), device="cpu")
        runner.load(str(CHECKPOINT), map_location="cpu")
        actor = runner.get_inference_policy()
    print(json.dumps(navigate(env, actor,
                              (args.x, args.y), args.seconds, args.viewer)))


if __name__ == "__main__":
    main()
