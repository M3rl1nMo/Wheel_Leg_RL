"""同一模型、力矩上限和初始姿态下比较 PPO 与项目 LQR/VMC 站立控制器。"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from RL.common.lqr_imitation import LQRTeacher, LIMITS, robot_info_update
from ..config import TRAIN_CFG
from ..environment import StandingEnv


@torch.inference_mode()
def trial(env, policy, pitch, roll):
    env.reset()
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    env.data.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
    mujoco.mj_forward(env.model, env.data)
    origin = env.data.xpos[env.torso, :2].copy()
    teacher = LQRTeacher(env.model.opt.timestep) if policy is None else None
    max_drift = 0.
    fallen = False
    for step in range(env.max_episode_length * env.cfg["substeps"]):
        if teacher is not None:
            torque = teacher.action(robot_info_update(env.data))
            env.data.ctrl[:] = np.clip(torque, -LIMITS.numpy(), LIMITS.numpy())
        else:
            if step % env.cfg["substeps"] == 0:
                targets = env.action_to_targets(policy(env.get_observations()))
            env.data.ctrl[:] = env.motor_torques(*targets)
        mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        info = robot_info_update(env.data)
        max_drift = max(max_drift, float(np.linalg.norm(env.data.xpos[env.torso, :2] - origin)))
        fallen = bool(max(abs(info["euler"]["pitch"]), abs(info["euler"]["roll"])) > 0.8
                      or env.data.xpos[env.torso, 2] < 0.12)
        if fallen:
            break
    return dict(pitch=pitch, roll=roll, stood=not fallen, seconds=round(env.data.time, 3),
                max_drift_m=round(max_drift, 4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = StandingEnv()
    runner = OnPolicyRunner(env, TRAIN_CFG, device="cpu")
    runner.load(str(args.checkpoint), map_location="cpu")
    policy = runner.get_inference_policy()
    cases = [(0, 0), (.05, 0), (-.05, 0), (0, .05), (0, -.05), (.05, -.05), (-.05, .05)]
    report = {}
    for name, controller in (("ppo", policy), ("lqr_vmc", None)):
        rows = [trial(env, controller, *case) for case in cases]
        report[name] = dict(cases=rows, standing=sum(x["stood"] for x in rows),
                            mean_seconds=float(np.mean([x["seconds"] for x in rows])))
        print(name, report[name]["standing"], "/", len(cases), report[name]["mean_seconds"])
    args.checkpoint.with_suffix(".standing.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
