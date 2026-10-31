"""相同初态：1 s 站立、5 s 移动、3 s 停车；使用确定性动作。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from .config import movement_train_cfg, CHECKPOINT
from .movement import ResidualMovementEnv


@torch.inference_mode()
def trial(env, actor, v, w):
    env.reset()
    vx_samples, yaw_samples, stop_v, stop_w = [], [], [], []
    distance = yaw = 0.
    ended = False
    completed = 0
    for k in range(450):
        move = 50 <= k < 300
        env.command[:] = [v,w] if move else [0.,0.]
        obs = env.get_observations()
        before = env.data.xpos[env.torso,:2].copy()
        _, _, done, _ = env.step(actor(obs))
        if done.item():
            ended = True
            break
        completed += 1
        vx = float(env.data.sensor("linear_vel").data[0])
        wz = float(env.data.sensor("angular_vel").data[2])
        if move:
            delta = env.data.xpos[env.torso,:2] - before
            distance += float(delta[0])  # 世界 x 位移，直行任务衡量前进/倒退
            yaw += wz * env.dt
        if 100 <= k < 300:
            vx_samples.append(vx)
            yaw_samples.append(wz)
        if k >= 400:
            stop_v.append(float(np.linalg.norm(env.data.sensor("linear_vel").data[:2])))
            stop_w.append(abs(wz))
    def mean(xs):
        return float(np.mean(xs)) if xs else None
    return dict(v_cmd=v,w_cmd=w,stood=not ended,seconds=completed*env.dt,
                mean_v=mean(vx_samples),mean_w=mean(yaw_samples),
                move_dx=distance,turn_rad=yaw,stop_speed=mean(stop_v),stop_yawrate=mean(stop_w))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints",nargs="*",type=Path,default=[CHECKPOINT])
    args=parser.parse_args()
    torch.set_num_threads(1)
    env=ResidualMovementEnv(sample_commands=False)
    runner=OnPolicyRunner(env,movement_train_cfg(),device="cpu")
    for ckpt in args.checkpoints:
        runner.load(str(ckpt),map_location="cpu")
        actor=runner.get_inference_policy()
        results=[trial(env,actor,v,w) for v,w in [(0.,0.),(.15,0.),(-.15,0.),
                                                   (0.,.5),(0.,-.5),(.12,.4)]]
        ckpt.with_suffix(".movement.json").write_text(json.dumps(results,indent=2))
        print(ckpt.name, [(r["stood"],round(r["move_dx"],2),round(r["turn_rad"],2)) for r in results])


if __name__ == "__main__":
    main()
