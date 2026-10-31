"""用 LQR 力矩监督训练策略网络；不训练价值网络。

训练：python RL/common/lqr_imitation.py
查看：python RL/common/lqr_imitation.py --evaluate --viewer
"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from pathlib import Path
import argparse
import copy
import csv
import json
import math
import sys
import time

import mujoco
import numpy as np
import torch

RL_DIR = _PROJECT_ROOT / "RL"
ROOT = RL_DIR.parent
sys.path.insert(0, str(ROOT / 'LQR_VMC'))
from robot_control_main import (PIDController, control_left_leg, control_right_leg,
    five_links_param, lqr_controller, RollLegController, robot_info_update)
from RL.common.networks import PolicyNetwork
from RL.common.observations import (build_observation, OBSERVATION_DIM,
                          OBSERVATION_ORDER, OBSERVATION_SCALES)

LIMITS = torch.tensor([1.8, 1.8, 1.8, 1.8, 1.5, 1.5])
OUTPUT = RL_DIR / 'archive/data/experiments' / 'lqr_imitation_silu'


class LQRTeacher:
    def __init__(self, dt):
        self.left, self.right = five_links_param(dt=dt), five_links_param(dt=dt)
        self.ll, self.lr = lqr_controller(dt=dt), lqr_controller(dt=dt)
        self.yl, self.yr = [PIDController(.5, 0., 1., 2., 0.) for _ in range(2)]
        self.pl, self.pr = [PIDController(700., 0., 5., 150., 0.) for _ in range(2)]
        self.bal = RollLegController(kp=1., ki=0., kd=0., max_out=.14, min_leg=.10, max_leg=.14)

    def action(self, info):
        lengths = self.bal.compute_leg_lengths(current_roll=-info['euler']['roll'], target_roll=0., base_length=.1)
        speed = (info['vel']['leftwheel'] - info['vel']['rightwheel']) / 2
        pos = (info['pos']['leftwheel'] - info['pos']['rightwheel']) / 2
        args = (-info['euler']['pitch'], -info['gyro_pitch'], info['gyro_yaw'], 0.)
        lc = control_left_leg(info, self.left, self.ll, self.yl, self.pl,
                             *args, lengths[0], pos, speed, .06, False, False)
        rc = control_right_leg(info, self.right, self.lr, self.yr, self.pr,
                              *args, lengths[1], pos, speed, .06, False, False)
        return np.array([rc[0], rc[1], lc[0], lc[1], rc[2], lc[2]], dtype=np.float32)


def rollout(model, policy=None, pitch=0., roll=0., viewer=False):
    """每一步先保存 obs 和 LQR 目标，再执行动作。policy=None 表示 LQR 执行。"""
    data = mujoco.MjData(model)
    # 自由关节的四元数顺序为 w,x,y,z；这里设置小幅初始姿态变化。
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    data.qpos[3:7] = [cr*cp, sr*cp, cr*sp, -sr*sp]
    mujoco.mj_forward(model, data)
    torso = model.body('torso').id
    teacher = LQRTeacher(float(model.opt.timestep))
    observations, targets = [], []
    window = None
    if viewer:
        from mujoco import viewer as mjviewer
        window = mjviewer.launch_passive(model, data)
    fallen = False
    try:
        for _ in range(round(3. / model.opt.timestep)):
            if window is not None and not window.is_running():
                break
            start = time.perf_counter()
            info = robot_info_update(data)
            obs = build_observation(info, data, torso, 3.)
            target = teacher.action(info)
            observations.append(obs[0]); targets.append(torch.from_numpy(target))
            if policy is None:
                data.ctrl[:] = target
            else:
                with torch.no_grad():
                    data.ctrl[:] = (torch.tanh(policy(obs)) * LIMITS)[0].numpy()
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            info = robot_info_update(data)
            fallen = abs(info['euler']['pitch']) > .8 or abs(info['euler']['roll']) > .8 or data.xpos[torso, 2] < .12
            if window is not None:
                window.sync()
                time.sleep(max(0., model.opt.timestep - (time.perf_counter()-start)))
            if fallen:
                break
    finally:
        if window is not None:
            window.close()
    return torch.stack(observations), torch.stack(targets), dict(
        pitch=pitch, roll=roll, duration=float(data.time), success=not fallen and data.time >= 3.-1e-8)


def evaluate(model, policy, states):
    return [rollout(model, policy, p, r)[2] for p, r in states]


def fit(policy, optimizer, x, y, vx, vy, epochs):
    best_loss = float('inf')
    best = None
    history = []
    for epoch in range(epochs):
        policy.train()
        order = torch.randperm(len(x))
        for ids in order.split(256):
            # 在力矩除以各自上限后的尺度上拟合，使六个电机都有可比的误差权重。
            predicted = torch.tanh(policy(x[ids]))
            loss = ((predicted - y[ids] / LIMITS) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            validation_loss = ((torch.tanh(policy(vx)) - vy / LIMITS) ** 2).mean().item()
        history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best = validation_loss, copy.deepcopy(policy.state_dict())
        if (epoch + 1) % 20 == 0:
            print(f'epoch={epoch+1}, validation_loss={validation_loss:.6f}', flush=True)
    policy.load_state_dict(best)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=OUTPUT / 'policy.pt', help='评估时加载的策略文件')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'LQR_VMC/mjcf/robot.xml'))
    policy = PolicyNetwork(OBSERVATION_DIM)
    if args.evaluate:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        policy.load_state_dict(saved['policy_state_dict'])
        policy.eval()
        print(rollout(model, policy, viewer=args.viewer)[2])
        raise SystemExit
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'policy.pt').exists():
        raise FileExistsError('Existing imitation result: preserve or move it before retraining.')
    train_states = [(0., 0.)] + [(p, r) for p in (-.04, -.02, .02, .04) for r in (-.02, 0., .02)]
    val_states = [(-.03, -.01), (-.01, .01), (.01, -.01), (.03, .01)]
    test_states = [(0.,0.),(-.025,.015),(.025,-.015),(-.015,-.005),(.015,.005)]
    tx, ty, vx, vy, teacher_results = [], [], [], [], []
    for split, states in [('train', train_states), ('validation', val_states)]:
        for pitch, roll in states:
            x, y, result = rollout(model, pitch=pitch, roll=roll)
            teacher_results.append(dict(split=split, **result))
            (tx if split == 'train' else vx).append(x)
            (ty if split == 'train' else vy).append(y)
    x, y, vx, vy = map(torch.cat, (tx, ty, vx, vy))
    torch.save(dict(obs=x, lqr_torques=y, validation_obs=vx, validation_lqr_torques=vy,
                    observation_order=OBSERVATION_ORDER, observation_scales=OBSERVATION_SCALES,
                    teacher_episodes=teacher_results), OUTPUT / 'dataset.pt')
    print(f'Collected train={len(x)}, validation={len(vx)}; teacher successes={sum(r["success"] for r in teacher_results)}/{len(teacher_results)}', flush=True)
    before = evaluate(model, policy, test_states)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    history = fit(policy, optimizer, x, y, vx, vy, args.epochs)
    after = evaluate(model, policy, test_states)
    with torch.no_grad():
        mae = ((torch.tanh(policy(vx)) * LIMITS) - vy).abs().mean(dim=0).tolist()
    report = dict(train_samples=len(x), validation_samples=len(vx), teacher=teacher_results,
                  before=before, after=after, validation_loss_history=history, validation_torque_mae=mae)
    checkpoint = dict(format_version=1, algorithm='behavior_cloning', policy_state_dict=policy.state_dict(),
                      observation_dim=OBSERVATION_DIM, observation_order=OBSERVATION_ORDER,
                      observation_scales=OBSERVATION_SCALES, policy_activation='silu',
                      torque_limits=LIMITS.tolist(), action_transform='tanh_then_scale',
                      training_samples=len(x), epochs=args.epochs)
    torch.save(checkpoint, OUTPUT / 'policy.pt')
    (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(before=before, after=after, validation_torque_mae=mae)), flush=True)
