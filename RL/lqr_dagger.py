"""在学生策略实际访问的状态上查询 LQR，追加监督数据并重训。

不会执行奖励训练，不改变价值网络，也不覆盖首次模仿学习结果。
"""
from pathlib import Path
import json
import time
import torch
import mujoco
from lqr_imitation import (ROOT, OUTPUT, LIMITS, LQRTeacher, rollout, evaluate, fit,
                           PolicyNetwork, OBSERVATION_DIM)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(43)
    run_dir = OUTPUT.parent / ('lqr_dagger_' + time.strftime('%Y%m%d_%H%M%S'))
    run_dir.mkdir(parents=True, exist_ok=False)
    original = torch.load(OUTPUT / 'policy.pt', weights_only=True, map_location='cpu')
    dataset = torch.load(OUTPUT / 'dataset.pt', weights_only=True, map_location='cpu')
    policy = PolicyNetwork(OBSERVATION_DIM)
    policy.load_state_dict(original['policy_state_dict'])
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'LQR_VMC/mjcf/robot.xml'))
    x, y = dataset['obs'], dataset['lqr_torques']
    vx, vy = dataset['validation_obs'], dataset['validation_lqr_torques']
    # 采集集、验证集和最终评估集的初始姿态分开。
    train_states = [(0.,0.)] + [(p,r) for p in (-.04,-.02,.02,.04) for r in (-.02,0.,.02)]
    val_states = [(-.03,-.01),(-.01,.01),(.01,-.01),(.03,.01)]
    # 默认状态作为共同基准；其余四个姿态不参与训练及模型选择。
    test_states = [(0.,0.),(-.025,.015),(.025,-.015),(-.015,-.005),(.015,.005)]
    before = evaluate(model, policy, test_states)
    report = dict(source=str(OUTPUT / 'policy.pt'), before=before, rounds=[])
    print('OUTPUT',run_dir,flush=True)
    print('BEFORE',json.dumps(before),flush=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    for round_id in range(1,4):
        newx, newy, newvx, newvy = [], [], [], []
        collected = []
        for split, states in [('train',train_states),('validation',val_states)]:
            for p,r in states:
                # 网络执行动作；rollout 内的 LQR 只计算每一步的目标标签。
                sx, sy, result = rollout(model, policy, p, r)
                (newx if split=='train' else newvx).append(sx)
                (newy if split=='train' else newvy).append(sy)
                collected.append(dict(split=split, **result))
        added = sum(len(a) for a in newx)
        x,y = torch.cat([x,*newx]), torch.cat([y,*newy])
        vx,vy = torch.cat([vx,*newvx]), torch.cat([vy,*newvy])
        print(f'ROUND {round_id}: added={added}, total_train={len(x)}, validation={len(vx)}',flush=True)
        # fit 用验证误差选择本轮参数；从选中参数开始时重建优化器，避免动量状态不匹配。
        optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
        history = fit(policy, optimizer, x, y, vx, vy, 80)
        results = evaluate(model, policy, test_states)
        with torch.no_grad():
            mae = ((policy(vx).tanh()*LIMITS)-vy).abs().mean(dim=0).tolist()
        record = dict(round=round_id, added_samples=added, training_samples=len(x),
                      validation_samples=len(vx), collection=collected,
                      validation_loss_history=history, validation_torque_mae=mae, evaluation=results)
        report['rounds'].append(record)
        checkpoint = dict(original, algorithm='dagger_behavior_cloning',
                          policy_state_dict=policy.state_dict(), training_samples=len(x),
                          epochs_per_round=80, dagger_round=round_id, source_policy=report['source'])
        torch.save(checkpoint, run_dir / f'policy_round_{round_id}.pt')
        torch.save(dict(obs=x,lqr_torques=y,validation_obs=vx,validation_lqr_torques=vy,
                        observation_order=dataset['observation_order'],
                        observation_scales=dataset['observation_scales']),run_dir/'dataset.pt')
        (run_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('RESULT',json.dumps(dict(round=round_id,evaluation=results,mae=mae)),flush=True)
    print('DONE',run_dir,flush=True)
