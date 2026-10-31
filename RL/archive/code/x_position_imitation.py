"""世界 x 位置外环 + LQR 标注 + 22维策略模仿。默认朝向、小位移测试。

训练：python RL/archive/code/x_position_imitation.py
评估：python RL/archive/code/x_position_imitation.py --evaluate --checkpoint <policy.pt> --target-x 0.1 --viewer
"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from pathlib import Path
import argparse, json, time
import numpy as np
import torch, mujoco
from RL.common.lqr_imitation import ROOT, RL_DIR, LIMITS, LQRTeacher, PolicyNetwork, fit, robot_info_update
from RL.common.observations import build_observation, OBSERVATION_ORDER, OBSERVATION_SCALES

ORDER = OBSERVATION_ORDER + ['world_x_error', 'command_velocity']
SCALES = OBSERVATION_SCALES + [.2, .15]


class XPositionController:
    """error=target_x-current_x；积分补偿持续偏差，限制积分和速度。"""
    def __init__(self):
        self.integral = 0.

    def update(self, error, dt):
        self.integral = float(np.clip(self.integral + error * dt, -.2, .2))
        return float(np.clip(error + .6 * self.integral, -.15, .15))


def rollout(model, policy=None, target_x=0., initial_x=0., seconds=10., viewer=False):
    data = mujoco.MjData(model)
    data.qpos[0] = initial_x
    mujoco.mj_forward(model, data)
    torso = model.body('torso').id
    teacher, outer = LQRTeacher(float(model.opt.timestep)), XPositionController()
    xs, ys, records = [], [], []
    window = None
    if viewer:
        from mujoco import viewer as mjviewer
        window = mjviewer.launch_passive(model, data)
    failed = False
    try:
        for _ in range(round(seconds/model.opt.timestep)):
            if window is not None and not window.is_running():
                break
            start = time.perf_counter()
            info = robot_info_update(data)
            error = target_x-float(data.xpos[torso,0])
            command = outer.update(error, float(model.opt.timestep))
            # 不让旧轮子位置目标与新的世界x位置外环相互冲突。
            wheel_position = (info['pos']['leftwheel']-info['pos']['rightwheel'])/2*.06
            teacher.ll.x_b_set = teacher.lr.x_b_set = wheel_position
            teacher.ll.v_set = teacher.lr.v_set = command
            target = teacher.action(info)
            base = build_observation(info,data,torso,3.)  # 与原策略时间输入兼容
            obs = torch.cat([base,torch.tensor([[error/.2,command/.15]],dtype=torch.float32)],dim=1)
            xs.append(obs[0]);ys.append(torch.from_numpy(target))
            if policy is None:
                data.ctrl[:]=target
            else:
                with torch.no_grad():
                    action=policy(obs).tanh()*LIMITS
                if not torch.isfinite(action).all():
                    raise ValueError('Nonfinite action')
                data.ctrl[:]=action[0].numpy()
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
            info=robot_info_update(data)
            p,r=info['euler']['pitch'],info['euler']['roll']
            h=float(data.xpos[torso,2])
            records.append([float(data.time),float(data.xpos[torso,0]),float(data.xpos[torso,1]),h,p,r,command])
            failed=abs(p)>.8 or abs(r)>.8 or h<.12
            if window is not None:
                window.sync();time.sleep(max(0.,model.opt.timestep-(time.perf_counter()-start)))
            if failed:break
    finally:
        if window is not None:window.close()
    a=np.array(records)
    last=a[a[:,0]>=max(0.,seconds-5.)]
    summary=dict(target_x=target_x,initial_x=initial_x,duration=float(data.time),
        standing=not failed and data.time>=seconds-1e-8,final_x=float(a[-1,1]),
        final_error=float(target_x-a[-1,1]),final_y=float(a[-1,2]),
        last5_x_rmse=float(np.sqrt(np.mean((last[:,1]-target_x)**2))) if len(last) else None,
        max_pitch=float(abs(a[:,4]).max()),min_height=float(a[:,3].min()))
    return torch.stack(xs),torch.stack(ys),summary,a


def evaluate(model,policy,seconds=30.):
    return [rollout(model,policy,t,seconds=seconds)[2] for t in (-.1,0.,.1)]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--target-x',type=float,default=0.)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--viewer',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(44)
    model=mujoco.MjModel.from_xml_path(str(ROOT/'LQR_VMC/mjcf/robot.xml'))
    policy=PolicyNetwork(22)
    if args.evaluate:
        if args.checkpoint is None:parser.error('--checkpoint is required')
        c=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        assert c['observation_order']==ORDER and c['observation_scales']==SCALES
        policy.load_state_dict(c['policy_state_dict']);policy.eval()
        print(json.dumps(rollout(model,policy,args.target_x,seconds=args.seconds,viewer=args.viewer)[2]))
        raise SystemExit
    output=RL_DIR/'archive/data/experiments'/('x_position_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    # 迁移已站稳的20维策略；新增两个输入的初始权重为0，初始动作保持一致。
    old=torch.load(RL_DIR/'archive/data/results/lqr_dagger_policy.pt',map_location='cpu',weights_only=True)['policy_state_dict']
    old['network.0.weight']=torch.cat([old['network.0.weight'],torch.zeros(32,2)],dim=1)
    policy.load_state_dict(old)
    train_states=[(t,x) for t in (-.1,0.,.1) for x in (-.04,0.,.04)]
    val_states=[(-.075,.02),(.025,-.02),(.075,.02)]
    x=y=vx=vy=None
    report=dict(teacher=evaluate(model,None),before=evaluate(model,policy),rounds=[])
    print('OUTPUT',output,flush=True);print('BASELINE',json.dumps(report),flush=True)
    for round_id in range(4):
        driver=None if round_id==0 else policy
        tx,ty,tvx,tvy=[],[],[],[]
        for split,states in [('train',train_states),('validation',val_states)]:
            for target,initial in states:
                sx,sy,_,_=rollout(model,driver,target,initial,seconds=10.)
                (tx if split=='train' else tvx).append(sx)
                (ty if split=='train' else tvy).append(sy)
        x=torch.cat(([x] if x is not None else [])+tx)
        y=torch.cat(([y] if y is not None else [])+ty)
        vx=torch.cat(([vx] if vx is not None else [])+tvx)
        vy=torch.cat(([vy] if vy is not None else [])+tvy)
        print('ROUND',round_id,'train',len(x),'validation',len(vx),flush=True)
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        history=fit(policy,optimizer,x,y,vx,vy,80)
        results=evaluate(model,policy)
        record=dict(round=round_id,training_samples=len(x),validation_samples=len(vx),loss=history,evaluation=results)
        report['rounds'].append(record)
        torch.save(dict(policy_state_dict=policy.state_dict(),observation_order=ORDER,observation_scales=SCALES,
                        policy_activation='silu',observation_dim=22,outer_controller=dict(kp=1.,ki=.6,integral_limit=.2,velocity_limit=.15),
                        torque_limits=LIMITS.tolist(),round=round_id),output/f'policy_round_{round_id}.pt')
        torch.save(dict(obs=x,torques=y,validation_obs=vx,validation_torques=vy),output/'dataset.pt')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('RESULT',json.dumps(record['evaluation']),flush=True)
    for t in (-.1,0.,.1):
        *_,trajectory=rollout(model,policy,t,seconds=30.)
        np.savetxt(output/f'final_target_{t}.csv',trajectory,delimiter=',',header='time,x,y,height,pitch,roll,command',comments='')
    print('DONE',output,flush=True)
