"""24维策略 + 平面位置外环。在无障碍平面中学习到点控制。

python RL/planar_navigation.py --train
python RL/planar_navigation.py --evaluate --checkpoint PATH --target-x .2 --target-y .2 --viewer
python RL/planar_navigation.py --teacher --target-x 0 --target-y .3 --viewer
"""
from pathlib import Path
import argparse
import json
import math
import time

import mujoco
import numpy as np
import torch
from lqr_imitation import ROOT, RL_DIR, LIMITS, PolicyNetwork, fit, robot_info_update
from observations import build_observation, OBSERVATION_ORDER, OBSERVATION_SCALES
from planar_control import PlanarPositionController, PlanarLQRTeacher

ORDER = OBSERVATION_ORDER + ['target_forward_error', 'target_left_error', 'command_velocity', 'command_turn_rate']
SCALES = OBSERVATION_SCALES + [.5, .5, .15, .6]
CONFIG = dict(scene='flat_no_obstacles',observation_horizon=3.,position_tolerance=.03,
              speed_tolerance=.03,yaw_rate_tolerance=.1,settle_seconds=2.,policy_activation='silu',
              observation_dim=24,observation_order=ORDER,observation_scales=SCALES,
              outer_controller=dict(kp=1.,ki=.6,integral_limit=.2,velocity_limit=.15,
              turn_kp=1.5,turn_limit=.6,hold_entry=.025,hold_exit=.15))


def make_model():
    model = mujoco.MjModel.from_xml_path(str(ROOT/'LQR_VMC/mjcf/robot.xml'))
    # 平面到点实验不包含避障；仅在内存中禁用原场景斜坡，不改原 XML。
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or ''
        if name.startswith('ramp_surface'):
            model.geom_contype[gid]=0
            model.geom_conaffinity[gid]=0
            model.geom_rgba[gid,3]=0.
    return model


def rollout(model,policy=None,target=(.3,0.),initial_yaw=0.,seconds=40.,viewer=False,collect=True):
    data=mujoco.MjData(model)
    data.qpos[3:7]=[math.cos(initial_yaw/2),0.,0.,math.sin(initial_yaw/2)]
    mujoco.mj_forward(model,data)
    torso=model.body('torso').id
    dt=float(model.opt.timestep)
    teacher=PlanarLQRTeacher(dt)
    outer=PlanarPositionController(initial_yaw)
    xs,ys,records=[],[],[]
    window=None
    if viewer:
        from mujoco import viewer as mjviewer
        window=mjviewer.launch_passive(model,data)
    failed=False
    try:
        for _ in range(round(seconds/dt)):
            if window is not None and not window.is_running():break
            start=time.perf_counter()
            info=robot_info_update(data)
            forward,left,velocity,turn=outer.update(data.xpos[torso,:2],info['euler']['yaw'],target,dt)
            # gyro传感器以局部xyz输出，绕z的分量才用于转向反馈。
            yaw_rate=float(data.sensor('angular_vel').data[2])
            label=teacher.action(info,yaw_rate,velocity,turn)
            obs=torch.cat([build_observation(info,data,torso,3.),
                           torch.tensor([[forward/.5,left/.5,velocity/.15,turn/.6]],dtype=torch.float32)],dim=1)
            if collect:
                xs.append(obs[0]);ys.append(torch.from_numpy(label))
            if policy is None:
                data.ctrl[:]=label
            else:
                with torch.no_grad():
                    action=policy(obs).tanh()*LIMITS
                if not torch.isfinite(action).all():raise ValueError('Nonfinite action')
                data.ctrl[:]=action[0].numpy()
            previous_xy=data.xpos[torso,:2].copy()
            mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
            info=robot_info_update(data)
            pitch,roll=info['euler']['pitch'],info['euler']['roll']
            height=float(data.xpos[torso,2])
            speed=float(np.linalg.norm(data.xpos[torso,:2]-previous_xy)/dt)
            distance=float(np.linalg.norm(np.array(target)-data.xpos[torso,:2]))
            yaw_rate=float(data.sensor('angular_vel').data[2])
            records.append([float(data.time),*data.xpos[torso].tolist(),pitch,roll,info['euler']['yaw'],distance,speed,yaw_rate,velocity,turn])
            failed=abs(pitch)>.8 or abs(roll)>.8 or height<.12
            if window is not None:
                window.sync();time.sleep(max(0.,dt-(time.perf_counter()-start)))
            if failed:break
    finally:
        if window is not None:window.close()
    a=np.array(records)
    standing=not failed and len(records)==round(seconds/dt)
    tail=a[-round(CONFIG['settle_seconds']/dt):]
    settled=standing and len(tail)==round(CONFIG['settle_seconds']/dt) and bool(
        (tail[:,7]<=CONFIG['position_tolerance']).all() and
        (tail[:,8]<=CONFIG['speed_tolerance']).all() and
        (abs(tail[:,9])<=CONFIG['yaw_rate_tolerance']).all())
    result=dict(target=list(target),initial_yaw=initial_yaw,duration=float(data.time),
                standing=standing,settled=settled,final_xy=a[-1,1:3].tolist(),
                final_distance=float(a[-1,7]),final_speed=float(a[-1,8]),
                last2_max_distance=float(tail[:,7].max()),last2_max_speed=float(tail[:,8].max()),
                last2_max_yaw_rate=float(abs(tail[:,9]).max()),max_pitch=float(abs(a[:,4]).max()),min_height=float(a[:,3].min()))
    return (torch.stack(xs) if xs else None,torch.stack(ys) if ys else None,result,a)


def evaluate(model,policy,cases,seconds=40.):
    return [rollout(model,policy,t,y,seconds,collect=False)[2] for t,y in cases]


def initialize_policy():
    # 从已验证的x控制模型迁移，新增侧向误差和转向指令权重先设0。
    source=RL_DIR/'results/x_position_policy.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    weights=checkpoint['policy_state_dict']
    old=weights['network.0.weight']
    new=torch.zeros(32,24)
    new[:,:20]=old[:,:20]
    new[:,20]=old[:,20]*2.5  # x误差原尺度.2，新前向误差尺度.5
    new[:,22]=old[:,21]
    weights['network.0.weight']=new
    policy=PolicyNetwork(24);policy.load_state_dict(weights)
    return policy,str(source)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--evaluate',action='store_true')
    mode.add_argument('--teacher',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--target-x',type=float,default=.2)
    parser.add_argument('--target-y',type=float,default=.2)
    parser.add_argument('--initial-yaw',type=float,default=0.,help='radians')
    parser.add_argument('--seconds',type=float,default=40.)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--rounds',type=int,default=3,help='包括首轮LQR模仿')
    args=parser.parse_args()
    if args.seconds<2. or args.epochs<=0 or args.rounds<=0:parser.error('seconds>=2, epochs>0, rounds>0 required')
    torch.set_num_threads(1);torch.manual_seed(45)
    model=make_model()
    if not args.train:
        policy=None
        if args.evaluate:
            if args.checkpoint is None:parser.error('--checkpoint required')
            saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
            if saved['config']!=CONFIG:raise ValueError('Checkpoint configuration mismatch')
            policy=PolicyNetwork(24);policy.load_state_dict(saved['policy_state_dict']);policy.eval()
        result=rollout(model,policy,(args.target_x,args.target_y),args.initial_yaw,args.seconds,args.viewer,False)[2]
        print(json.dumps(result),flush=True)
        raise SystemExit
    policy,source=initialize_policy()
    output=RL_DIR/'experiments'/('planar_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    # 训练状态覆盖前后、左右、斜向目标；验证目标和朝向单独设置。
    train_cases=[((.3*math.cos(a),.3*math.sin(a)),0.) for a in np.arange(8)*math.pi/4]
    train_cases += [((.15,0.),.5),((0.,.15),-.5),((-.15,0.),.5),((0.,-.15),-.5)]
    val_cases=[((.22,.12),.4),((-.18,.22),-.4),((.12,-.22),1.)]
    test_cases=[((.25,0.),0.),((0.,.25),0.),((-.25,0.),0.),((.18,-.18),0.),((.23,.14),1.)]
    print('OUTPUT',output,flush=True)
    teacher_results=evaluate(model,None,test_cases)
    print('TEACHER',json.dumps(teacher_results),flush=True)
    if not all(r['standing'] for r in teacher_results):raise RuntimeError('Teacher falls; stop before training')
    report=dict(config=CONFIG,source=source,teacher=teacher_results,rounds=[])
    x=y=vx=vy=None
    for round_id in range(args.rounds):
        driver=None if round_id==0 else policy
        tx,ty,tvx,tvy=[],[],[],[]
        for split,cases in [('train',train_cases),('validation',val_cases)]:
            for target,yaw in cases:
                sx,sy,_,_=rollout(model,driver,target,yaw,seconds=20.)
                (tx if split=='train' else tvx).append(sx)
                (ty if split=='train' else tvy).append(sy)
        x=torch.cat(([x] if x is not None else [])+tx);y=torch.cat(([y] if y is not None else [])+ty)
        vx=torch.cat(([vx] if vx is not None else [])+tvx);vy=torch.cat(([vy] if vy is not None else [])+tvy)
        print('ROUND',round_id,'train',len(x),'validation',len(vx),flush=True)
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        history=fit(policy,optimizer,x,y,vx,vy,args.epochs)
        results=evaluate(model,policy,test_cases)
        record=dict(round=round_id,train_samples=len(x),validation_samples=len(vx),loss=history,evaluation=results)
        report['rounds'].append(record)
        torch.save(dict(policy_state_dict=policy.state_dict(),config=CONFIG,source=source,round=round_id),output/f'policy_round_{round_id}.pt')
        torch.save(dict(obs=x,lqr_torques=y,validation_obs=vx,validation_lqr_torques=vy,config=CONFIG),output/'dataset.pt')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('RESULT',json.dumps(results),flush=True)
    print('DONE',output,flush=True)
