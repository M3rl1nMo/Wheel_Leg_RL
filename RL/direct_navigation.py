"""独立的目标相对位置→力矩模仿实验；评估时不创建或调用位置外环。

python RL/direct_navigation.py --train
python RL/direct_navigation.py --evaluate --checkpoint PATH --viewer
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, time
import mujoco
import numpy as np
import torch
from networks import PolicyNetwork
from observations import build_observation, OBSERVATION_ORDER, OBSERVATION_SCALES
from lqr_imitation import ROOT, RL_DIR, LIMITS, robot_info_update, fit

ORDER=OBSERVATION_ORDER[:19]+['target_forward_error','target_left_error']+[f'previous_torque_{i}' for i in range(6)]
CONFIG=dict(format_version=1,controller='direct_goal_to_torque',observation_dim=27,
    observation_order=ORDER,observation_scales=OBSERVATION_SCALES[:19]+[.5,.5]+LIMITS.tolist(),
    policy_activation='silu',hidden_units=32,action_transform='tanh_then_scale',
    torque_limits=LIMITS.tolist(),scene='flat_no_obstacles',runtime_position_outer=False)
TRAIN_CASES=[((.3*math.cos(a),.3*math.sin(a)),0.) for a in np.arange(8)*math.pi/4]
TRAIN_CASES += [((.15,0.),.5),((0.,.15),-.5),((-.15,0.),2.5),((0.,-.15),-2.5)]
VAL_CASES=[((.22,.12),.4),((-.18,.22),-.4),((.12,-.22),1.),((-.2,-.12),-2.)]
TEST_CASES=[((.25,0.),0.),((0.,.25),0.),((-.25,0.),0.),((.18,-.18),0.),((.23,.14),1.),
    ((-.3310953371672254,.08359958753452192),3.00303934229726),
    ((-.17782487679726747,.5075895967697843),-.03714260467785824),
    ((.3319516791010715,.09400023297468756),-3.0785905526499002)]


def make_model():
    model=mujoco.MjModel.from_xml_path(str(ROOT/'LQR_VMC/mjcf/robot.xml'))
    for gid in range(model.ngeom):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid) or ''
        if name.startswith('ramp_surface'):
            model.geom_contype[gid]=0;model.geom_conaffinity[gid]=0;model.geom_rgba[gid,3]=0.
    return model


def observation(info,data,body,target,previous):
    # 只作坐标转换，不生成速度指令，不保存目标积分或保持模式。
    dx,dy=np.asarray(target)-data.xpos[body,:2];yaw=info['euler']['yaw']
    f=math.cos(yaw)*dx+math.sin(yaw)*dy
    l=-math.sin(yaw)*dx+math.cos(yaw)*dy
    state=build_observation(info,data,body,3.)[:,:19]
    obs=torch.cat([state,torch.tensor([[f/.5,l/.5]],dtype=torch.float32),previous[None]/LIMITS],dim=1)
    if not torch.isfinite(obs).all():raise ValueError('Nonfinite direct observation')
    return obs


def rollout(policy,case,seconds=40.,collect=False,beta=0.,viewer=False):
    if policy is None and not collect:raise ValueError('Evaluation requires a policy')
    if not 0<=beta<=1:raise ValueError('beta outside [0,1]')
    model=make_model();data=mujoco.MjData(model);target,yaw=case
    data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
    mujoco.mj_forward(model,data);body=model.body('torso').id;dt=float(model.opt.timestep)
    previous=torch.zeros(6);xs=[];ys=[];records=[];arrival=None;streak=0;failed=False
    teacher=outer=None
    if collect:
        # 教师只存在于数据采集分支；评估分支没有外环实例或教师动作。
        from planar_control import PlanarPositionController, PlanarLQRTeacher
        outer=PlanarPositionController(yaw);teacher=PlanarLQRTeacher(dt)
    window=None
    if viewer:
        from mujoco import viewer as mjviewer
        window=mjviewer.launch_passive(model,data)
    try:
        with torch.no_grad():
            for step in range(round(seconds/dt)):
                if window is not None and not window.is_running():break
                start=time.perf_counter();info=robot_info_update(data)
                obs=observation(info,data,body,target,previous)
                label=None
                if collect:
                    _,_,v,w=outer.update(data.xpos[body,:2],info['euler']['yaw'],target,dt)
                    label=torch.from_numpy(teacher.action(info,float(data.sensor('angular_vel').data[2]),v,w))
                    if step%5==0 or step<100:xs.append(obs[0]);ys.append(label)
                if policy is None:tau=label
                else:
                    tau=policy(obs)[0].tanh()*LIMITS
                    if collect and beta>0:tau=beta*label+(1-beta)*tau
                if not torch.isfinite(tau).all() or (tau.abs()>LIMITS+1e-5).any():raise ValueError('Invalid torque')
                xy=data.xpos[body,:2].copy();data.ctrl[:]=tau.numpy();previous=tau.clone()
                mujoco.mj_step(model,data);mujoco.mj_forward(model,data)
                info=robot_info_update(data);pitch=info['euler']['pitch'];roll=info['euler']['roll']
                distance=float(np.linalg.norm(np.asarray(target)-data.xpos[body,:2]))
                speed=float(np.linalg.norm(data.xpos[body,:2]-xy)/dt)
                yawrate=float(data.sensor('angular_vel').data[2]);height=float(data.xpos[body,2])
                failed=abs(pitch)>.8 or abs(roll)>.8 or height<.12
                inside=distance<=.03 and speed<=.03 and abs(yawrate)<=.1 and not failed
                streak=streak+1 if inside else 0
                if streak>=round(2/dt) and arrival is None:arrival=float(data.time)
                records.append([data.time,*data.xpos[body,:2],distance,speed,yawrate,pitch,roll,height])
                if window is not None:window.sync();time.sleep(max(0.,dt-(time.perf_counter()-start)))
                if failed:break
    finally:
        if window is not None:window.close()
    a=np.asarray(records);standing=not failed and len(a)==round(seconds/dt)
    result=dict(target=list(target),initial_yaw=yaw,standing=standing,settled=standing and streak>=round(2/dt),
        arrival_time=arrival,duration=float(data.time),final_distance=distance,final_speed=speed,
        max_pitch=float(abs(a[:,6]).max()),min_height=float(a[:,8].min()),runtime_position_outer=collect)
    return (torch.stack(xs) if xs else None,torch.stack(ys) if ys else None,result,a)


def evaluate(policy,cases=TEST_CASES):
    results=[rollout(policy,case)[2] for case in cases]
    return dict(cases=len(results),standing=sum(r['standing'] for r in results),successes=sum(r['settled'] for r in results),
        mean_final_distance=float(np.mean([r['final_distance'] for r in results])),
        mean_arrival_time=float(np.mean([r['arrival_time'] for r in results])) if all(r['arrival_time'] is not None for r in results) else None,results=results)


def initialize():
    # 迁移已学的状态处理权重，丢弃剩余时间、速度指令权重；新增上一力矩权重置零。
    source=RL_DIR/'results/planar_policy.pt';saved=torch.load(source,weights_only=True)
    weights=copy.deepcopy(saved['policy_state_dict']);old=weights['network.0.weight']
    new=torch.zeros(32,27);new[:,:19]=old[:,:19];new[:,19:21]=old[:,20:22]
    weights['network.0.weight']=new
    policy=PolicyNetwork(27);policy.load_state_dict(weights)
    return policy,source


def train(args):
    output=RL_DIR/'experiments'/('direct_imitation_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(parents=True)
    print('OUTPUT',output,flush=True)
    policy,source=initialize();source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(config=CONFIG,source=str(source),source_sha256=source_hash,seed=61,teacher='smoothed position outer + LQR',rounds=[])
    vx=[];vy=[];validation_teacher=[]
    for case in VAL_CASES:
        x,y,r,_=rollout(None,case,collect=True,beta=1.)
        vx.append(x);vy.append(y);validation_teacher.append(r)
    vx,vy=torch.cat(vx),torch.cat(vy);report['validation_teacher']=validation_teacher
    x=y=None;best=None;best_key=(-1,-1,-float('inf'))
    for round_id,beta in enumerate([1.,.5,.25,0.][:args.rounds]):
        tx=[];ty=[];traces=[]
        for case in TRAIN_CASES:
            sx,sy,r,_=rollout(None if round_id==0 else policy,case,collect=True,beta=beta)
            tx.append(sx);ty.append(sy);traces.append(r)
        x=torch.cat(([x] if x is not None else [])+tx);y=torch.cat(([y] if y is not None else [])+ty)
        print('COLLECT',json.dumps(dict(round=round_id,beta=beta,samples=len(x),standing=sum(t['standing'] for t in traces))),flush=True)
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
        history=fit(policy,optimizer,x,y,vx,vy,args.epochs)
        validation=evaluate(policy,VAL_CASES)
        record=dict(round=round_id,teacher_fraction=beta,train_samples=len(x),validation_loss=history,
                    training_rollouts=traces,validation=validation)
        report['rounds'].append(record)
        torch.save(dict(config=CONFIG,policy_state_dict=policy.state_dict(),round=round_id,source=str(source)),output/f'policy_round_{round_id}.pt')
        key=(validation['successes'],validation['standing'],-validation['mean_final_distance'])
        if key>best_key:best_key=key;best=copy.deepcopy(policy.state_dict());report['selected_round']=round_id
        print('VALIDATION',json.dumps({k:v for k,v in validation.items() if k!='results'}),flush=True)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    policy.load_state_dict(best);report['test']=evaluate(policy)
    torch.save(dict(config=CONFIG,obs=x,lqr_torques=y,validation_obs=vx,validation_lqr_torques=vy),output/'dataset.pt')
    torch.save(dict(config=CONFIG,policy_state_dict=policy.state_dict(),selected_round=report['selected_round'],
        test_successes=report['test']['successes'],test_cases=report['test']['cases']),output/'policy.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE',json.dumps(dict(output=str(output),test={k:v for k,v in report['test'].items() if k!='results'})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train',action='store_true');mode.add_argument('--evaluate',action='store_true')
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--target-x',type=float,default=.2);parser.add_argument('--target-y',type=float,default=.2)
    parser.add_argument('--initial-yaw',type=float,default=0.);parser.add_argument('--seconds',type=float,default=40.)
    parser.add_argument('--epochs',type=int,default=80);parser.add_argument('--rounds',type=int,default=4)
    args=parser.parse_args()
    if args.epochs<=0 or not 1<=args.rounds<=4 or args.seconds<2:parser.error('Invalid training/evaluation limits')
    torch.set_num_threads(1);torch.manual_seed(61)
    if args.train:train(args)
    else:
        if args.checkpoint is None:parser.error('--checkpoint required')
        saved=torch.load(args.checkpoint,weights_only=True)
        if saved['config']!=CONFIG:raise ValueError('Direct policy configuration mismatch')
        policy=PolicyNetwork(27);policy.load_state_dict(saved['policy_state_dict']);policy.eval()
        print(json.dumps(rollout(policy,((args.target_x,args.target_y),args.initial_yaw),seconds=args.seconds,viewer=args.viewer)[2]),flush=True)
