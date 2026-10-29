"""27维无位置外环策略的 PPO 实验。奖励来自仿真，不采集 LQR 标签。"""
from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
import mujoco
from torch.distributions import Normal
import direct_navigation as direct
from networks import PolicyNetwork,ValueNetwork
from lqr_imitation import RL_DIR,LIMITS,robot_info_update
# 只复用优势计算和优化器更新，不使用带外环的环境或采集函数。
from planar_ppo import gae,update

REWARD=dict(progress=2.,potential=1.,epsilon=.05,time=.2,posture=.5,
            near_speed=.05,near_yaw=.05,torque_rate=.000005,success=5.,fall=10.)


def potential(distance):
    return REWARD['epsilon']/(distance+REWARD['epsilon'])


def shaping(old_distance,new_distance,gamma,terminated):
    # 真正终止后的吸收状态势能为0；时间截断继续bootstrap，不清零势能。
    return REWARD['potential']*(gamma*(0. if terminated else potential(new_distance))-potential(old_distance))


class DirectEnv:
    def __init__(self,stop_on_success=True,max_seconds=40.):
        self.model=direct.make_model();self.dt=float(self.model.opt.timestep)
        self.body=self.model.body('torso').id;self.max_steps=round(max_seconds/self.dt)
        self.stop_on_success=stop_on_success;self.gamma=math.exp(-self.dt/10.)

    def observe(self):
        return direct.observation(robot_info_update(self.data),self.data,self.body,self.target,self.previous)[0]

    def reset(self,case):
        self.target=np.array(case[0],dtype=np.float64);yaw=case[1]
        self.data=mujoco.MjData(self.model);self.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        mujoco.mj_forward(self.model,self.data)
        self.previous=torch.zeros(6);self.steps=0;self.streak=0;self.arrival=None
        self.distance=float(np.linalg.norm(self.target-self.data.xpos[self.body,:2]))
        self.total=0.;self.posture_sum=0.;self.rate_sum=0.;self.parts={}
        self.last_yaw=[]
        return self.observe()

    def step(self,torques):
        tau=torch.as_tensor(torques,dtype=torch.float32).clone()
        if tau.shape!=(6,) or not torch.isfinite(tau).all() or (tau.abs()>LIMITS+1e-6).any():
            raise ValueError('Invalid torques')
        rate=float(((tau-self.previous)/LIMITS).square().mean())/self.dt
        self.previous=tau
        xy=self.data.xpos[self.body,:2].copy();self.data.ctrl[:]=tau.numpy()
        mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        info=robot_info_update(self.data);pitch=info['euler']['pitch'];roll=info['euler']['roll']
        distance=float(np.linalg.norm(self.target-self.data.xpos[self.body,:2]))
        speed=float(np.linalg.norm(self.data.xpos[self.body,:2]-xy)/self.dt)
        yawrate=abs(float(self.data.sensor('angular_vel').data[2]));height=float(self.data.xpos[self.body,2])
        failed=abs(pitch)>.8 or abs(roll)>.8 or height<.12
        inside=distance<=.03 and speed<=.03 and yawrate<=.1 and not failed
        self.streak=self.streak+1 if inside else 0
        reached=self.streak>=round(2./self.dt)
        first_success=reached and self.arrival is None
        if first_success:self.arrival=float(self.data.time)
        self.steps+=1
        terminated=failed or (self.stop_on_success and first_success)
        truncated=self.steps>=self.max_steps and not terminated
        near=potential(distance)**2
        parts=dict(progress=REWARD['progress']*(self.distance-distance),
            potential=shaping(self.distance,distance,self.gamma,terminated),
            time=-REWARD['time']*self.dt,posture=-REWARD['posture']*(pitch*pitch+roll*roll)*self.dt,
            near_speed=-REWARD['near_speed']*near*(speed/.1)**2*self.dt,
            near_yaw=-REWARD['near_yaw']*near*(yawrate/.3)**2*self.dt,
            torque_rate=-REWARD['torque_rate']*rate,
            success=REWARD['success'] if first_success else 0.,fall=-REWARD['fall'] if failed else 0.)
        reward=sum(parts.values());self.total+=reward
        for k,v in parts.items():self.parts[k]=self.parts.get(k,0.)+v
        self.distance=distance;self.posture_sum+=(pitch*pitch+roll*roll)*self.dt;self.rate_sum+=rate
        self.last_yaw.append(yawrate)
        if len(self.last_yaw)>round(2/self.dt):self.last_yaw.pop(0)
        result=dict(duration=float(self.data.time),failed=failed,standing=not failed and self.steps>=self.max_steps,
            settled=reached and not failed,arrival_time=self.arrival,final_distance=distance,final_speed=speed,
            final_yawrate=yawrate,last2_mean_abs_yawrate=float(np.mean(self.last_yaw)),
            return_sum=self.total,reward_parts=self.parts.copy(),
            posture_rms=math.sqrt(self.posture_sum/max(self.data.time,self.dt)),
            torque_rate_rms=math.sqrt(self.rate_sum/max(self.data.time,self.dt)),runtime_position_outer=False)
        return self.observe(),reward,terminated,truncated,result


def evaluate(policy,cases):
    env=DirectEnv(stop_on_success=False);results=[]
    with torch.no_grad():
        for case in cases:
            obs=env.reset(case)
            while True:
                obs,_,term,trunc,r=env.step(policy(obs[None])[0].tanh()*LIMITS)
                if term or trunc:break
            results.append(dict(target=list(case[0]),initial_yaw=case[1],**r))
    return dict(cases=len(results),standing=sum(r['standing'] for r in results),successes=sum(r['settled'] and r['standing'] for r in results),
        mean_final_distance=float(np.mean([r['final_distance'] for r in results])),mean_reward=float(np.mean([r['return_sum'] for r in results])),
        mean_torque_rate_rms=float(np.mean([r['torque_rate_rms'] for r in results])),results=results)


def collect(policy,value,env,obs,count,std,lam,case_index):
    rows=[];episodes=[]
    with torch.no_grad():
        for _ in range(count):
            dist=Normal(policy(obs[None])[0],std);raw=dist.sample()
            val=value(obs[None]).squeeze();oldlog=dist.log_prob(raw).sum()
            nextobs,r,term,trunc,result=env.step(raw.tanh()*LIMITS)
            nv=torch.tensor(0.) if term else value(nextobs[None]).squeeze()
            rows.append((obs,raw,oldlog,val,nv,r,float(term),float(term or trunc)))
            obs=nextobs
            if term or trunc:
                episodes.append(dict(case_index=case_index,**result))
                case_index=(case_index+1)%len(direct.TRAIN_CASES);obs=env.reset(direct.TRAIN_CASES[case_index])
    columns=list(zip(*rows))
    batch={key:torch.stack(list(col)) for key,col in zip(('obs','raw','old_log','value','next_value'),columns[:5])}
    rewards,terminated,boundaries=[torch.tensor(c,dtype=torch.float32) for c in columns[5:]]
    advantage,returns=gae(rewards,batch['value'],batch['next_value'],terminated,boundaries,env.gamma,lam)
    batch['returns']=returns;batch['advantage']=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    return batch,obs,case_index,episodes


def score(e):
    return (e['standing'],e['successes'],-e['mean_final_distance'])


def preserves_baseline(e,baseline):
    return all((not b['standing'] or a['standing']) and (not b['settled'] or a['settled'])
               for a,b in zip(e['results'],baseline['results']))


def main(args):
    torch.set_num_threads(1);torch.manual_seed(72)
    source=RL_DIR/'results/direct_imitation_candidate.pt';saved=torch.load(source,weights_only=True)
    if saved['config']!=direct.CONFIG:raise ValueError('Requires direct 27-dimensional checkpoint')
    policy=PolicyNetwork(27);policy.load_state_dict(saved['policy_state_dict']);reference=copy.deepcopy(policy).eval()
    for p in reference.parameters():p.requires_grad_(False)
    value=ValueNetwork(27);env=DirectEnv();lam=math.exp(-env.dt/.5)
    actor_optimizer=torch.optim.Adam(policy.parameters(),lr=1e-5)
    critic_optimizer=torch.optim.Adam(value.parameters(),lr=.0003)
    output=RL_DIR/'experiments'/('direct_ppo_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(parents=True)
    report=dict(config=direct.CONFIG,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        hyper=dict(seed=72,std=args.std,steps=args.steps,iterations=args.iterations,gamma=env.gamma,gae_lambda=lam,
            actor_lr=1e-5,critic_lr=.0003,clip=.1,update_epochs=4,batch_size=512,reference_kl_weight=.01,reward=REWARD.copy()),updates=[])
    print('OUTPUT',output,flush=True)
    report['baseline_validation']=evaluate(policy,direct.VAL_CASES)
    report['baseline_test']=evaluate(policy,direct.TEST_CASES)
    print('BASELINE',json.dumps({k:v for k,v in report['baseline_test'].items() if k!='results'}),flush=True)
    save_report=lambda:(output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save_report()
    # 先冻结策略采集两批，更新价值网络。摔倒和未完成都保留，不要求先学会任务。
    report['critic_warmup']=[];obs=env.reset(direct.TRAIN_CASES[0]);case_index=0
    for _ in range(2):
        batch,obs,case_index,episodes=collect(policy,value,env,obs,args.steps,args.std,lam,case_index)
        with torch.no_grad():before=((value(batch['obs']).squeeze(-1)-batch['returns'])**2).mean().item()
        for epoch in range(20):
            for ids in torch.randperm(args.steps).split(512):
                loss=((value(batch['obs'][ids]).squeeze(-1)-batch['returns'][ids])**2).mean()
                critic_optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),.5);critic_optimizer.step()
        with torch.no_grad():after=((value(batch['obs']).squeeze(-1)-batch['returns'])**2).mean().item()
        report['critic_warmup'].append(dict(before=before,after=after,episodes=episodes))
    print('CRITIC_WARMUP',json.dumps([dict(before=r['before'],after=r['after'],episodes=len(r['episodes'])) for r in report['critic_warmup']]),flush=True)
    save_report();best=copy.deepcopy(policy.state_dict());best_score=score(report['baseline_validation']);best_iteration=0
    for iteration in range(1,args.iterations+1):
        batch,obs,case_index,episodes=collect(policy,value,env,obs,args.steps,args.std,lam,case_index)
        stats=update(policy,value,reference,actor_optimizer,critic_optimizer,batch,args.std)
        if not all(torch.isfinite(p).all() for p in policy.parameters()):raise RuntimeError('Nonfinite actor')
        evaluation=evaluate(policy,direct.VAL_CASES);safe=preserves_baseline(evaluation,report['baseline_validation'])
        if safe and score(evaluation)>best_score:
            best_score=score(evaluation);best=copy.deepcopy(policy.state_dict());best_iteration=iteration
        report['updates'].append(dict(iteration=iteration,training_steps=iteration*args.steps,evaluation=evaluation,preserves_baseline=safe,ppo=stats,episodes=episodes))
        torch.save(dict(config=direct.CONFIG,policy_state_dict=policy.state_dict(),value_state_dict=value.state_dict(),hyper=report['hyper'],
            actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),iteration=iteration),output/f'candidate_{iteration}.pt')
        print('UPDATE',json.dumps(dict(iteration=iteration,preserves_baseline=safe,**{k:v for k,v in evaluation.items() if k!='results'})),flush=True);save_report()
    policy.load_state_dict(best);report['selected_iteration']=best_iteration
    report['selected_test']=evaluate(policy,direct.TEST_CASES)
    accepted=best_iteration>0 and preserves_baseline(report['selected_test'],report['baseline_test']) and score(report['selected_test'])>score(report['baseline_test'])
    report['passed_pilot_gate']=accepted;report['exported_iteration']=best_iteration if accepted else 0
    if not accepted:policy.load_state_dict(reference.state_dict())
    torch.save(dict(config=direct.CONFIG,policy_state_dict=policy.state_dict(),exported_iteration=report['exported_iteration'],
        selected_iteration=best_iteration,passed_pilot_gate=accepted,hyper=report['hyper']),output/'selected_policy.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==report['source_sha256']
    save_report();print('DONE',json.dumps(dict(output=str(output),selected_iteration=best_iteration,exported_iteration=report['exported_iteration'],
        test={k:v for k,v in report['selected_test'].items() if k!='results'})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--iterations',type=int,default=12)
    parser.add_argument('--steps',type=int,default=8192);parser.add_argument('--std',type=float,default=.01)
    args=parser.parse_args()
    if args.iterations<1 or args.steps<2 or not math.isfinite(args.std) or args.std<=0:parser.error('Invalid run size or std')
    main(args)
