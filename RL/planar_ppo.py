"""从已验证的 SiLU 模仿策略开始，小规模 PPO 微调。

python RL/planar_ppo.py --iterations 4 --steps 8192
默认保留原成功模型，候选模型和评估结果存入独立目录。
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import time

import mujoco
import numpy as np
import torch
from torch.distributions import Normal
from networks import PolicyNetwork, ValueNetwork
from planar_navigation import CONFIG, ORDER, make_model, validate_policy_config
from planar_control import PlanarPositionController, PlanarLQRTeacher
from observations import build_observation
from lqr_imitation import RL_DIR, LIMITS, robot_info_update

# 所有物理步保持0.002秒；率惩罚乘dt，避免按步数隐式改变时间权重。
REWARD = dict(progress=2., time=.05, posture=.5, torque_rate=.00002,
              success=5., fall=10.)
TRAIN_CASES = [((.3,0.),0.),((0.,.3),0.),((-.3,0.),0.),((.2,-.2),.5)]
EVAL_CASES = [((.25,0.),0.),((0.,.25),0.),((-.25,0.),0.),((.18,-.18),0.),((.23,.14),1.)]
HOLDOUT_CASES = [((-.3310953371672254,.08359958753452192),3.00303934229726),
                 ((-.17782487679726747,.5075895967697843),-.03714260467785824),
                 ((.3319516791010715,.09400023297468756),-3.0785905526499002)]


class PlanarEnv:
    """外环每个物理步只更新一次，策略和LQR共用同一观测及评价规则。"""
    def __init__(self, max_seconds=40., stop_on_success=True):
        self.model=make_model()
        self.dt=float(self.model.opt.timestep)
        self.body=self.model.body('torso').id
        self.max_steps=round(max_seconds/self.dt)
        self.stop_on_success=stop_on_success

    def observe(self):
        self.info=robot_info_update(self.data)
        self.commands=self.outer.update(self.data.xpos[self.body,:2],self.info['euler']['yaw'],self.target,self.dt)
        f,l,v,w=self.commands
        self.obs=torch.cat([build_observation(self.info,self.data,self.body,3.),
                            torch.tensor([[f/.5,l/.5,v/.15,w/.6]],dtype=torch.float32)],dim=1)[0]
        return self.obs

    def reset(self, case):
        target,yaw=case
        self.target=np.array(target,dtype=np.float64)
        self.data=mujoco.MjData(self.model)
        self.data.qpos[3:7]=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        mujoco.mj_forward(self.model,self.data)
        self.outer=PlanarPositionController(yaw)
        self.teacher=PlanarLQRTeacher(self.dt)
        self.previous=np.zeros(6,dtype=np.float64)
        self.steps=0;self.streak=0;self.arrival=None;self.failed=False
        self.distance=float(np.linalg.norm(self.target-self.data.xpos[self.body,:2]))
        self.return_sum=0.;self.posture_integral=0.;self.change_sum=0.;self.rate_integral=0.
        self.parts={k:0. for k in REWARD}
        return self.observe()

    def teacher_action(self):
        _,_,v,w=self.commands
        return self.teacher.action(self.info,float(self.data.sensor('angular_vel').data[2]),v,w)

    def step(self, torques):
        tau=np.asarray(torques,dtype=np.float64)
        if tau.shape!=(6,) or not np.isfinite(tau).all():raise ValueError('Invalid torques')
        if (abs(tau)>LIMITS.numpy()+1e-6).any():raise ValueError('Torque limit exceeded')
        delta=(tau-self.previous)/LIMITS.numpy()
        change=float(np.mean(delta**2))
        rate=change/self.dt  # integral ((delta_tau/limit)/dt)^2 dt
        self.previous=tau.copy()
        xy=self.data.xpos[self.body,:2].copy()
        self.data.ctrl[:]=tau
        mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data)
        info=robot_info_update(self.data)
        pitch,roll=info['euler']['pitch'],info['euler']['roll']
        height=float(self.data.xpos[self.body,2])
        distance=float(np.linalg.norm(self.target-self.data.xpos[self.body,:2]))
        speed=float(np.linalg.norm(self.data.xpos[self.body,:2]-xy)/self.dt)
        yawrate=abs(float(self.data.sensor('angular_vel').data[2]))
        self.failed=abs(pitch)>.8 or abs(roll)>.8 or height<.12
        inside=distance<=.03 and speed<=.03 and yawrate<=.1 and not self.failed
        self.streak=self.streak+1 if inside else 0
        reached=self.streak>=round(2./self.dt)
        first_success=reached and self.arrival is None
        if first_success:self.arrival=float(self.data.time)
        posture=(pitch*pitch+roll*roll)*self.dt
        parts=dict(progress=REWARD['progress']*(self.distance-distance),
                   time=-REWARD['time']*self.dt,posture=-REWARD['posture']*posture,
                   torque_rate=-REWARD['torque_rate']*rate,
                   success=REWARD['success'] if first_success else 0.,
                   fall=-REWARD['fall'] if self.failed else 0.)
        reward=sum(parts.values())
        self.return_sum+=reward
        for k in parts:self.parts[k]+=parts[k]
        self.distance=distance;self.steps+=1
        self.posture_integral+=posture;self.change_sum+=change;self.rate_integral+=rate
        terminated=self.failed or (self.stop_on_success and first_success)
        truncated=self.steps>=self.max_steps and not terminated
        next_obs=self.observe()
        summary=dict(duration=float(self.data.time),failed=self.failed,arrival_time=self.arrival,
                     final_distance=distance,final_speed=speed,
                     settled=reached and not self.failed,return_sum=self.return_sum,
                     posture_rms=math.sqrt(self.posture_integral/max(self.data.time,self.dt)),
                     normalized_torque_change_sum=self.change_sum,
                     normalized_torque_rate_rms=math.sqrt(self.rate_integral/max(self.data.time,self.dt)),
                     reward_parts=self.parts.copy())
        return next_obs,reward,terminated,truncated,summary


def gae(rewards, values, next_values, terminated, boundaries, gamma, lam):
    """真正结束不bootstrap；时间截断保留next_value，但不跨重置传播优势。"""
    advantages=torch.zeros_like(rewards)
    tail=torch.tensor(0.)
    for i in reversed(range(len(rewards))):
        delta=rewards[i]+gamma*next_values[i]*(1.-terminated[i])-values[i]
        tail=delta+gamma*lam*(1.-boundaries[i])*tail
        advantages[i]=tail
    return advantages,advantages+values


def evaluate(policy, cases=EVAL_CASES, teacher=False):
    env=PlanarEnv(stop_on_success=False)
    results=[]
    for case in cases:
        obs=env.reset(case)
        with torch.no_grad():
            while True:
                tau=env.teacher_action() if teacher else (policy(obs[None]).tanh()*LIMITS)[0].numpy()
                obs,_,terminated,truncated,summary=env.step(tau)
                if terminated or truncated:break
        results.append(dict(target=list(case[0]),initial_yaw=case[1],**summary))
    summary=dict(successes=sum(r['settled'] and not r['failed'] for r in results),cases=len(results),
                 mean_reward=float(np.mean([r['return_sum'] for r in results])),
                 mean_posture_rms=float(np.mean([r['posture_rms'] for r in results])),
                 mean_torque_rate_rms=float(np.mean([r['normalized_torque_rate_rms'] for r in results])),
                 mean_arrival_time=float(np.mean([r['arrival_time'] for r in results])) if all(r['arrival_time'] is not None for r in results) else None,
                 results=results)
    return summary


def collect(policy,value,env,obs,count,std,gamma,lam,case_index):
    records=[];episodes=[]
    with torch.no_grad():
        for _ in range(count):
            mu=policy(obs[None])[0]
            distribution=Normal(mu,std)
            raw=distribution.sample()
            oldlog=distribution.log_prob(raw).sum()
            val=value(obs[None]).squeeze()
            nextobs,reward,term,trunc,summary=env.step((raw.tanh()*LIMITS).numpy())
            nextval=torch.tensor(0.) if term else value(nextobs[None]).squeeze()
            records.append((obs,raw,oldlog,val,nextval,reward,float(term),float(term or trunc)))
            obs=nextobs
            if term or trunc:
                episodes.append(summary)
                case_index=(case_index+1)%len(TRAIN_CASES)
                obs=env.reset(TRAIN_CASES[case_index])
    columns=list(zip(*records))
    batch={name:torch.stack(list(col)) for name,col in zip(('obs','raw','old_log','value','next_value'),columns[:5])}
    rewards,term,boundary=[torch.tensor(col,dtype=torch.float32) for col in columns[5:]]
    adv,returns=gae(rewards,batch['value'],batch['next_value'],term,boundary,gamma,lam)
    batch['advantage']=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    batch['returns']=returns
    return batch,obs,case_index,episodes


def warmup_critic(policy,value,std,gamma):
    env=PlanarEnv()
    xs,ys=[],[]
    summaries=[]
    for case in TRAIN_CASES:
        obs=env.reset(case);episode_obs=[];rewards=[]
        with torch.no_grad():
            while True:
                raw=Normal(policy(obs[None])[0],std).sample()
                episode_obs.append(obs)
                obs,r,term,trunc,summary=env.step((raw.tanh()*LIMITS).numpy())
                rewards.append(r)
                if term or trunc:break
        if trunc:raise RuntimeError('Exploration did not complete task in critic warmup; stop before PPO.')
        if summary['failed']:raise RuntimeError('Exploration caused a fall in critic warmup; stop before PPO.')
        ret=0.;returns=[]
        for r in reversed(rewards):ret=r+gamma*ret;returns.append(ret)
        xs.extend(episode_obs);ys.extend(reversed(returns));summaries.append(summary)
    x,y=torch.stack(xs),torch.tensor(ys,dtype=torch.float32)
    optimizer=torch.optim.Adam(value.parameters(),lr=.001)
    with torch.no_grad():before=((value(x).squeeze(-1)-y)**2).mean().item()
    for epoch in range(40):
        for ids in torch.randperm(len(x)).split(512):
            loss=((value(x[ids]).squeeze(-1)-y[ids])**2).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),.5);optimizer.step()
    with torch.no_grad():after=((value(x).squeeze(-1)-y)**2).mean().item()
    return dict(samples=len(x),before_mse=before,after_mse=after,episodes=summaries)


def update(policy,value,reference,actor_optimizer,critic_optimizer,batch,std,epochs=4):
    logs=[]
    # 使用固定采样的raw动作；tanh和缩放雅可比在新旧密度比中抵消。
    for epoch in range(epochs):
        with torch.no_grad():
            newlog=Normal(policy(batch['obs']),std).log_prob(batch['raw']).sum(-1)
            logratio=newlog-batch['old_log']
            kl=((logratio.exp()-1)-logratio).mean().item()
        if kl>.01:break
        for ids in torch.randperm(len(batch['obs'])).split(512):
            obs,raw,old=batch['obs'][ids],batch['raw'][ids],batch['old_log'][ids]
            mu=policy(obs)
            ratio=(Normal(mu,std).log_prob(raw).sum(-1)-old).exp()
            adv=batch['advantage'][ids]
            clipped=ratio.clamp(.9,1.1)
            with torch.no_grad():ref=reference(obs)
            anchor=((mu-ref)**2/(2*std**2)).sum(-1).mean()
            actor_loss=-torch.minimum(ratio*adv,clipped*adv).mean()+.01*anchor
            actor_optimizer.zero_grad();actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);actor_optimizer.step()
            critic_loss=((value(obs).squeeze(-1)-batch['returns'][ids])**2).mean()
            critic_optimizer.zero_grad();critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(),.5);critic_optimizer.step()
        logs.append(dict(epoch=epoch,kl_before=kl,actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach())))
    with torch.no_grad():
        logratio=Normal(policy(batch['obs']),std).log_prob(batch['raw']).sum(-1)-batch['old_log']
        finalkl=((logratio.exp()-1)-logratio).mean().item()
    return dict(epochs=logs,final_kl=finalkl)


def sanity_checks():
    # 终止时忽略下个状态价值；截断时保留bootstrap，且不跨回合串联。
    a,_=gae(torch.tensor([1.,2.]),torch.tensor([.5,.2]),torch.tensor([9.,3.]),
            torch.tensor([1.,0.]),torch.tensor([1.,1.]),.9,.95)
    assert torch.allclose(a,torch.tensor([.5,4.5]))
    a,_=gae(torch.tensor([1.,2.]),torch.zeros(2),torch.zeros(2),
            torch.tensor([0.,1.]),torch.tensor([0.,1.]),1.,1.)
    assert torch.allclose(a,torch.tensor([3.,2.]))
    raw=torch.tensor([[.1,.2,.3,.4,.5,.6]])
    dist=Normal(torch.zeros_like(raw),.02)
    assert torch.equal((dist.log_prob(raw).sum(-1)-dist.log_prob(raw).sum(-1)).exp(),torch.ones(1))


def selection_score(evaluation, metric):
    """统一以越大越好计分；到点用负耗时，未全部成功则不具备候选资格。"""
    if evaluation['successes'] != evaluation['cases']:
        return -math.inf
    if metric == 'arrival':
        arrival = evaluation['mean_arrival_time']
        return -arrival if arrival is not None and math.isfinite(arrival) else -math.inf
    return evaluation['mean_reward']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=4)
    parser.add_argument('--steps',type=int,default=8192)
    parser.add_argument('--std',type=float,default=.02)
    parser.add_argument('--time-weight',type=float,default=.05,help='耗时惩罚，每仿真秒扣分')
    parser.add_argument('--selection',choices=('reward','arrival'),default='reward',help='通过全部测试后按奖励或到点停稳时间选择')
    parser.add_argument('--init',type=Path,default=RL_DIR/'results/planar_policy.pt')
    args=parser.parse_args()
    if args.iterations<=0 or args.steps<2 or args.std<=0:parser.error('Invalid training limits')
    if not math.isfinite(args.time_weight) or args.time_weight<0:parser.error('time-weight must be finite and nonnegative')
    REWARD['time']=args.time_weight
    torch.set_num_threads(1);torch.manual_seed(52);sanity_checks()
    checkpoint=torch.load(args.init,map_location='cpu',weights_only=True)
    validate_policy_config(checkpoint['config'])
    if checkpoint['config']['policy_activation']!='silu':raise ValueError('PPO entry requires SiLU policy')
    policy=PolicyNetwork(24,activation='silu');policy.load_state_dict(checkpoint['policy_state_dict'])
    reference=copy.deepcopy(policy).eval()
    for p in reference.parameters():p.requires_grad_(False)
    value=ValueNetwork(24)
    env=PlanarEnv()
    # 选择物理时间尺度：折扣时间常数10秒，GAE trace时间常数0.5秒。
    gamma=math.exp(-env.dt/10.);lam=math.exp(-env.dt/.5)
    hyper=dict(gamma=gamma,gae_lambda=lam,std=args.std,actor_lr=1e-5,critic_lr=.0003,
               clip=.1,epochs=4,batch_size=512,reference_kl_weight=.01,seed=52,dt=env.dt,
               steps=args.steps,iterations=args.iterations,reward=REWARD,max_seconds=40.,selection=args.selection)
    output=RL_DIR/'experiments'/('ppo_silu_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(config=CONFIG,source=str(args.init),source_sha256=hashlib.sha256(args.init.read_bytes()).hexdigest(),hyper=hyper,updates=[])
    print('OUTPUT',output,flush=True)
    report['lqr_baseline']=evaluate(None,teacher=True)
    report['imitation_baseline']=evaluate(policy)
    print('LQR',json.dumps({k:v for k,v in report['lqr_baseline'].items() if k!='results'}),flush=True)
    print('IMITATION',json.dumps({k:v for k,v in report['imitation_baseline'].items() if k!='results'}),flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['critic_warmup']=warmup_critic(policy,value,args.std,gamma)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CRITIC_WARMUP',json.dumps({k:v for k,v in report['critic_warmup'].items() if k!='episodes'}),flush=True)
    actor_optimizer=torch.optim.Adam(policy.parameters(),lr=hyper['actor_lr'])
    critic_optimizer=torch.optim.Adam(value.parameters(),lr=hyper['critic_lr'])
    case_index=0;obs=env.reset(TRAIN_CASES[case_index]);best_score=selection_score(report['imitation_baseline'],args.selection)
    best_state=copy.deepcopy(policy.state_dict());best_iteration=0
    for iteration in range(1,args.iterations+1):
        batch,obs,case_index,episodes=collect(policy,value,env,obs,args.steps,args.std,gamma,lam,case_index)
        previous=copy.deepcopy(policy.state_dict())
        stats=update(policy,value,reference,actor_optimizer,critic_optimizer,batch,args.std)
        if not all(torch.isfinite(t).all() for t in policy.parameters()):raise RuntimeError('Nonfinite policy')
        evaluation=evaluate(policy)
        safe=evaluation['successes']==evaluation['cases']
        record=dict(iteration=iteration,training_steps=iteration*args.steps,ppo=stats,episodes=episodes,evaluation=evaluation,passed_success_gate=safe)
        report['updates'].append(record)
        torch.save(dict(policy_state_dict=policy.state_dict(),value_state_dict=value.state_dict(),config=CONFIG,
                        algorithm='PPO-Clip with fixed std and reference KL penalty',hyper=hyper,
                        actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
                        iteration=iteration,torch_rng_state=torch.get_rng_state(),source=str(args.init)),output/f'candidate_{iteration}.pt')
        score=selection_score(evaluation,args.selection)
        if safe and score>best_score:
            best_score=score;best_state=copy.deepcopy(policy.state_dict());best_iteration=iteration
        print('UPDATE',json.dumps(dict(iteration=iteration,kl=stats['final_kl'],safe=safe,**{k:v for k,v in evaluation.items() if k!='results'})),flush=True)
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        if not safe:
            policy.load_state_dict(previous)
            report['stopped_reason']='Success rate regressed; no further policy updates.'
            break
    policy.load_state_dict(best_state)
    report['selected_iteration']=best_iteration
    report['holdout_reference']=evaluate(reference,HOLDOUT_CASES)
    report['holdout_selected']=evaluate(policy,HOLDOUT_CASES)
    report['deployable_improvement']=best_iteration>0 and report['holdout_selected']['successes']==len(HOLDOUT_CASES) and selection_score(report['holdout_selected'],args.selection)>selection_score(report['holdout_reference'],args.selection)
    # 独立测试未通过时，最终导出恢复为原模仿策略；候选权重仍单独保留。
    report['exported_iteration']=best_iteration if report['deployable_improvement'] else 0
    if not report['deployable_improvement']:
        policy.load_state_dict(reference.state_dict())
    torch.save(dict(policy_state_dict=policy.state_dict(),config=CONFIG,selected_iteration=best_iteration,
                    exported_iteration=report['exported_iteration'],
                    deployable_improvement=report['deployable_improvement'],source=str(args.init)),output/'selected_policy.pt')
    assert hashlib.sha256(args.init.read_bytes()).hexdigest()==report['source_sha256']
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('DONE',json.dumps(dict(output=str(output),selected_iteration=best_iteration,deployable_improvement=report['deployable_improvement'])),flush=True)
