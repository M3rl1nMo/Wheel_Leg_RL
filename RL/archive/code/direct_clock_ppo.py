"""独立实验：27维策略、28维含停稳计时的价值网络、真实起点混合采样。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import copy, hashlib, json, math, time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import RL.v3_direct.direct_navigation as d
import RL.v3_direct.direct_ppo as p
import RL.archive.code.direct_value_calibration as c
from RL.common.networks import PolicyNetwork, ValueNetwork
from RL.common.lqr_imitation import RL_DIR, LIMITS


def critic_obs(env, obs):
    # 策略不使用这个计时；价值网络需要它判断距离成功终止还有多久。
    return torch.cat((obs, torch.tensor([min(env.streak * env.dt / 2., 1.)])))


class ActorView(nn.Module):
    """复用PPO更新函数时，只给策略传前27项，价值网络读取全部28项。"""
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, obs):
        return self.policy(obs[..., :27])


def rollout(policy, snapshot, seed):
    torch.manual_seed(seed)
    env = c.SnapshotEnv(max_seconds=80); obs = env.restore(snapshot)
    xs, indices, rewards = [], [], []
    with torch.no_grad():
        for step in range(env.max_steps):
            if step % 10 == 0 and step * env.dt < 20:
                xs.append(critic_obs(env, obs)); indices.append(step)
            raw = Normal(policy(obs[None])[0], .01).sample()
            obs, reward, term, trunc, result = env.step(raw.tanh() * LIMITS)
            rewards.append(reward)
            if term or trunc: break
    returns = np.zeros(len(rewards)); tail = 0.
    for i in reversed(range(len(rewards))):
        tail = rewards[i] + env.gamma * tail; returns[i] = tail
    return torch.stack(xs), torch.tensor(returns[indices], dtype=torch.float32), dict(
        origin=snapshot['origin'], category=snapshot['category'], seed=seed,
        samples=len(xs), initial_return=float(returns[0]), **result)


def fit(value, datasets, summaries, epochs=200):
    x, y = datasets['train']; vx, vy = datasets['validation']
    weights = torch.cat([torch.full((r['samples'],), len(x)/(len(summaries['train'])*r['samples'])) for r in summaries['train']])
    optimizer = torch.optim.Adam(value.parameters(), lr=.0003)
    best = copy.deepcopy(value.state_dict()); bestloss = c.macro_metrics(value, vx, vy, summaries['validation'])['mse']
    selected = 0; history = []
    torch.manual_seed(96)
    for epoch in range(1, epochs+1):
        for ids in torch.randperm(len(x)).split(256):
            loss = ((value(x[ids]).squeeze(-1)-y[ids]).square()*weights[ids]).mean()
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(value.parameters(), 1.); optimizer.step()
        if epoch % 10 == 0:
            score = c.macro_metrics(value, vx, vy, summaries['validation'])['mse']
            history.append(dict(epoch=epoch, validation_mse=score))
            if score < bestloss: bestloss=score; best=copy.deepcopy(value.state_dict()); selected=epoch
    value.load_state_dict(best)
    return dict(selected_epoch=selected, history=history,
        metrics={s:c.macro_metrics(value,*xy,summaries[s]) for s,xy in datasets.items()})


def collect(policy, value, states, count, seed):
    # 每批从所有训练起点等量采样，避免一个80秒回合占满整批。
    torch.manual_seed(seed); rows=[]; episodes=[]; diagnostics=[]
    for index, state in enumerate(states):
        env=c.SnapshotEnv(max_seconds=40); obs=env.restore(state)
        n=count//len(states)+(index<count%len(states)); local=[]; near=spin=0
        with torch.no_grad():
            for _ in range(n):
                vo=critic_obs(env,obs); dist=Normal(policy(obs[None])[0],.01); raw=dist.sample()
                v=value(vo[None]).squeeze(); log=dist.log_prob(raw).sum()
                no,r,term,trunc,result=env.step(raw.tanh()*LIMITS)
                nv=torch.tensor(0.) if term else value(critic_obs(env,no)[None]).squeeze()
                local.append((vo,raw,log,v,nv,r,float(term),float(term or trunc)))
                near+=result['final_distance']<.08
                spin+=result['final_distance']<.08 and result['final_yawrate']>.1
                obs=no
                if term or trunc:
                    episodes.append(dict(category=state['category'],**result));obs=env.restore(state)
        cols=list(zip(*local)); ts=[torch.stack(list(col)) for col in cols[:5]]
        rewards,terms,bounds=[torch.tensor(col) for col in cols[5:]]
        # 每个起点片段单独计算GAE，片段末尾保留下一状态价值，不串接其他起点。
        adv,ret=p.gae(rewards,ts[3],ts[4],terms,bounds,env.gamma,math.exp(-env.dt/2.))
        rows.append((*ts,adv,ret));diagnostics.append(dict(category=state['category'],steps=n,near_steps=near,spin_steps=spin,
            reward_mean=float(rewards.mean()),advantage_mean=float(adv.mean()),advantage_std=float(adv.std(unbiased=False))))
    batch={key:torch.cat([row[i] for row in rows]) for i,key in enumerate(('obs','raw','old_log','value','next_value','advantage','returns'))}
    adv=batch['advantage'];batch['advantage']=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    return batch,episodes,diagnostics


def main():
    torch.set_num_threads(1);torch.manual_seed(101)
    source=RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt'; source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    actor_saved=torch.load(source,weights_only=True)
    bank_saved=torch.load(RL_DIR/'archive/data/results/direct_state_bank.pt',weights_only=True)
    critic_saved=torch.load(RL_DIR/'archive/data/results/direct_value_calibrated_candidate.pt',weights_only=True)
    assert actor_saved['config']==bank_saved['config']==critic_saved['config']==d.CONFIG
    assert bank_saved['actor_sha256']==critic_saved['policy_sha256']==source_hash and critic_saved['action_std']==.01
    policy=PolicyNetwork(27);policy.load_state_dict(actor_saved['policy_state_dict'])
    reference=copy.deepcopy(policy).eval()
    for parameter in reference.parameters():parameter.requires_grad_(False)
    prior=ValueNetwork(27);prior.load_state_dict(critic_saved['value_state_dict'])
    clock=ValueNetwork(28);state=copy.deepcopy(prior.state_dict())
    state['network.0.weight']=torch.cat((state['network.0.weight'],torch.zeros(32,1)),dim=1)
    clock.load_state_dict(state)
    out=RL_DIR/'archive/data/experiments'/('direct_clock_ppo_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(source_sha256=source_hash,actor_input_dim=27,critic_input_dim=28,clock='min(streak*dt/2,1)',
        reward=p.REWARD.copy(),std=.01,gae_seconds=2.,actor_lr=1e-5,critic_lr=.0003,steps=32768,iterations=4,rollouts={},updates=[])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True);datasets={}
    for split,states in bank_saved['bank'].items():
        xs=[];ys=[];summaries=[]
        for i,snapshot in enumerate(states):
            for repeat in range(3 if split=='test' else 1):
                seed={'train':93000,'validation':94000,'test':95000}[split]+i*10+repeat
                x,y,summary=rollout(reference,snapshot,seed);xs.append(x);ys.append(y);summaries.append(summary)
            print('DATA',split,i,snapshot['category'],flush=True)
        datasets[split]=(torch.cat(xs),torch.cat(ys));report['rollouts'][split]=summaries;save()
    # 同样数据、种子、训练预算，27维与28维作对照，隔离新增计时输入的作用。
    control=copy.deepcopy(prior)
    control_data={s:(x[:,:27],y) for s,(x,y) in datasets.items()}
    report['control27']=fit(control,control_data,report['rollouts'])
    report['clock28']=fit(clock,datasets,report['rollouts']);save()
    torch.save(dict(datasets=datasets,config=d.CONFIG,critic_extra='settle_fraction'),out/'value_dataset.pt')
    torch.save(dict(value_state_dict=clock.state_dict(),actor_config=d.CONFIG,critic_dim=28,source_sha256=source_hash),out/'calibrated_critic.pt')
    report['baseline_validation']=p.evaluate(policy,d.VAL_CASES);report['baseline_test']=p.evaluate(policy,d.TEST_CASES);save()
    print('CALIBRATION',json.dumps({k:{s:v['mse'] for s,v in report[k]['metrics'].items()} for k in ('control27','clock28')}),flush=True)
    # 若计时输入未改善验证误差，不开展策略更新；保存证据供下一步分析。
    report['clock_gate']=report['clock28']['metrics']['validation']['mse']<report['control27']['metrics']['validation']['mse']
    if not report['clock_gate']:
        save();print('STOP clock validation did not improve',flush=True);return
    actor=ActorView(policy);ref=ActorView(reference)
    ao=torch.optim.Adam(actor.parameters(),lr=1e-5);co=torch.optim.Adam(clock.parameters(),lr=.0003)
    best=copy.deepcopy(policy.state_dict());bestscore=p.score(report['baseline_validation']);selected=0
    for iteration in range(1,5):
        batch,episodes,diagnostics=collect(policy,clock,bank_saved['bank']['train'],32768,110+iteration)
        stats=p.update(actor,clock,ref,ao,co,batch,.01)
        assert all(torch.isfinite(par).all() for par in policy.parameters())
        evaluation=p.evaluate(policy,d.VAL_CASES);safe=p.preserves_baseline(evaluation,report['baseline_validation'])
        if safe and p.score(evaluation)>bestscore:
            bestscore=p.score(evaluation);best=copy.deepcopy(policy.state_dict());selected=iteration
        report['updates'].append(dict(iteration=iteration,evaluation=evaluation,ppo=stats,episodes=episodes,sampling=diagnostics))
        torch.save(dict(config=d.CONFIG,policy_state_dict=policy.state_dict(),value_state_dict=clock.state_dict(),critic_dim=28,
            actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),iteration=iteration),out/f'candidate_{iteration}.pt')
        save();print('UPDATE',iteration,json.dumps({k:v for k,v in evaluation.items() if k!='results'}),flush=True)
    policy.load_state_dict(best);report['selected_iteration']=selected;report['selected_test']=p.evaluate(policy,d.TEST_CASES)
    accepted=selected>0 and p.preserves_baseline(report['selected_test'],report['baseline_test']) and p.score(report['selected_test'])>p.score(report['baseline_test'])
    report['accepted']=accepted;report['exported_iteration']=selected if accepted else 0
    if not accepted:policy.load_state_dict(reference.state_dict())
    torch.save(dict(config=d.CONFIG,policy_state_dict=policy.state_dict(),exported_iteration=report['exported_iteration']),out/'selected_policy.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    save();print('DONE',str(out),accepted,flush=True)


if __name__=='__main__':main()
