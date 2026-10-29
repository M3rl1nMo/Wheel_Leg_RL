"""独立实验：27维策略、28维含停稳计时的价值网络、真实起点混合采样。"""
import copy, hashlib, json, math, time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import direct_navigation as d
import direct_ppo as p
import direct_value_calibration as c
from networks import PolicyNetwork, ValueNetwork
from lqr_imitation import RL_DIR, LIMITS


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


def collect(policy,value,states,count,seed,gae_seconds=2.):
    torch.manual_seed(seed);rows=[];episodes=[];diagnostics=[]
    for snapshot in states:
        env=c.SnapshotEnv(max_seconds=40);obs=env.restore(snapshot);local=[];rewards=[]
        with torch.no_grad():
            while True:
                vo=critic_obs(env,obs);dist=Normal(policy(obs[None])[0],.01);raw=dist.sample()
                local.append((vo,raw,dist.log_prob(raw).sum(),value(vo[None]).squeeze()))
                obs,reward,term,trunc,result=env.step(raw.tanh()*LIMITS);rewards.append(reward)
                if term or trunc:break
        # 有限40秒任务的完整实际回报；截断以后按任务结束处理，不用价值预测填尾。
        returns=torch.empty(len(rewards));tail=0.
        for i in reversed(range(len(rewards))):tail=rewards[i]+env.gamma*tail;returns[i]=tail
        cols=[torch.stack(list(col)) for col in zip(*local)]
        rows.append((*cols,returns-cols[3],returns));episodes.append(dict(category=snapshot['category'],**result))
        diagnostics.append(dict(category=snapshot['category'],steps=len(rewards),return0=float(returns[0])))
    batch={key:torch.cat([row[i] for row in rows]) for i,key in enumerate(('obs','raw','old_log','value','advantage','returns'))}
    a=batch['advantage'];batch['advantage']=(a-a.mean())/(a.std(unbiased=False)+1e-8)
    return batch,episodes,diagnostics

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--gae-seconds',type=float,default=10.)
    parser.add_argument('--steps',type=int,default=131072);parser.add_argument('--iterations',type=int,default=6)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(301)
    source=RL_DIR/'results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    saved=torch.load(source,weights_only=True);policy=PolicyNetwork(27);policy.load_state_dict(saved['policy_state_dict'])
    reference=copy.deepcopy(policy).eval()
    for par in reference.parameters():par.requires_grad_(False)
    value=ValueNetwork(28);cs=torch.load(RL_DIR/'results/direct_clock_critic_candidate.pt',weights_only=True)
    assert cs['source_sha256']==digest;value.load_state_dict(cs['value_state_dict'])
    bank=torch.load(RL_DIR/'results/direct_state_bank.pt',weights_only=True)
    assert bank['actor_sha256']==digest
    out=RL_DIR/'experiments'/('mc_ppo_'+str(args.gae_seconds)+'_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(config=d.CONFIG,critic_dim=28,source_sha256=digest,args=vars(args),method="finite_40s_monte_carlo_ppo",seed=301,updates=[])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    report['baseline_validation']=p.evaluate(policy,d.VAL_CASES);report['baseline_test']=p.evaluate(policy,d.TEST_CASES)
    best=copy.deepcopy(policy.state_dict());bestscore=p.score(report['baseline_validation']);chosen=0
    actor=ActorView(policy);ref=ActorView(reference)
    ao=torch.optim.Adam(actor.parameters(),lr=3e-6);co=torch.optim.Adam(value.parameters(),lr=.0001)
    for iteration in range(1,args.iterations+1):
        batch,episodes,diag=collect(policy,value,bank['bank']['train'],args.steps,301+iteration,args.gae_seconds)
        stats=p.update(actor,value,ref,ao,co,batch,.01,epochs=2)
        evaluation=p.evaluate(policy,d.VAL_CASES)
        if p.preserves_baseline(evaluation,report['baseline_validation']) and p.score(evaluation)>bestscore:
            bestscore=p.score(evaluation);best=copy.deepcopy(policy.state_dict());chosen=iteration
        report['updates'].append(dict(iteration=iteration,evaluation=evaluation,ppo=stats,episodes=episodes,sampling=diag))
        torch.save(dict(config=d.CONFIG,policy_state_dict=policy.state_dict(),value_state_dict=value.state_dict(),critic_dim=28),out/f'candidate_{iteration}.pt')
        save();print('UPDATE',iteration,json.dumps({k:v for k,v in evaluation.items() if k!='results'}),flush=True)
    policy.load_state_dict(best);report['selected_iteration']=chosen;report['selected_test']=p.evaluate(policy,d.TEST_CASES)
    accepted=chosen>0 and p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
    report['accepted']=accepted
    torch.save(dict(config=d.CONFIG,policy_state_dict=policy.state_dict(),selected_iteration=chosen,accepted=accepted),out/'best_candidate.pt')
    if not accepted:policy.load_state_dict(reference.state_dict())
    torch.save(dict(config=d.CONFIG,policy_state_dict=policy.state_dict(),accepted=accepted),out/'selected_policy.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,accepted,flush=True)

if __name__=='__main__':main()
