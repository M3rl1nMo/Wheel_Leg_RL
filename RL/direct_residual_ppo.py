"""冻结500Hz基础策略，50Hz神经网络残差；无需位置外环/LQR。"""
import copy,json,time,hashlib,math
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import direct_clock_ppo as m

REPEAT=10
SCALE=.1
STD=.2

class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(27,32),nn.SiLU(),nn.Linear(32,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,obs):return self.net(obs[...,:27])

def advance(env,base,obs,raw):
    delta=SCALE*raw.tanh();reward=0.;discount=1.
    for _ in range(REPEAT):
        tau=(base(obs[None])[0]+delta).tanh()*m.LIMITS
        obs,r,term,trunc,result=env.step(tau)
        reward+=discount*r;discount*=env.gamma
        if term or trunc:break
    return obs,reward,term,trunc,result,discount

def evaluate(base,actor,cases):
    rows=[]
    with torch.no_grad():
        for case in cases:
            env=m.p.DirectEnv(stop_on_success=False);obs=env.reset(case)
            while True:
                obs,_,term,trunc,result,_=advance(env,base,obs,actor(obs[None])[0])
                if term or trunc:break
            rows.append(dict(target=list(case[0]),initial_yaw=case[1],**result))
    return dict(standing=sum(r['standing'] for r in rows),successes=sum(r['standing'] and r['settled'] for r in rows),
        mean_final_distance=float(np.mean([r['final_distance'] for r in rows])),results=rows)

def collect(base,actor,value,states,seed):
    torch.manual_seed(seed);parts=[];episodes=[]
    with torch.no_grad():
        for state in states:
            env=m.c.SnapshotEnv(max_seconds=40);obs=env.restore(state);rows=[]
            for _ in range(600):
                vo=m.critic_obs(env,obs);dist=Normal(actor(vo[None])[0],STD);raw=dist.sample()
                v=value(vo[None]).squeeze();log=dist.log_prob(raw).sum()
                obs,r,term,trunc,result,discount=advance(env,base,obs,raw)
                nv=torch.tensor(0.) if term else value(m.critic_obs(env,obs)[None]).squeeze()
                rows.append((vo,raw,log,v,nv,r,float(term),float(term or trunc)))
                if term or trunc:
                    episodes.append(dict(category=state['category'],**result));obs=env.restore(state)
            cols=list(zip(*rows));ts=[torch.stack(list(c)) for c in cols[:5]]
            rew,term,bound=[torch.tensor(c) for c in cols[5:]]
            adv,returns=m.p.gae(rew,ts[3],ts[4],term,bound,env.gamma**REPEAT,math.exp(-env.dt*REPEAT/2.))
            parts.append((*ts,adv,returns))
    batch={key:torch.cat([row[i] for row in parts]) for i,key in enumerate(('obs','raw','old_log','value','next_value','advantage','returns'))}
    a=batch['advantage'];batch['advantage']=(a-a.mean())/(a.std(unbiased=False)+1e-8)
    return batch,episodes

def main():
    torch.set_num_threads(1);torch.manual_seed(912)
    source=m.RL_DIR/'results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    base=m.PolicyNetwork(27);base.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    for par in base.parameters():par.requires_grad_(False)
    actor=Residual();reference=copy.deepcopy(actor)
    for par in reference.parameters():par.requires_grad_(False)
    value=m.ValueNetwork(28);cs=torch.load(m.RL_DIR/'results/direct_clock_critic_candidate.pt',weights_only=True)
    assert cs['source_sha256']==digest;value.load_state_dict(cs['value_state_dict'])
    states=torch.load(m.RL_DIR/'results/direct_state_bank.pt',weights_only=True)['bank']['train']
    out=m.RL_DIR/'experiments'/('residual_ppo_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(method='frozen_base_500hz_neural_residual_50hz',base_sha256=digest,residual_scale=SCALE,std=STD,repeat=REPEAT,updates=[])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    report['baseline_validation']=evaluate(base,actor,m.d.VAL_CASES);report['baseline_test']=evaluate(base,actor,m.d.TEST_CASES)
    best=copy.deepcopy(actor.state_dict());bestscore=m.p.score(report['baseline_validation']);chosen=0
    ao=torch.optim.Adam(actor.parameters(),lr=.0001);co=torch.optim.Adam(value.parameters(),lr=.0001)
    for iteration in range(1,9):
        batch,episodes=collect(base,actor,value,states,912+iteration)
        stats=m.p.update(actor,value,reference,ao,co,batch,STD,epochs=4)
        ev=evaluate(base,actor,m.d.VAL_CASES)
        if m.p.preserves_baseline(ev,report['baseline_validation']) and m.p.score(ev)>bestscore:
            best=copy.deepcopy(actor.state_dict());bestscore=m.p.score(ev);chosen=iteration
        report['updates'].append(dict(iteration=iteration,evaluation=ev,ppo=stats,episodes=episodes))
        torch.save(dict(base_config=m.d.CONFIG,base_state_dict=base.state_dict(),residual_state_dict=actor.state_dict(),
            value_state_dict=value.state_dict(),repeat=REPEAT,scale=SCALE,method=report['method']),out/f'candidate_{iteration}.pt')
        save();print('UPDATE',iteration,ev['standing'],ev['successes'],ev['mean_final_distance'],flush=True)
    actor.load_state_dict(best);report['selected_iteration']=chosen;report['selected_test']=evaluate(base,actor,m.d.TEST_CASES)
    report['accepted']=chosen>0 and m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
    torch.save(dict(base_config=m.d.CONFIG,base_state_dict=base.state_dict(),residual_state_dict=actor.state_dict(),repeat=REPEAT,scale=SCALE,
        accepted=report['accepted'],method=report['method']),out/'best_candidate.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],flush=True)

if __name__=='__main__':main()
