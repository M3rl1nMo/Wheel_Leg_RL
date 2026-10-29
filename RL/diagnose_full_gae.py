"""配对随机续跑：实际回报、单步TD、训练片段GAE与长轨迹GAE。"""
import json,time,hashlib,math
import numpy as np
import torch
import direct_clock_ppo as m

def summary(values):
    x=np.asarray(values);rng=np.random.default_rng(551)
    boot=rng.choice(x,(10000,len(x)),replace=True).mean(axis=1)
    return dict(mean=float(x.mean()),std=float(x.std()),positive=int((x>0).sum()),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(),values=x.tolist())

def main():
    torch.set_num_threads(1);root=m.RL_DIR
    source=root/'results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    policy=m.PolicyNetwork(27);policy.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    value=m.ValueNetwork(28);checkpoint=torch.load(root/'results/direct_clock_critic_candidate.pt',weights_only=True)
    assert checkpoint['source_sha256']==digest
    value.load_state_dict(checkpoint['value_state_dict'])
    bank=torch.load(root/'results/direct_state_bank.pt',weights_only=True)['bank']['test']
    actions=[('baseline',0.,0.),('same_small',.01,.01),('opposite_small',.01,-.01),
        ('same_large',.1,.1),('opposite_large',-.1,.1)]
    out=root/'experiments'/('full_gae_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(seeds=list(range(201,211)),horizon_seconds=30,fragment_steps=2340,gae_lambda_seconds=2.,
        actor_sha256=digest,critic_sha256=hashlib.sha256((root/'results/direct_clock_critic_candidate.pt').read_bytes()).hexdigest(),
        policy_updated=False,results=[])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    with torch.no_grad():
        for category in ('spin','slow'):
            state=next(s for s in bank if s['category']==category);rows=[]
            for name,left,right in actions:
                runs=[]
                for seed in report['seeds']:
                    env=m.c.SnapshotEnv(max_seconds=30);obs=env.restore(state)
                    val=value(m.critic_obs(env,obs)[None]).item();gamma=env.gamma;lam=math.exp(-env.dt/2)
                    discounted=0.;gae=0.;fragment=0.;discount=1.;trace=1.;td0=None
                    torch.manual_seed(seed)
                    for step in range(env.max_steps):
                        mean=policy(obs[None])[0]
                        if step==0:
                            raw=mean.clone();raw[4]+=left;raw[5]+=right
                        else:raw=mean+torch.randn(6)*.01
                        obs,reward,term,trunc,result=env.step(raw.tanh()*m.LIMITS)
                        nv=0. if term else value(m.critic_obs(env,obs)[None]).item()
                        td=reward+gamma*nv-val
                        if td0 is None:td0=td
                        gae+=trace*td
                        if step<2340:fragment+=trace*td
                        discounted+=discount*reward;discount*=gamma;trace*=gamma*lam;val=nv
                        if term or trunc:break
                    runs.append(dict(seed=seed,return30=discounted,td=td0,gae_fragment=fragment,gae_full=gae,
                        duration=result['duration'],failed=result['failed'],settled=result['settled'],distance=result['final_distance'],yaw=result['last2_mean_abs_yawrate']))
                    print('RUN',category,name,seed,flush=True)
                row=dict(action=name,raw_wheel_delta=[left,right],runs=runs)
                base=runs if not rows else rows[0]['runs']
                row['paired']={key:summary([r[key]-b[key] for r,b in zip(runs,base)]) for key in ('return30','td','gae_fragment','gae_full')}
                rows.append(row);report['current']=dict(category=category,rows=rows);save()
                print('ACTION',category,name,json.dumps({k:v['mean'] for k,v in row['paired'].items()}),flush=True)
            report['results'].append(dict(category=category,origin=state['origin'],rows=rows));save()
    report.pop('current',None);assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,flush=True)

if __name__=='__main__':main()
