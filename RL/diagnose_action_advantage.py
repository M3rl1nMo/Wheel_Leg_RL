"""固定同一物理状态，只改变第一个动作，配对随机续跑检查优势排序。"""
import json,time,hashlib
import numpy as np
import torch
import direct_clock_ppo as m

def main():
    torch.set_num_threads(1)
    root=m.RL_DIR;source=root/'results/direct_imitation_candidate.pt'
    policy=m.PolicyNetwork(27);policy.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    value=m.ValueNetwork(28);saved=torch.load(root/'results/direct_clock_critic_candidate.pt',weights_only=True)
    assert saved['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    value.load_state_dict(saved['value_state_dict'])
    bank=torch.load(root/'results/direct_state_bank.pt',weights_only=True)['bank']['test']
    states=[next(s for s in bank if s['category']==category) for category in ('spin','slow')]
    actions=[('baseline',torch.zeros(6))]
    for scale in (.01,.1):
        for sign in (-1,1):
            for pattern in ('same','opposite'):
                delta=torch.zeros(6);delta[4]=scale*sign;delta[5]=scale*sign*(1 if pattern=='same' else -1)
                actions.append((f'{pattern}_{scale}_{sign}',delta))
    out=root/'experiments'/('action_advantage_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(horizon_seconds=30,seeds=[201,202,203],perturbation='first raw action only; then original stochastic policy std=.01',
        policy_sha256=saved['source_sha256'],results=[])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    with torch.no_grad():
        for snapshot in states:
            rows=[]
            for name,delta in actions:
                runs=[]
                for seed in report['seeds']:
                    env=m.c.SnapshotEnv(max_seconds=30);obs=env.restore(snapshot)
                    initial=value(m.critic_obs(env,obs)[None]).item();discount=1.;total=0.;first_td=None
                    torch.manual_seed(seed)
                    for step in range(env.max_steps):
                        mean=policy(obs[None])[0]
                        raw=mean+delta if step==0 else mean+torch.randn(6)*.01
                        obs,reward,term,trunc,result=env.step(raw.tanh()*m.LIMITS)
                        total+=discount*reward;discount*=env.gamma
                        if step==0:first_td=reward+env.gamma*(0 if term else value(m.critic_obs(env,obs)[None]).item())-initial
                        if term or trunc:break
                    runs.append(dict(seed=seed,return30=total,td=first_td,failed=result['failed'],settled=result['settled'],
                        distance=result['final_distance'],yaw=result['last2_mean_abs_yawrate']))
                rows.append(dict(action=name,delta=delta.tolist(),runs=runs,mean_return=float(np.mean([r['return30'] for r in runs])),td=runs[0]['td']))
                report['current']=dict(category=snapshot['category'],rows=rows);save()
                print('ACTION',snapshot['category'],name,rows[-1]['mean_return'],rows[-1]['td'],flush=True)
            baseline=np.array([r['return30'] for r in rows[0]['runs']])
            for row in rows:
                diffs=np.array([r['return30'] for r in row['runs']])-baseline
                row['paired_improvement']=dict(mean=float(diffs.mean()),std=float(diffs.std()),values=diffs.tolist())
                row['td_improvement']=row['td']-rows[0]['td']
            report['results'].append(dict(category=snapshot['category'],origin=snapshot['origin'],rows=rows,
                correlation=float(np.corrcoef([r['td'] for r in rows],[r['mean_return'] for r in rows])[0,1])))
            save()
    report.pop('current',None)
    gamma=np.exp(-.002/10);report['gamma_check']=dict(current=gamma,requested=gamma+.005,
        weight_after_1s=float((gamma+.005)**500),weight_after_30s=float((gamma+.005)**15000))
    save();print('DONE',out,flush=True)

if __name__=='__main__':main()
