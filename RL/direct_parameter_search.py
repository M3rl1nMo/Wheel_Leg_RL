"""基于仿真回报的参数搜索：仅调整策略最后一层偏置，保持500Hz反馈。"""
import copy,json,time,hashlib
import numpy as np
import torch
import direct_clock_ppo as m

def evaluate_starts(policy,states):
    rows=[]
    with torch.no_grad():
        for state in states:
            env=m.c.SnapshotEnv(stop_on_success=False,max_seconds=12);obs=env.restore(state)
            while True:
                obs,_,term,trunc,result=env.step(policy(obs[None])[0].tanh()*m.LIMITS)
                if term or trunc:break
            rows.append(result)
    return dict(standing=sum(r['standing'] for r in rows),successes=sum(r['standing'] and r['settled'] for r in rows),
        reward=float(np.mean([r['return_sum'] for r in rows])),results=rows)

def rank(result):return(result['standing'],result['successes'],result['reward'])

def main():
    torch.set_num_threads(1);torch.manual_seed(811)
    source=m.RL_DIR/'results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    policy=m.PolicyNetwork(27);policy.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    original=copy.deepcopy(policy.state_dict());bias=original['network.2.bias'].clone()
    bank=torch.load(m.RL_DIR/'results/direct_state_bank.pt',weights_only=True)['bank']['train']
    # 仅训练组真实状态，兼顾持续转圈、距离残差、保持与普通出发。
    states=[next(s for s in bank if s['category']==k) for k in ('spin','slow','ready','normal')]
    out=m.RL_DIR/'experiments'/('parameter_search_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(source_sha256=digest,method='reward_based_last_layer_bias_search',seed=811,training_states=[{k:s[k] for k in ('category','origin')} for s in states],rounds=[])
    report['baseline_validation']=m.p.evaluate(policy,m.d.VAL_CASES);report['baseline_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    current=evaluate_starts(policy,states);report['baseline_training']=current
    best=copy.deepcopy(original);bestscore=m.p.score(report['baseline_validation']);chosen=0
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    for iteration,scale in enumerate([.001,.003,.003,.001,.0005,.0005],1):
        candidates=[];winner=bias.clone();winner_score=rank(current)
        directions=torch.randn(4,6);directions/=directions.norm(dim=1,keepdim=True)
        for direction in directions:
            for sign in (-1,1):
                trial=bias+sign*scale*direction
                with torch.no_grad():policy.network[2].bias.copy_(trial)
                result=evaluate_starts(policy,states);candidates.append(dict(bias=trial.tolist(),evaluation=result))
                if rank(result)>winner_score and m.p.preserves_baseline(result,report['baseline_training']):
                    winner_score=rank(result);winner=trial.clone();current=result
        bias=winner
        with torch.no_grad():policy.network[2].bias.copy_(bias)
        validation=m.p.evaluate(policy,m.d.VAL_CASES)
        if m.p.preserves_baseline(validation,report['baseline_validation']) and m.p.score(validation)>bestscore:
            best=copy.deepcopy(policy.state_dict());bestscore=m.p.score(validation);chosen=iteration
        report['rounds'].append(dict(iteration=iteration,scale=scale,candidates=candidates,training=current,validation=validation))
        torch.save(dict(config=m.d.CONFIG,policy_state_dict=policy.state_dict(),method=report['method']),out/f'candidate_{iteration}.pt')
        save();print('ROUND',iteration,current['successes'],validation['standing'],validation['successes'],flush=True)
    policy.load_state_dict(best);report['selected_round']=chosen;report['selected_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    report['accepted']=chosen>0 and m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
    torch.save(dict(config=m.d.CONFIG,policy_state_dict=policy.state_dict(),accepted=report['accepted'],method=report['method']),out/'best_candidate.pt')
    if not report['accepted']:policy.load_state_dict(original)
    torch.save(dict(config=m.d.CONFIG,policy_state_dict=policy.state_dict(),accepted=report['accepted']),out/'selected_policy.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],flush=True)

if __name__=='__main__':main()
