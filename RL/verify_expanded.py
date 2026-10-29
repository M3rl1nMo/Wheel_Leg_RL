"""按保留基线任务能力的约束，优先选择验证停稳数，再作固定测试。"""
import json,time,torch,numpy as np
import direct_clock_ppo as m
from direct_expanded_imitation import ExpandedPolicy

def main():
    torch.set_num_threads(1)
    source=m.RL_DIR/'experiments/expanded_imitation_20261029_190543'
    training=json.loads((source/'report.json').read_text(encoding='utf-8'))
    eligible=[h for h in training['history'] if m.p.preserves_baseline(h['evaluation'],training['baseline_validation'])]
    chosen=max(eligible,key=lambda h:(h['evaluation']['successes'],h['evaluation']['standing'],-h['evaluation']['mean_final_distance']))
    out=m.RL_DIR/'experiments'/('expanded_verification_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(source=str(source),selected_epoch=chosen['epoch'],selection='preserve each baseline validation standing/settled task, then maximize successes, standing, negative distance',validation=chosen['evaluation'])
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # 在任何新测试之前写入选择规则与冻结候选。
    save();print('SELECTED',chosen['epoch'],out,flush=True)
    policy=ExpandedPolicy();policy.load_state_dict(torch.load(source/f"candidate_{chosen['epoch']}.pt",weights_only=True)['policy_state_dict'])
    baseline=m.PolicyNetwork(27);baseline.load_state_dict(torch.load(m.RL_DIR/'results/direct_imitation_candidate.pt',weights_only=True)['policy_state_dict'])
    report['baseline_test']=m.p.evaluate(baseline,m.d.TEST_CASES);report['candidate_test']=m.p.evaluate(policy,m.d.TEST_CASES);save()
    print('TEST',report['baseline_test']['standing'],report['baseline_test']['successes'],report['candidate_test']['standing'],report['candidate_test']['successes'],flush=True)
    rng=np.random.default_rng(1501);cases=[]
    for _ in range(8):
        radius=rng.uniform(.15,.35);angle=rng.uniform(-np.pi,np.pi);yaw=rng.uniform(-np.pi,np.pi)
        cases.append(((float(radius*np.cos(angle)),float(radius*np.sin(angle))),float(yaw)))
    report['fresh_cases']=cases;save()
    report['baseline_fresh']=m.p.evaluate(baseline,cases);report['candidate_fresh']=m.p.evaluate(policy,cases)
    report['accepted_known']=m.p.preserves_baseline(report['candidate_test'],report['baseline_test']) and report['candidate_test']['successes']>report['baseline_test']['successes']
    report['fresh_preserves_baseline']=m.p.preserves_baseline(report['candidate_fresh'],report['baseline_fresh'])
    torch.save(dict(method='frozen_base_neural_teacher_correction',base_config=m.d.CONFIG,policy_state_dict=policy.state_dict(),
        selected_epoch=chosen['epoch'],accepted_known=report['accepted_known'],fresh_preserves_baseline=report['fresh_preserves_baseline']),out/'candidate.pt')
    save();print('DONE',out,report['accepted_known'],report['fresh_preserves_baseline'],report['candidate_fresh']['standing'],report['candidate_fresh']['successes'],flush=True)

if __name__=='__main__':main()
