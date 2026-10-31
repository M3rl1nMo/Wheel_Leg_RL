"""仿真回报选择输入连接权重倍率，部署仍是原27维神经网络。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import copy,json,time,hashlib,itertools
import torch
import RL.archive.code.direct_clock_ppo as m

def main():
    torch.set_num_threads(1)
    source=m.RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    saved=torch.load(source,weights_only=True);original=saved['policy_state_dict'];policy=m.PolicyNetwork(27);policy.load_state_dict(original)
    cases=m.c.CASES['train']
    out=m.RL_DIR/'archive/data/experiments'/('sensitivity_search_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(method='reward_search_goal_and_yaw_input_connections',source_sha256=digest,trials=[],validation=[])
    report['baseline_train']=m.p.evaluate(policy,cases);report['baseline_validation']=m.p.evaluate(policy,m.d.VAL_CASES);report['baseline_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    def weights(goal,yaw):
        state=copy.deepcopy(original);state['network.0.weight'][:,19:21]*=goal;state['network.0.weight'][:,4]*=yaw
        return state
    for goal,yaw in itertools.product((1.,1.5,2.,3.),(1.,2.,4.,8.)):
        policy.load_state_dict(weights(goal,yaw));ev=m.p.evaluate(policy,cases)
        report['trials'].append(dict(goal_scale=goal,yaw_scale=yaw,evaluation=ev))
        save();print('TRIAL',goal,yaw,ev['standing'],ev['successes'],flush=True)
    eligible=[r for r in report['trials'] if m.p.preserves_baseline(r['evaluation'],report['baseline_train'])]
    eligible.sort(key=lambda r:(r['evaluation']['successes'],r['evaluation']['standing'],-r['evaluation']['mean_final_distance']),reverse=True)
    chosen=None;key=(report['baseline_validation']['successes'],report['baseline_validation']['standing'],-report['baseline_validation']['mean_final_distance'])
    for row in eligible[:5]:
        policy.load_state_dict(weights(row['goal_scale'],row['yaw_scale']));ev=m.p.evaluate(policy,m.d.VAL_CASES)
        report['validation'].append(dict(goal_scale=row['goal_scale'],yaw_scale=row['yaw_scale'],evaluation=ev))
        candidate_key=(ev['successes'],ev['standing'],-ev['mean_final_distance'])
        if m.p.preserves_baseline(ev,report['baseline_validation']) and candidate_key>key:chosen=row;key=candidate_key
        save()
    report['chosen']=None if chosen is None else {k:chosen[k] for k in ('goal_scale','yaw_scale')}
    policy.load_state_dict(original if chosen is None else weights(chosen['goal_scale'],chosen['yaw_scale']))
    report['selected_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    report['accepted']=chosen is not None and m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
    torch.save(dict(config=m.d.CONFIG,policy_state_dict=policy.state_dict(),method=report['method'],accepted=report['accepted']),out/'candidate.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],report['selected_test']['standing'],report['selected_test']['successes'],flush=True)

if __name__=='__main__':main()
