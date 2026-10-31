"""大网络的在策略状态教师标注；推理无教师/位置外环。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import copy,json,time,hashlib
import torch
import RL.archive.code.direct_clock_ppo as m
from RL.archive.code.direct_capacity_test import WidePolicy

def main():
    torch.set_num_threads(1);torch.manual_seed(1102)
    source=m.RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    old=m.PolicyNetwork(27);old.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    init=m.RL_DIR/'archive/data/experiments/capacity_imitation_20261029_185530/candidate_200.pt'
    policy=WidePolicy();policy.load_state_dict(torch.load(init,weights_only=True)['policy_state_dict'])
    data=torch.load(m.RL_DIR/'v3_direct/data/experiments/direct_imitation_20261029_130649/dataset.pt',weights_only=True)
    x,y=data['obs'],data['lqr_torques']/m.LIMITS
    out=m.RL_DIR/'archive/data/experiments'/('wide_dagger_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(method='wide_imitation_dagger',architecture=[27,128,128,6],source_sha256=digest,initial=str(init),rounds=[])
    report['baseline_validation']=m.p.evaluate(old,m.d.VAL_CASES);report['baseline_test']=m.p.evaluate(old,m.d.TEST_CASES)
    best=None;score=m.p.score(report['baseline_validation']);chosen=0;rx=ry=None
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    for ri,beta in enumerate((.75,.5,.25,0.),1):
        xs=[];ys=[];traces=[]
        for case in m.d.TRAIN_CASES:
            sx,sy,r,_=m.d.rollout(policy,case,seconds=40,collect=True,beta=beta)
            xs.append(sx);ys.append(sy/m.LIMITS);traces.append(r)
        nx,ny=torch.cat(xs),torch.cat(ys)
        rx=nx if rx is None else torch.cat((rx,nx));ry=ny if ry is None else torch.cat((ry,ny))
        optimizer=torch.optim.Adam(policy.parameters(),lr=.0001)
        for epoch in range(60):
            for ids in torch.randperm(len(rx)).split(512):
                oldids=torch.randint(len(x),(len(ids),))
                xx=torch.cat((rx[ids],x[oldids]));yy=torch.cat((ry[ids],y[oldids]))
                loss=(policy(xx).tanh()-yy).square().mean()
                optimizer.zero_grad();loss.backward();optimizer.step()
        ev=m.p.evaluate(policy,m.d.VAL_CASES)
        if m.p.preserves_baseline(ev,report['baseline_validation']) and m.p.score(ev)>score:
            best=copy.deepcopy(policy.state_dict());score=m.p.score(ev);chosen=ri
        report['rounds'].append(dict(round=ri,beta=beta,new_samples=len(nx),total_new_samples=len(rx),training=traces,validation=ev))
        torch.save(dict(config=m.d.CONFIG,architecture=report['architecture'],policy_state_dict=policy.state_dict(),method=report['method']),out/f'candidate_{ri}.pt')
        save();print('ROUND',ri,len(nx),ev['standing'],ev['successes'],ev['mean_final_distance'],flush=True)
    report['selected_round']=chosen
    if best is not None:
        policy.load_state_dict(best);report['selected_test']=m.p.evaluate(policy,m.d.TEST_CASES)
        report['accepted']=m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
        torch.save(dict(config=m.d.CONFIG,architecture=report['architecture'],policy_state_dict=policy.state_dict(),method=report['method'],accepted=report['accepted']),out/'best_candidate.pt')
    else:report['accepted']=False
    torch.save(dict(obs=rx,normalized_torques=ry),out/'new_dataset.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],flush=True)

if __name__=='__main__':main()
