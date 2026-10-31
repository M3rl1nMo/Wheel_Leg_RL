"""已有教师数据的网络容量对照；这是模仿学习，不是PPO。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import copy,json,time,hashlib
import torch
from torch import nn
import RL.archive.code.direct_clock_ppo as m

class WidePolicy(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(27,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,6))
    def forward(self,x):return self.network(x)

def main():
    torch.set_num_threads(1);torch.manual_seed(1001)
    source=m.RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    old=m.PolicyNetwork(27);old.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    data=torch.load(m.RL_DIR/'v3_direct/data/experiments/direct_imitation_20261029_130649/dataset.pt',weights_only=True)
    x,y=data['obs'],data['lqr_torques']/m.LIMITS;vx,vy=data['validation_obs'],data['validation_lqr_torques']/m.LIMITS
    out=m.RL_DIR/'archive/data/experiments'/('capacity_imitation_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    policy=WidePolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    with torch.no_grad():baseline_loss=((old(vx).tanh()-vy)**2).mean().item()
    report=dict(method='imitation_existing_data_only',architecture=[27,128,128,6],source_sha256=digest,baseline_validation_mse=baseline_loss,history=[])
    report['baseline_validation']=m.p.evaluate(old,m.d.VAL_CASES);report['baseline_test']=m.p.evaluate(old,m.d.TEST_CASES)
    best=None;score=m.p.score(report['baseline_validation']);chosen=0
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,'BASELINE_MSE',baseline_loss,flush=True)
    for epoch in range(1,201):
        for ids in torch.randperm(len(x)).split(2048):
            loss=((policy(x[ids]).tanh()-y[ids])**2).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch%25==0:
            with torch.no_grad():vl=((policy(vx).tanh()-vy)**2).mean().item()
            ev=m.p.evaluate(policy,m.d.VAL_CASES)
            report['history'].append(dict(epoch=epoch,validation_mse=vl,evaluation=ev))
            if m.p.preserves_baseline(ev,report['baseline_validation']) and m.p.score(ev)>score:
                score=m.p.score(ev);best=copy.deepcopy(policy.state_dict());chosen=epoch
            torch.save(dict(architecture=report['architecture'],config=m.d.CONFIG,policy_state_dict=policy.state_dict(),method=report['method']),out/f'candidate_{epoch}.pt')
            save();print('EPOCH',epoch,vl,ev['standing'],ev['successes'],flush=True)
    report['selected_epoch']=chosen
    if best is not None:
        policy.load_state_dict(best);report['selected_test']=m.p.evaluate(policy,m.d.TEST_CASES)
        report['accepted']=m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
        torch.save(dict(architecture=report['architecture'],config=m.d.CONFIG,policy_state_dict=policy.state_dict(),accepted=report['accepted'],method=report['method']),out/'best_candidate.pt')
    else:report['accepted']=False
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],flush=True)

if __name__=='__main__':main()
