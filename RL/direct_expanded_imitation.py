"""冻结原策略，增加不读取上一动作的教师修正网络；不是强化学习。"""
import copy,json,time,hashlib
import torch
from torch import nn
import direct_clock_ppo as m

class ExpandedPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.base=m.PolicyNetwork(27)
        for p in self.base.parameters():p.requires_grad_(False)
        self.correction=nn.Sequential(nn.Linear(21,96),nn.SiLU(),nn.Linear(96,6))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)
    def forward(self,x):return self.base(x)+self.correction(x[...,:21])

def main():
    torch.set_num_threads(1);torch.manual_seed(1201)
    source=m.RL_DIR/'results/direct_imitation_candidate.pt';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    policy=ExpandedPolicy();policy.base.load_state_dict(torch.load(source,weights_only=True)['policy_state_dict'])
    data=torch.load(m.RL_DIR/'experiments/direct_imitation_20261029_130649/dataset.pt',weights_only=True)
    x,y=data['obs'],data['lqr_torques']/m.LIMITS;vx,vy=data['validation_obs'],data['validation_lqr_torques']/m.LIMITS
    out=m.RL_DIR/'experiments'/('expanded_imitation_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
    report=dict(method='frozen_base_neural_teacher_correction',source_sha256=digest,history=[])
    report['baseline_validation']=m.p.evaluate(policy,m.d.VAL_CASES);report['baseline_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    best=copy.deepcopy(policy.state_dict());score=m.p.score(report['baseline_validation']);chosen=0
    optimizer=torch.optim.Adam(policy.correction.parameters(),lr=.00003)
    save=lambda:(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    for epoch in range(1,101):
        for ids in torch.randperm(len(x)).split(1024):
            loss=(policy(x[ids]).tanh()-y[ids]).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch%10==0:
            with torch.no_grad():vl=(policy(vx).tanh()-vy).square().mean().item()
            ev=m.p.evaluate(policy,m.d.VAL_CASES)
            if m.p.preserves_baseline(ev,report['baseline_validation']) and m.p.score(ev)>score:
                score=m.p.score(ev);best=copy.deepcopy(policy.state_dict());chosen=epoch
            report['history'].append(dict(epoch=epoch,validation_mse=vl,evaluation=ev))
            torch.save(dict(method=report['method'],base_config=m.d.CONFIG,policy_state_dict=policy.state_dict()),out/f'candidate_{epoch}.pt')
            save();print('EPOCH',epoch,vl,ev['standing'],ev['successes'],flush=True)
    policy.load_state_dict(best);report['selected_epoch']=chosen;report['selected_test']=m.p.evaluate(policy,m.d.TEST_CASES)
    report['accepted']=chosen>0 and m.p.preserves_baseline(report['selected_test'],report['baseline_test']) and report['selected_test']['successes']>report['baseline_test']['successes']
    torch.save(dict(method=report['method'],base_config=m.d.CONFIG,policy_state_dict=policy.state_dict(),accepted=report['accepted']),out/'best_candidate.pt')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    save();print('DONE',out,report['accepted'],flush=True)

if __name__=='__main__':main()
