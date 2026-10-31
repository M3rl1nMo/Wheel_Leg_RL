
# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import json,time,torch
import RL.archive.code.direct_clock_ppo as m
torch.set_num_threads(1)
p=m.PolicyNetwork(27);p.load_state_dict(torch.load(m.RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt',weights_only=True)['policy_state_dict'])
out=m.RL_DIR/'archive/data/experiments'/('action_period_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
report={}
with torch.no_grad():
    for repeat in (5,10):
        rows=[]
        for case in m.d.TEST_CASES:
            env=m.p.DirectEnv(stop_on_success=False);obs=env.reset(case)
            while True:
                tau=p(obs[None])[0].tanh()*m.LIMITS
                for _ in range(repeat):
                    obs,_,term,trunc,r=env.step(tau)
                    if term or trunc:break
                if term or trunc:break
            rows.append(r)
        report[str(repeat)]=dict(standing=sum(r['standing'] for r in rows),successes=sum(r['settled'] and r['standing'] for r in rows),results=rows)
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(repeat,report[str(repeat)]['standing'],report[str(repeat)]['successes'],out,flush=True)
