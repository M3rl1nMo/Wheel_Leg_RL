"""固定数据上的 SiLU/ReLU 配对实验：相同初始参数、数据顺序和训练预算。

不是从SiLU权重直接切换激活函数；两个模型都从同一组随机参数开始。
数据由此前SiLU策略的数据聚合实验产生，结论仅限该数据分布与配置。
"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from pathlib import Path
import argparse
import copy
import hashlib
import json
import time

import numpy as np
import torch
from RL.common.networks import PolicyNetwork
from RL.common.lqr_imitation import RL_DIR, LIMITS, fit
from RL.v2_outer.planar_navigation import CONFIG, make_model, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=RL_DIR/'v2_outer/data/experiments/planar_20261029_110853/dataset.pt')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seeds', type=int, nargs='+', default=[46,47,48])
    args=parser.parse_args()
    if args.epochs <= 0:parser.error('--epochs must be positive')
    torch.set_num_threads(1)
    data=torch.load(args.dataset,weights_only=True,map_location='cpu')
    assert data['config']==CONFIG
    x,y,vx,vy=data['obs'],data['lqr_torques'],data['validation_obs'],data['validation_lqr_torques']
    assert x.shape[1]==vx.shape[1]==24 and y.shape[1]==vy.shape[1]==6
    assert all(torch.isfinite(t).all() for t in (x,y,vx,vy))
    output=RL_DIR/'archive/data/experiments'/('activation_comparison_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    fixed=[((.25,0.),0.),((0.,.25),0.),((-.25,0.),0.),((.18,-.18),0.),((.23,.14),1.)]
    extra=[((-.3310953371672254,.08359958753452192),3.00303934229726),
           ((-.17782487679726747,.5075895967697843),-.03714260467785824),
           ((.3319516791010715,.09400023297468756),-3.0785905526499002)]
    cases=fixed+extra
    model=make_model()
    report=dict(dataset=str(args.dataset),dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                train_samples=len(x),validation_samples=len(vx),epochs=args.epochs,seeds=args.seeds,
                initialization='fresh paired random linear parameters; no pretrained weights',
                optimizer=dict(name='Adam',lr=.0003,batch_size=256),
                selection='minimum validation loss; evaluation does not select epochs',
                cases=cases,runs=[])
    print('OUTPUT',output,flush=True)
    for seed in args.seeds:
        torch.manual_seed(seed)
        initial=copy.deepcopy(PolicyNetwork(24,activation='silu').state_dict())
        for activation in ('silu','relu'):
            policy=PolicyNetwork(24,activation=activation)
            policy.load_state_dict(initial)
            assert all(torch.equal(policy.state_dict()[k],v) for k,v in initial.items())
            # 两个激活函数使用相同的小批次排列。
            torch.manual_seed(seed+10000)
            optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
            print(f'START seed={seed} activation={activation}',flush=True)
            start=time.perf_counter()
            history=fit(policy,optimizer,x,y,vx,vy,args.epochs)
            train_seconds=time.perf_counter()-start
            policy.eval()
            with torch.no_grad():
                mae=((policy(vx).tanh()*LIMITS)-vy).abs().mean(dim=0).tolist()
            evaluation=evaluate(model,policy,cases,seconds=40.)
            checkpoint=output/f'{activation}_seed_{seed}.pt'
            torch.save(dict(policy_state_dict=policy.state_dict(),config=dict(CONFIG,policy_activation=activation),
                            seed=seed,epochs=args.epochs,dataset_sha256=report['dataset_sha256']),checkpoint)
            run=dict(seed=seed,activation=activation,train_seconds=train_seconds,
                     validation_loss_history=history,best_validation_loss=min(history),
                     mean_torque_mae=float(np.mean(mae)),torque_mae_per_motor=mae,
                     standing=sum(r['standing'] for r in evaluation),settled=sum(r['settled'] for r in evaluation),
                     mean_final_distance=float(np.mean([r['final_distance'] for r in evaluation])),
                     evaluation=evaluation,checkpoint=str(checkpoint))
            report['runs'].append(run)
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('RESULT',json.dumps({k:v for k,v in run.items() if k not in ('validation_loss_history','evaluation')}),flush=True)
    aggregate={}
    for activation in ('silu','relu'):
        runs=[r for r in report['runs'] if r['activation']==activation]
        aggregate[activation]=dict(standing=sum(r['standing'] for r in runs),settled=sum(r['settled'] for r in runs),
            evaluations=len(runs)*len(cases),mean_torque_mae=float(np.mean([r['mean_torque_mae'] for r in runs])),
            mean_validation_loss=float(np.mean([r['best_validation_loss'] for r in runs])),
            mean_train_seconds=float(np.mean([r['train_seconds'] for r in runs])))
    report['aggregate']=aggregate
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AGGREGATE',json.dumps(aggregate),flush=True)
    print('DONE',output,flush=True)
