"""固定策略：真实轨迹状态库 + 长回报价值校准。不会更新策略网络。"""

# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
from pathlib import Path
import argparse,copy,hashlib,json,math,time
import numpy as np
import torch
import mujoco
from torch.distributions import Normal
import RL.v3_direct.direct_ppo as p
import RL.v3_direct.direct_navigation as d
from RL.common.networks import PolicyNetwork,ValueNetwork
from RL.common.lqr_imitation import RL_DIR,LIMITS

CASES={
    'train':[((.24,0.),0.),((.26,0.),0.),((0.,.24),0.),((0.,.26),0.),((-.24,0.),0.),((.17,-.17),0.)],
    'validation':[((.22,0.),0.),((0.,.23),0.),((-.22,0.),0.),((.16,-.15),.2),((0.,.253),0.)],
    'test':[((.29,0.),0.),((0.,.255),0.),((-.27,0.),0.),((.19,-.16),.35),((0.,.251),0.)]}
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION


class SnapshotEnv(p.DirectEnv):
    def snapshot(self,origin,category):
        state=np.empty(mujoco.mj_stateSize(self.model,STATE_SPEC),dtype=np.float64)
        mujoco.mj_getState(self.model,self.data,state,STATE_SPEC)
        return dict(state=torch.from_numpy(state.copy()),previous=self.previous.clone(),target=self.target.tolist(),
            observation=self.observe().clone(),origin=origin,category=category,source_time=float(self.data.time),
            distance=self.distance,state_spec=int(STATE_SPEC),config=d.CONFIG)

    def restore(self,snapshot):
        if snapshot['config']!=d.CONFIG or snapshot['state_spec']!=int(STATE_SPEC):raise ValueError('Snapshot config mismatch')
        self.reset((tuple(snapshot['target']),0.))
        mujoco.mj_setState(self.model,self.data,snapshot['state'].numpy(),STATE_SPEC)
        # 新回合从0计时；姿态、速度、控制与积分器状态来自真实轨迹。
        self.data.time=0.;mujoco.mj_forward(self.model,self.data)
        self.previous=snapshot['previous'].clone()
        self.distance=float(np.linalg.norm(self.target-self.data.xpos[self.body,:2]))
        obs=self.observe()
        if not torch.allclose(obs,snapshot['observation'],atol=1e-6,rtol=1e-6):raise RuntimeError('Snapshot observation mismatch')
        if not np.allclose(self.data.ctrl,self.previous.numpy(),atol=1e-6):raise RuntimeError('Previous action mismatch')
        assert self.streak==0 and self.arrival is None and self.steps==0
        return obs


def select_near(pool,count):
    selected=[];used=set()
    # 优先覆盖转动、缓慢残差、移动；同一轨迹同一类别不重复占满状态库。
    for category in ('spin','ready','slow','moving','spin','ready','slow','moving'):
        candidates=[s for s in pool if s['category']==category and (s['origin'],category) not in used]
        if not candidates:continue
        origins=sorted(set(s['origin'] for s in candidates))
        subset=[s for s in candidates if s['origin']==origins[0]]
        item=subset[len(subset)//2];selected.append(item);used.add((item['origin'],category))
        if len(selected)>=count:return selected
    for item in pool:
        key=(item['origin'],item['category'])
        if key not in used:selected.append(item);used.add(key)
        if len(selected)>=count:return selected
    return selected


def create_bank(policy):
    bank={};summaries={}
    for split,cases in CASES.items():
        normals=[];pool=[];episodes=[]
        for index,case in enumerate(cases):
            env=SnapshotEnv(stop_on_success=False);obs=env.reset(case);origin=f'{split}_{index}'
            normals.append(env.snapshot(origin,'normal'))
            with torch.no_grad():
                for step in range(20000):
                    obs,_,term,trunc,result=env.step(policy(obs[None])[0].tanh()*LIMITS)
                    if not term and step%250==249 and .015<=result['final_distance']<=.08:
                        if result['final_yawrate']>.3 and result['last2_mean_abs_yawrate']>.3:
                            category='spin'
                        elif result['final_distance']<=.03 and result['final_speed']<=.03 and result['final_yawrate']<=.1:
                            category='ready'
                        elif result['final_speed']>.03 or result['final_yawrate']>.1:
                            category='moving'
                        else:category='slow'
                        pool.append(env.snapshot(origin,category))
                    if term or trunc:break
            episodes.append(dict(origin=origin,**result))
        normal_count=6 if split=='train' else 2;near_count=8 if split=='train' else 3
        # 不同split使用不同目标/初始姿态轨迹；不随机打散相邻帧跨训练和测试。
        chosen=normals[:normal_count]+select_near(pool,near_count)
        if not any(s['category']=='spin' for s in chosen):raise RuntimeError(f'No spinning state in {split}; inspect bank before training')
        if not any(s['category']=='ready' for s in chosen):raise RuntimeError(f'No ready-to-settle state in {split}; inspect bank before training')
        bank[split]=chosen;summaries[split]=dict(origins=cases,episodes=episodes,available_near=len(pool),
            chosen=[{k:s[k] for k in ('origin','category','source_time','distance','target')} for s in chosen])
        print('BANK',json.dumps(dict(split=split,normal=sum(s['category']=='normal' for s in chosen),
            near=sum(s['category']!='normal' for s in chosen),categories=[s['category'] for s in chosen])),flush=True)
    return bank,summaries


def collect_returns(policy,snapshot,seed,horizon=80.,prefix=20.,stride=10):
    torch.manual_seed(seed)
    env=SnapshotEnv(max_seconds=horizon);obs=env.restore(snapshot)
    xs=[];indices=[];rewards=[];near_steps=spinning_steps=0
    with torch.no_grad():
        for step in range(round(horizon/env.dt)):
            if step%stride==0 and step*env.dt<prefix:xs.append(obs.clone());indices.append(step)
            raw=Normal(policy(obs[None])[0],.01).sample()
            obs,reward,term,trunc,result=env.step(raw.tanh()*LIMITS)
            rewards.append(reward)
            near_steps+=result['final_distance']<.08
            spinning_steps+=result['final_distance']<.08 and result['final_yawrate']>.1
            if term or trunc:break
    # 真正终止时回报完整；80秒截断时不使用价值网络自举。
    # 只用前20秒样本，截断样本至少有60秒真实后续，尾部折扣至多 exp(-6)。
    returns=np.zeros(len(rewards));tail=0.
    for i in reversed(range(len(rewards))):tail=rewards[i]+env.gamma*tail;returns[i]=tail
    x=torch.stack(xs);y=torch.tensor(returns[indices],dtype=torch.float32)
    return x,y,dict(origin=snapshot['origin'],category=snapshot['category'],seed=seed,
        initial_return=float(returns[0]),initial_observation=snapshot['observation'].tolist(),
        exact_terminal=term,cutoff=trunc,samples=len(x),near_steps=int(near_steps),spinning_steps=int(spinning_steps),**result)


def metrics(value,x,y):
    with torch.no_grad():pred=value(x).squeeze(-1)
    error=pred-y
    return dict(mse=float(error.square().mean()),mae=float(error.abs().mean()),bias=float(error.mean()),
        explained_variance=float(1-error.var(unbiased=False)/(y.var(unbiased=False)+1e-8)))


def macro_metrics(value,x,y,rollouts):
    offset=0;groups=[]
    for r in rollouts:
        n=r['samples'];m=metrics(value,x[offset:offset+n],y[offset:offset+n]);offset+=n
        groups.append(dict(origin=r['origin'],category=r['category'],seed=r['seed'],**m))
    assert offset==len(x)
    return dict(mse=float(np.mean([g['mse'] for g in groups])),mae=float(np.mean([g['mae'] for g in groups])),groups=groups)


def main(args):
    torch.set_num_threads(1);torch.manual_seed(93)
    actor_path=RL_DIR/'v3_direct/data/results/direct_imitation_candidate.pt'
    critic_path=RL_DIR/'v3_direct/data/experiments/direct_ppo_20261029_163131/candidate_1.pt'
    actor_saved=torch.load(actor_path,weights_only=True)
    policy=PolicyNetwork(27);policy.load_state_dict(actor_saved['policy_state_dict']);policy.eval()
    for par in policy.parameters():par.requires_grad_(False)
    value=ValueNetwork(27);value.load_state_dict(torch.load(critic_path,weights_only=True)['value_state_dict'])
    prior=copy.deepcopy(value).eval()
    original_actor=copy.deepcopy(policy.state_dict());actor_hash=hashlib.sha256(actor_path.read_bytes()).hexdigest()
    output=RL_DIR/'archive/data/experiments'/('value_calibration_'+time.strftime('%Y%m%d_%H%M%S'));output.mkdir(parents=True)
    print('OUTPUT',output,flush=True)
    bank,bank_summary=create_bank(policy)
    torch.save(dict(config=d.CONFIG,actor_sha256=actor_hash,bank=bank),output/'state_bank.pt')
    report=dict(config=d.CONFIG,actor_source=str(actor_path),actor_sha256=actor_hash,critic_source=str(critic_path),
        policy_updated=False,reward=p.REWARD.copy(),horizon_seconds=80,prefix_seconds=20,std=.01,
        sampling_revision=2,equal_weight_per_rollout=True,bank=bank_summary,rollouts={})
    save=lambda:(output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();datasets={}
    for split in ('train','validation','test'):
        xs=[];ys=[];summaries=[];seeds=3 if split=='test' else 1
        for index,state in enumerate(bank[split]):
            for repeat in range(seeds):
                seed={'train':93000,'validation':94000,'test':95000}[split]+index*10+repeat
                x,y,summary=collect_returns(policy,state,seed)
                xs.append(x);ys.append(y);summaries.append(summary)
                print('ROLLOUT',json.dumps(dict(split=split,index=index,repeat=repeat,category=state['category'],
                    duration=summary['duration'],failed=summary['failed'],success=summary['settled'],return0=summary['initial_return'])),flush=True)
                report['rollouts'][split]=summaries;save()
        datasets[split]=(torch.cat(xs),torch.cat(ys));report.setdefault('before',{})[split]=metrics(prior,*datasets[split]);save()
    torch.manual_seed(96)
    optimizer=torch.optim.Adam(value.parameters(),lr=.0003)
    x,y=datasets['train'];vx,vy=datasets['validation'];best=copy.deepcopy(value.state_dict())
    weights=torch.cat([torch.full((r['samples'],),len(x)/(len(report['rollouts']['train'])*r['samples'])) for r in report['rollouts']['train']])
    best_loss=macro_metrics(value,vx,vy,report['rollouts']['validation'])['mse'];best_epoch=0;history=[]
    for epoch in range(1,args.epochs+1):
        for ids in torch.randperm(len(x)).split(256):
            loss=((value(x[ids]).squeeze(-1)-y[ids]).square()*weights[ids]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(value.parameters(),1.);optimizer.step()
        if epoch%10==0:
            validation=macro_metrics(value,vx,vy,report['rollouts']['validation']);history.append(dict(epoch=epoch,validation=validation))
            if validation['mse']<best_loss:best_loss=validation['mse'];best=copy.deepcopy(value.state_dict());best_epoch=epoch
            print('FIT',epoch,json.dumps({k:v for k,v in validation.items() if k!='groups'}),flush=True)
    value.load_state_dict(best);report['selected_epoch']=best_epoch;report['history']=history
    report['after']={split:metrics(value,*xy) for split,xy in datasets.items()}
    report['macro_before']={split:macro_metrics(prior,*xy,report['rollouts'][split]) for split,xy in datasets.items()}
    report['macro_after']={split:macro_metrics(value,*xy,report['rollouts'][split]) for split,xy in datasets.items()}
    report['passed_calibration_gate']=best_epoch>0 and report['macro_after']['test']['mse']<report['macro_before']['test']['mse']
    initial=[]
    with torch.no_grad():
        for index,snapshot in enumerate(bank['test']):
            obs=snapshot['observation'][None]
            realized=[r['initial_return'] for r in report['rollouts']['test'] if r['seed']//10==(95000+index*10)//10]
            initial.append(dict(origin=snapshot['origin'],category=snapshot['category'],distance=snapshot['distance'],
                before=prior(obs).item(),after=value(obs).item(),return_mean=float(np.mean(realized)),return_std=float(np.std(realized))))
    report['heldout_initial_states']=initial
    torch.save(dict(config=d.CONFIG,value_state_dict=value.state_dict(),policy_sha256=actor_hash,action_std=.01,
        selected_epoch=best_epoch,passed_calibration_gate=report['passed_calibration_gate']),output/'critic_candidate.pt')
    torch.save(dict(config=d.CONFIG,datasets=datasets),output/'value_dataset.pt')
    assert all(torch.equal(v,original_actor[k]) for k,v in policy.state_dict().items())
    assert hashlib.sha256(actor_path.read_bytes()).hexdigest()==actor_hash
    save();print('DONE',json.dumps(dict(output=str(output),epoch=best_epoch,gate=report['passed_calibration_gate'],
        before=report['before']['test'],after=report['after']['test'],initial=initial)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=200)
    args=parser.parse_args()
    if args.epochs<10:parser.error('epochs must be at least 10')
    main(args)
