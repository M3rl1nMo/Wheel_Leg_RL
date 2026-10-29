import torch,json
import direct_clock_ppo as m
torch.set_num_threads(1)
p=m.PolicyNetwork(27);p.load_state_dict(torch.load(m.RL_DIR/'results/direct_imitation_candidate.pt',weights_only=True)['policy_state_dict'])
bank=torch.load(m.RL_DIR/'results/direct_state_bank.pt',weights_only=True)['bank']['test']
for s in bank:
    x=s['observation'].clone().requires_grad_(True)
    j=torch.autograd.functional.jacobian(lambda z:p(z).tanh(),x)[:,-6:]
    eig=torch.linalg.eigvals(j)
    print(json.dumps(dict(category=s['category'],spectral_radius=float(eig.abs().max()),diagonal=j.diag().tolist())),flush=True)
