# Actor–Critic 小实验：同一条手写假经历重复更新 200 次
# 观察：V(obs) 逐渐上升（评委习惯了），advantage 逐渐趋向 0（不再是惊喜）
# 注意：这不是 PPO，也没接 MuJoCo，机器人什么都没学会
import torch
from networks import PolicyNetwork, ValueNetwork   # 能这样导入，靠的是 networks.py 里的 __main__ 判断

policy = PolicyNetwork()     # 选手：决定做什么动作
value_net = ValueNetwork()   # 评委：预测这个状态值多少分

# 两个网络各自一个优化器，各管各的参数
policy_optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
value_optimizer = torch.optim.SGD(value_net.parameters(), lr=1e-2)
gamma = 0.99   # 折扣：未来的分打个折

"""规定一步状态"""
# 同一条假经历：每次都从 obs 出发，拿到奖励 1，到达 next_obs
# 以后接 MuJoCo：obs/reward/next_obs 由仿真给出，不再手写
obs = torch.tensor([[0.1, 0.2, 0.0, 0.0]], dtype=torch.float32)
reward = torch.tensor([1.0])
next_obs = torch.tensor([[0.05, 0.1, 0.0, 0.0]], dtype=torch.float32)

num_updates = 200   # 重复更新的次数
print_every = 20    # 每隔多少次打印一行

print("次数   V(obs)    target    advantage")

for i in range(num_updates):
    # 每次用当前策略重新抽一个动作
    mean = policy(obs)
    std = torch.full_like(mean, 0.2)
    distribution = torch.distributions.Normal(mean, std)
    action = distribution.sample()
    log_prob = distribution.log_prob(action).sum(dim=-1)   # 联合 log_prob，形状 [1]

    """计算advantage"""
    # squeeze(-1)：[1,1] → [1]，和 log_prob 形状对齐
    # 不对齐的话，批量时 [64] × [64,1] 会被扩展成 [64,64]，算错且不报错
    value = value_net(obs).squeeze(-1)             # 预期的：V(当前状态)
    next_value = value_net(next_obs).squeeze(-1)

    # target = 实际得到的 = 这步奖励 + 打折后的下一状态预期（若真正摔倒终止，则只有 reward）
    # .detach()：把 target 当固定目标，不让梯度从 next_value 那边改网络
    target = (reward + gamma * next_value).detach()
    # advantage = 实际 − 预期，即"比预期好多少"；detach 防止策略更新时顺带改到价值网络
    advantage = (target - value).detach()

    """更新价值网络"""
    # 和回归一样：让预测 value 靠近 target
    value_loss = ((value - target) ** 2).mean()
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    """更新策略网络"""
    # 与 networks.py 里的实验相同，只是 advantage 由评委算出，不再手写
    policy_loss = -(log_prob * advantage).mean()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # 打印的是本次更新前的预测，看它随次数怎么变化
    # V 停在几十（每次运行不同）而非 100：只训练过 obs 这一个状态，next_obs 之后从没训练过
    if i % print_every == 0 or i == num_updates - 1:
        print(f"{i + 1:4d}   {value.item():7.3f}   {target.item():7.3f}   {advantage.item():7.3f}")
