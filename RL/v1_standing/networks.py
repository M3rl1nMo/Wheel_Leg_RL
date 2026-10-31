
# 支持从项目根目录直接运行此文件。
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "RL").parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))
import torch
from torch import nn
from pathlib import Path
import sys

# 策略网络（Actor）：看观测 obs，输出 6 个电机动作的"均值"
# 继承 nn.Module：自动管理参数；policy(obs) 会经 __call__ 自动调用 forward
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()   # 先初始化 nn.Module 内部结构（不是训练）

        self.network = nn.Sequential(
            nn.Linear(4, 32),    # 4 个观测 → 32 个隐藏值
            nn.ReLU(),
            nn.Linear(32, 6)     # → 6 个均值，对应 6 个电机
        )

    def forward(self, obs):
        mean = self.network(obs)   # obs 形状 [批次, 4] → mean 形状 [批次, 6]
        return mean
# 价值网络（Critic）：看观测 obs，预测"从这个状态往后，大概总共能拿多少分"
# 它输出的是预测，不是奖励；只看 obs 不看 action，评价的是状态本身好不好
class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 1)   # 只输出 1 个数：对未来总得分的预测
            )
        def forward(self, obs):
            value = self.network(obs)   # 形状 [批次, 1]
            return value




# 只有直接运行本文件时才执行；被 import 时整块跳过（__name__ 不是 "__main__"）
if __name__ == "__main__":

    policy = PolicyNetwork()   # 创建网络，权重随机初始化，每次运行输出都不同
    print("测试网络")
    obs = torch.tensor([[0.1, 0.2, 0.0, 0.0]], dtype=torch.float32)   # 手写的假观测，不是机器人数据
    mean = policy(obs)
    print(obs.shape)
    print(mean.shape)
    print(mean)
    # 正态分布：mean 是中心，std 是随机偏多远；6 个分量各自独立
    # Normal 不会改变 mean，只是创建一个存着 mean/std、能 sample/log_prob 的对象
    std = torch.full_like(mean, 0.2)   # 和 mean 同形状，全填 0.2（教学先固定）
    distribution = torch.distributions.Normal(mean, std)
    action = distribution.sample()     # 每次抽样都不同 = 探索；结果没有 grad_fn，和网络断开
    # log_prob：这个 action 在分布里"有多常见"（对数密度，可为正）
    # 它由 mean 算出，带 grad_fn，梯度能经它传回网络；对 mean 的导数 = (action-mean)/std²
    # 参考值（std=0.2）：在均值上 ≈ 0.6905；偏离 0.4 ≈ -1.3095
    log_prob = distribution.log_prob(action)   # 形状 [1, 6]
    """开始梯度更新"""
    # advantage > 0：动作好 → mean 靠近 action；< 0：动作坏 → mean 远离 action
    advantage = -1.0
    # 优化器只会让 loss 变小，想让 log_prob×advantage 变大，所以加负号
    # sum(dim=-1)：沿最后一维（6 个电机）相加 = 联合 log_prob，每条数据各一个
    loss = -(log_prob.sum(dim=-1) * advantage).mean()
    # lr 要和梯度大小一起看：std 小 → 梯度大 → 同样 lr 步子也大（lr=0.01 时一步挪了约 65%）
    optimizer = torch.optim.SGD(policy.parameters(), lr=1.0E-03)
    optimizer.zero_grad()   # 清旧梯度
    loss.backward()         # 算梯度（只算，不改参数）
    optimizer.step()        # 真正改参数；注意要带括号，不带括号什么都不做
    new_mean = policy(obs)  # 改的是权重，mean 不会自己变，要重新算
    new_log_prob = torch.distributions.Normal(new_mean, std).log_prob(action).sum(dim=-1)
    # .detach() 只是去掉 grad_fn 让打印干净；.item() 把单个数的张量变成普通数字
    print("更新前 mean：", mean.detach())
    print("动作 action：", action)
    print("更新后 mean：", new_mean.detach())
    print("mean 变化量：", (new_mean - mean).detach())
    print("action - 旧mean：", (action - mean).detach())
    print("更新前联合 log_prob：", log_prob.sum(dim=-1).item())
    print("更新后联合 log_prob：", new_log_prob.item())
    """价值网络测试"""
    value_net = ValueNetwork()
    value = value_net(obs)   # 形状 [1, 1]，未训练时是随机猜测
    print("价值预测：", value)
    print("价值形状：", value.shape)


