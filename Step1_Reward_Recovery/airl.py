import torch
from torch import nn
import torch.nn.functional as F


# --- 1. 基础组件 ---
def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


# --- 2. 原始 AIRL 判别器 (用于标准模式) ---
class AIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # 数值稳定的判别输出：sigmoid(f - log_pi)
        logits = self.f(states, dones, next_states) - log_pis.unsqueeze(-1)
        logits = torch.clamp(logits, -50.0, 50.0)
        return torch.sigmoid(logits)

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return (torch.log(logits + 1e-3) - torch.log((1 - logits) + 1e-3))


# --- 3. 新增：共享社会势场网络 (SocialPotential) ---
class SocialPotentialNet(nn.Module):
    """
    Helbing 椭圆势能版的共享势场网络，兼容旧版 MLP 势场并叠加显式势能项。
    """
    def __init__(self, state_shape, hidden_units=[64, 64], hidden_activation=nn.ReLU(inplace=True),
                 A=1, B=8.0, rx=6.0, ry=2.0):
        super().__init__()
        layers = []
        units = state_shape
        for next_units in hidden_units:
            layers.append(nn.Linear(units, next_units))
            layers.append(hidden_activation)
            units = next_units
        layers.append(nn.Linear(units, 1))  # 输出标量势能 Phi
        self.net = nn.Sequential(*layers)

        # Helbing 势场参数
        self.A = A
        self.B = B
        # 与环境归一化保持一致：index0/4 为横向 ∈[0,25]，index1/5 为纵向 ∈[0,500]
        self.norm_x = 25.0   # lateral
        self.norm_y = 500.0  # longitudinal
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry  # 横向距离放大倍率

    def helbing_potential(self, states):
        """
        椭圆排斥势能：使用最新帧的相对 dx/dy（索引 4/5）。
        """
        latest = states[:, -16:]
        dx_norm = latest[:, 4]  # FV 相对 LCV 的 Δx（横向）
        dy_norm = latest[:, 5]  # FV 相对 LCV 的 Δy（纵向）

        dx = dx_norm * self.norm_x
        dy = dy_norm * self.norm_y

        d_eff = torch.sqrt(dx ** 2 + (dy * self.lat_weight) ** 2 + 1e-6)
        exponent = (self.rx - d_eff) / self.B
        # 保护：防止指数爆炸
        exponent = torch.clamp(exponent, max=10.0)
        U = self.A * torch.exp(exponent)
        # 二次保护：限制势能最大值
        U = torch.clamp(U, 0.0, 5.0)
        return -1.0 * U.unsqueeze(-1)

    def forward(self, states):
        base_phi = self.net(states)
        helbing_phi = self.helbing_potential(states)
        return base_phi + helbing_phi


# --- 4. 新增：支持势场分解的改进判别器 (SocialAIRLDiscrim) ---
class SocialAIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma, shared_phi_net,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.gamma = gamma
        self.shared_phi_net = shared_phi_net  # 外部传入的共享网络实例

        # 1. 私有偏好网络 (epsilon^i)
        self.private_g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r,
            output_activation=None  # 去掉 ReLU，避免奖励恒为正导致判别器饱和
        )

        # 2. 势能函数 V (用于 Advantage 估计)
        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        # 3. 可学习的权重 alpha (控制对社会势场的依赖程度)；初始较小，避免判别器过度依赖 Phi
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def get_reward(self, states):
        # R_total = alpha * Phi(s) + epsilon(s)
        phi = self.shared_phi_net(states)
        epsilon = self.private_g(states)
        # return self.alpha * phi + epsilon, phi, epsilon
        return epsilon, phi, epsilon

    # def f(self, states, dones, next_states):
    #     # AIRL 的核心：f(s,s') = g(s) + gamma * h(s') - h(s)
    #     # 这里 g(s) 是复合奖励
    #     rs, _, _ = self.get_reward(states)
    #     vs = self.h(states)
    #     next_vs = self.h(next_states)
    #     # f 实际上近似于优势函数 A(s, a)
    #     return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs
    def f(self, states, dones, next_states):
        # ---- 1. 计算私有 reward ----
        rs, phi, epsilon = self.get_reward(states)

        # ---- 2. 值函数 advantage 项 ----
        vs = self.h(states)
        next_vs = self.h(next_states)

        # ---- 3. 社会势差分项 beta*(next_phi - phi) ----
        phi = self.shared_phi_net(states)
        next_phi = self.shared_phi_net(next_states)
        social_term = self.alpha * (next_phi - phi)  # α = β，可换名
        # social_term = self.alpha * ( self.gamma * next_phi - phi )

        # ---- 4. 合成 AIRL 的 f(s,a,s') ----
        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs + social_term

    def forward(self, states, dones, log_pis, next_states):
        # 数值稳定的判别输出：sigmoid(f - log_pi)
        logits = self.f(states, dones, next_states) - log_pis.unsqueeze(-1)
        logits = torch.clamp(logits, -50.0, 50.0)
        return torch.sigmoid(logits)

    def calculate_reward(self, states, dones, log_pis, next_states):
        # 用于 PPO 更新的伪奖励
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return (torch.log(logits + 1e-3) - torch.log((1 - logits) + 1e-3))
