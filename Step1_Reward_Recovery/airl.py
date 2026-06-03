import torch
from torch import nn
import torch.nn.functional as F

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

class SocialPotentialNet(nn.Module):
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

        self.A = A
        self.B = B
        self.norm_x = 25.0   
        self.norm_y = 500.0 
        self.rx = rx
        self.ry = ry
        self.lat_weight = self.rx / self.ry 

    def helbing_potential(self, states):

        latest = states[:, -16:]
        dx_norm = latest[:, 4]  # FV 相对 LCV 的 Δx（横向）
        dy_norm = latest[:, 5]  # FV 相对 LCV 的 Δy（纵向）

        dx = dx_norm * self.norm_x
        dy = dy_norm * self.norm_y

        d_eff = torch.sqrt(dx ** 2 + (dy * self.lat_weight) ** 2 + 1e-6)
        exponent = (self.rx - d_eff) / self.B
        exponent = torch.clamp(exponent, max=10.0)
        U = self.A * torch.exp(exponent)
        U = torch.clamp(U, 0.0, 5.0)
        return -1.0 * U.unsqueeze(-1)

    def forward(self, states):
        base_phi = self.net(states)
        helbing_phi = self.helbing_potential(states)
        return base_phi + helbing_phi

class SocialAIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma, shared_phi_net,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.gamma = gamma
        self.shared_phi_net = shared_phi_net

        self.private_g = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r,
            output_activation=None
        )

        self.h = build_mlp(
            input_dim=state_shape,
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def get_reward(self, states):
        # R_total = alpha * Phi(s) + epsilon(s)
        phi = self.shared_phi_net(states)
        epsilon = self.private_g(states)
        # return self.alpha * phi + epsilon, phi, epsilon
        return epsilon, phi, epsilon

    def f(self, states, dones, next_states):
        rs, phi, epsilon = self.get_reward(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        phi = self.shared_phi_net(states)
        next_phi = self.shared_phi_net(next_states)
        social_term = self.alpha * (next_phi - phi) 
        # social_term = self.alpha * ( self.gamma * next_phi - phi )

        return rs + self.gamma * (1 - dones.unsqueeze(-1)) * next_vs - vs + social_term

    def forward(self, states, dones, log_pis, next_states):
        logits = self.f(states, dones, next_states) - log_pis.unsqueeze(-1)
        logits = torch.clamp(logits, -50.0, 50.0)
        return torch.sigmoid(logits)

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return (torch.log(logits + 1e-3) - torch.log((1 - logits) + 1e-3))
