import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .reward_config import ALPHA_INIT
else:
    from reward_config import ALPHA_INIT


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


def split_latest_state(states):
    """Recover normalized vehicle states from the latest 16-dimensional frame.

    Each vehicle uses [position_lateral, position_longitudinal,
    velocity_longitudinal, velocity_lateral].
    """
    latest = states[:, -16:]

    lcv = latest[:, 0:4]

    # Current repository definition:
    # fv_rel = lcv - fv
    # nlv_rel = nlv - lcv
    # olv_rel = olv - lcv
    fv_rel = latest[:, 4:8]
    nlv_rel = latest[:, 8:12]
    olv_rel = latest[:, 12:16]

    fv = lcv - fv_rel
    nlv = lcv + nlv_rel
    olv = lcv + olv_rel

    return {
        "lcv": lcv,
        "fv": fv,
        "nlv": nlv,
        "olv": olv,
    }


def extract_private_state(states, role, history_len=4):
    """Extract only the target vehicle's normalized state history."""
    frames = states.reshape(-1, history_len, 16)
    lcv = frames[..., 0:4]
    fv_rel = frames[..., 4:8]
    if role == "lcv":
        ego = lcv
    elif role == "fv":
        ego = lcv - fv_rel
    else:
        raise ValueError(role)
    return ego.reshape(states.shape[0], -1)


class AnisotropicInteractionPotential(nn.Module):
    """Shared explicit interaction rule, evaluated for each target vehicle."""

    def __init__(self, L_long=15.0, L_lat=3.5, dir_lambda=0.3):
        super().__init__()
        self.L_long = L_long
        self.L_lat = L_lat
        self.dir_lambda = dir_lambda

    def _physical_state(self, x):
        """Convert [lat_pos, long_pos, v_long, v_lat] from normalized units."""
        lat = x[:, 0] * 25.0
        lon = x[:, 1] * 500.0
        v_long = x[:, 2] * 80.0 - 40.0
        v_lat = x[:, 3] * 80.0 - 40.0
        return lat, lon, v_long, v_lat

    def pair_potential(self, target, neighbor):
        lat_i, lon_i, v_long_i, v_lat_i = self._physical_state(target)
        lat_j, lon_j, _, _ = self._physical_state(neighbor)

        # Displacement from target i to neighbor j.
        d_long = lon_j - lon_i
        d_lat = lat_j - lat_i
        psi = torch.atan2(v_lat_i, v_long_i)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        # Global -> ego-aligned coordinates.
        dx_e = cos_psi * d_long + sin_psi * d_lat
        dy_e = -sin_psi * d_long + cos_psi * d_lat
        q = (dx_e / self.L_long) ** 2 + (dy_e / self.L_lat) ** 2
        spatial = torch.exp(-0.5 * q)
        phi = torch.atan2(dy_e, dx_e)
        direction_weight = (
            self.dir_lambda
            + (1.0 - self.dir_lambda) * (1.0 + torch.cos(phi)) / 2.0
        )
        return direction_weight * spatial

    def forward(self, states, role="lcv"):
        vehicles = split_latest_state(states)
        if role == "lcv":
            target = vehicles["lcv"]
            neighbors = [vehicles["fv"], vehicles["nlv"], vehicles["olv"]]
        elif role == "fv":
            target = vehicles["fv"]
            neighbors = [vehicles["lcv"], vehicles["nlv"], vehicles["olv"]]
        else:
            raise ValueError(f"Unsupported role: {role}")
        potentials = [self.pair_potential(target, nbr) for nbr in neighbors]
        total = torch.stack(potentials, dim=-1).sum(dim=-1)
        return total.unsqueeze(-1)


# --- 4. 新增：支持势场分解的改进判别器 (SocialAIRLDiscrim) ---
class SocialAIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma, shared_interaction_potential,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True),
                 role="lcv", history_len=None):
        super().__init__()

        if role not in ("lcv", "fv"):
            raise ValueError(f"Unsupported role: {role}")
        if history_len is None:
            history_len = state_shape // 16
        if history_len < 1 or state_shape != history_len * 16:
            raise ValueError("state_shape must equal history_len * 16")
        self.role = role
        self.history_len = history_len
        self.gamma = gamma
        self.interaction_potential = shared_interaction_potential

        # 1. 私有偏好网络 (epsilon^i)
        self.private_g = build_mlp(
            input_dim=4 * history_len,
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

        # Inverse softplus keeps the initial positive interaction weight at ALPHA_INIT.
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(ALPHA_INIT))))

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def get_reward(self, states):
        private_state = extract_private_state(
            states,
            role=self.role,
            history_len=self.history_len,
        )
        r_private = self.private_g(private_state)
        phi = self.interaction_potential(states, role=self.role)
        alpha = self.alpha
        r_total = r_private - alpha * phi
        return r_total, r_private, phi

    def f(self, states, dones, next_states):
        reward, _, _ = self.get_reward(states)
        value = self.h(states)
        next_value = self.h(next_states)
        done = dones.reshape(-1, 1).float()
        return reward + self.gamma * (1.0 - done) * next_value - value

    def forward(self, states, dones, log_pis, next_states):
        # 数值稳定的判别输出：sigmoid(f - log_pi)
        logits = self.f(states, dones, next_states) - log_pis.unsqueeze(-1)
        logits = torch.clamp(logits, -50.0, 50.0)
        return torch.sigmoid(logits)

    def calculate_reward(self, states, dones, log_pis, next_states):
        """Return recovered total reward for PPO, without discriminator shaping."""
        with torch.no_grad():
            r_total, r_private, phi = self.get_reward(states)
            return r_total

