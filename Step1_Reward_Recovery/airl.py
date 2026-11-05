import torch
from torch import nn
import torch.nn.functional as F

if __package__:
    from .reward_config import ALPHA_INIT, DT, HISTORY_LEN, POS_LAT_SCALE, POS_LONG_SCALE, VEL_MIN, VEL_SCALE, PRED_HORIZON, PRED_STEPS
else:
    from reward_config import ALPHA_INIT, DT, HISTORY_LEN, POS_LAT_SCALE, POS_LONG_SCALE, VEL_MIN, VEL_SCALE, PRED_HORIZON, PRED_STEPS


def build_mlp(input_dim, output_dim, hidden_units=(64, 64), hidden_activation=nn.Tanh(), output_activation=None):
    layers, units = [], input_dim
    for next_units in hidden_units:
        layers.extend((nn.Linear(units, next_units), hidden_activation))
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class AIRLDiscrim(nn.Module):
    """Unstructured AIRL retained for a fair independent baseline."""
    def __init__(self, state_shape, gamma, hidden_units_r=(64, 64), hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True), hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()
        self.g = build_mlp(state_shape, 1, hidden_units_r, hidden_activation_r)
        self.h = build_mlp(state_shape, 1, hidden_units_v, hidden_activation_v)
        self.gamma = gamma

    def f(self, states, dones, next_states):
        return self.g(states) + self.gamma * (1 - dones.reshape(-1, 1).float()) * self.h(next_states) - self.h(states)

    def discriminator_logits(self, states, dones, log_pis, next_states):
        return self.f(states, dones, next_states) - log_pis.reshape(-1, 1)

    def forward(self, states, dones, log_pis, next_states):
        return torch.sigmoid(self.discriminator_logits(states, dones, log_pis, next_states))

    def calculate_policy_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            return self.discriminator_logits(states, dones, log_pis, next_states)

    calculate_reward = calculate_policy_reward


def split_latest_state(states):
    latest = states[:, -16:]
    lcv, fv_rel, nlv_rel, olv_rel = latest[:, 0:4], latest[:, 4:8], latest[:, 8:12], latest[:, 12:16]
    return {"lcv": lcv, "fv": lcv - fv_rel, "nlv": lcv + nlv_rel, "olv": lcv + olv_rel}


def extract_private_state(states, role, history_len=HISTORY_LEN):
    frames = states.reshape(-1, history_len, 16)
    lcv, fv_rel = frames[..., 0:4], frames[..., 4:8]
    if role == "lcv":
        return lcv.reshape(states.shape[0], -1)
    if role == "fv":
        return (lcv - fv_rel).reshape(states.shape[0], -1)
    raise ValueError(role)


def extract_motion_features(states, role, history_len=HISTORY_LEN):
    """Causal ego-only [relative position, velocity, acceleration, jerk]."""
    frames = states.reshape(-1, history_len, 16)
    ego = frames[..., 0:4] if role == "lcv" else frames[..., 0:4] - frames[..., 4:8]
    position, velocity = ego[..., 0:2], ego[..., 2:4]
    rel_position = position - position[:, :1]
    acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1]) / DT
    jerk = torch.diff(acceleration, dim=1, prepend=acceleration[:, :1]) / DT
    return torch.cat((rel_position, velocity, acceleration, jerk), dim=-1).transpose(1, 2)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x):
        y = self.conv(x)
        return y[..., :-self.pad] if self.pad else y


class TemporalPrivateReward(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(input_dim, hidden_dim, dilation=1), nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, dilation=2), nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, dilation=4), nn.ReLU(),
        )
        self.reward_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features):
        return self.reward_head(self.encoder(features)[:, :, -1])


class AnisotropicInteractionPotential(nn.Module):
    """One bounded, learnable dynamic interaction kernel shared across roles."""
    def __init__(self, L_long=15.0, L_lat=3.5, dir_lambda=0.3):
        super().__init__()
        self._l_long_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(L_long)))))
        self._l_lat_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(L_lat)))))
        self._dir_lambda_raw = nn.Parameter(torch.logit(torch.tensor(float(dir_lambda)).clamp(1e-4, 1 - 1e-4)))

    @property
    def L_long(self): return F.softplus(self._l_long_raw).clamp_min(1e-3)

    @property
    def L_lat(self): return F.softplus(self._l_lat_raw).clamp_min(1e-3)

    @property
    def dir_lambda(self): return torch.sigmoid(self._dir_lambda_raw)

    @staticmethod
    def _physical_state(x):
        return x[:, 0] * POS_LAT_SCALE, x[:, 1] * POS_LONG_SCALE, x[:, 2] * VEL_SCALE + VEL_MIN, x[:, 3] * VEL_SCALE + VEL_MIN

    def pair_potential(self, target, neighbor):
        lat_i, lon_i, vl_i, vt_i = self._physical_state(target)
        lat_j, lon_j, vl_j, vt_j = self._physical_state(neighbor)
        psi = torch.atan2(vt_i, vl_i)
        cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)
        d_long, d_lat = lon_j - lon_i, lat_j - lat_i
        rel_long, rel_lat = vl_j - vl_i, vt_j - vt_i
        x0, y0 = cos_psi * d_long + sin_psi * d_lat, -sin_psi * d_long + cos_psi * d_lat
        vx, vy = cos_psi * rel_long + sin_psi * rel_lat, -sin_psi * rel_long + cos_psi * rel_lat
        horizon = torch.linspace(DT, PRED_HORIZON, PRED_STEPS, device=target.device, dtype=target.dtype)
        x_pred, y_pred = x0[:, None] + vx[:, None] * horizon, y0[:, None] + vy[:, None] * horizon
        q = (x_pred / self.L_long).square() + (y_pred / self.L_lat).square()
        spatial = torch.exp(-q)
        distance = torch.sqrt(x_pred.square() + y_pred.square())
        cos_phi = torch.where(distance > 1e-4, x_pred / distance.clamp_min(1e-4), torch.ones_like(distance))
        direction_weight = self.dir_lambda + (1.0 - self.dir_lambda) * (1.0 + cos_phi) / 2.0
        return (direction_weight * spatial).mean(dim=1)

    def forward(self, states, role="lcv"):
        vehicles = split_latest_state(states)
        if role == "lcv":
            target, neighbors = vehicles["lcv"], (vehicles["fv"], vehicles["nlv"], vehicles["olv"])
        elif role == "fv":
            target, neighbors = vehicles["fv"], (vehicles["lcv"], vehicles["nlv"], vehicles["olv"])
        else:
            raise ValueError(role)
        return torch.stack([self.pair_potential(target, other) for other in neighbors], dim=-1).sum(dim=-1, keepdim=True)


class SocialAIRLDiscrim(nn.Module):
    def __init__(self, state_shape, gamma, shared_interaction_potential, role="lcv", history_len=HISTORY_LEN,
                 hidden_units_v=(64, 64), **kwargs):
        super().__init__()
        if role not in ("lcv", "fv") or state_shape != history_len * 16:
            raise ValueError("SocialAIRLDiscrim requires role lcv/fv and state_shape == history_len * 16")
        self.role, self.history_len, self.gamma = role, history_len, gamma
        self.interaction_potential = shared_interaction_potential
        self.private_g = TemporalPrivateReward()
        self.h = build_mlp(state_shape, 1, hidden_units_v, nn.ReLU())
        self.alpha_raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(ALPHA_INIT))))

    @property
    def alpha(self): return F.softplus(self.alpha_raw)

    def get_reward(self, states):
        r_private = self.private_g(extract_motion_features(states, self.role, self.history_len))
        phi = self.interaction_potential(states, role=self.role)
        return r_private - self.alpha * phi, r_private, phi

    def f(self, states, dones, next_states):
        reward, _, _ = self.get_reward(states)
        return reward + self.gamma * (1.0 - dones.reshape(-1, 1).float()) * self.h(next_states) - self.h(states)

    def discriminator_logits(self, states, dones, log_pis, next_states):
        return self.f(states, dones, next_states) - log_pis.reshape(-1, 1)

    def forward(self, states, dones, log_pis, next_states):
        return torch.sigmoid(self.discriminator_logits(states, dones, log_pis, next_states))

    def calculate_policy_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            return self.discriminator_logits(states, dones, log_pis, next_states)

