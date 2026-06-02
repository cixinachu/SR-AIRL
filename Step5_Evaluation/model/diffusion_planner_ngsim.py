"""
NGSIM-adapted Diffusion-Planner baseline.

This module follows the design pattern of ZhengYinan-AIR/Diffusion-Planner more
closely than the earlier lightweight baseline:
- an encoder produces fused scene/agent/route tokens
- a decoder denoises noised future trajectories with DiT-style blocks
- training predicts x_start under a VP-SDE perturbation
- sampling keeps the current state fixed and denoises future states

The official project targets nuPlan closed-loop planning with map, route, ego,
and neighbor tensors. This adaptation keeps the same architectural idea but
maps this repository's NGSIM lane-changing data into a local four-agent setup
LCV/FV/NLV/OLV and writes plan_data.pkl for the existing evaluator.
"""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Step5_Evaluation.utils.dataset import SequenceDataset


ARCHITECTURE_VERSION = "ngsim_official_style_encoder_decoder_v2"
DT = 0.1
AGENT_COUNT = 4
LANE_CENTERS = [
    (3.6576 + 7.3152) / 2,
    (7.3152 + 10.9728) / 2,
    (10.9728 + 14.6304) / 2,
    (14.6304 + 18.2880) / 2,
    (18.2880 + 21.9456) / 2,
]
BOUND_MAX = torch.tensor([168.4894, 12.1933], dtype=torch.float32)
BOUND_MIN = torch.tensor([31.9857, -7.8002], dtype=torch.float32)


class GoalPredictor(torch.nn.Module):
    def __init__(self, context_dim: int, input_size: int = 2):
        super().__init__()
        self.context_dim = context_dim
        self.lcv_gru = nn.GRU(input_size=16, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.fv_gru = nn.GRU(input_size=4, hidden_size=context_dim, num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, lcv: torch.Tensor, fv: torch.Tensor, nlv: torch.Tensor, olv: torch.Tensor) -> torch.Tensor:
        x = torch.cat([lcv, fv, nlv, olv], dim=-1)
        _, hidden_lcv = self.lcv_gru(x)
        return self.mlp(hidden_lcv[0])


@dataclass
class PlannerNorm:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> "PlannerNorm":
        return PlannerNorm(self.mean.to(device), self.std.to(device))

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        dim = state.shape[-1]
        mean = self.mean[:dim].to(device=state.device, dtype=state.dtype)
        std = self.std[:dim].to(device=state.device, dtype=state.dtype)
        return (state - mean) / std

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        dim = state.shape[-1]
        mean = self.mean[:dim].to(device=state.device, dtype=state.dtype)
        std = self.std[:dim].to(device=state.device, dtype=state.dtype)
        return state * std + mean

    def normalize_traj(self, traj: torch.Tensor) -> torch.Tensor:
        return self.normalize_state(traj)

    def denormalize_traj(self, traj: torch.Tensor) -> torch.Tensor:
        return self.denormalize_state(traj)

    def normalize_state4(self, state: torch.Tensor) -> torch.Tensor:
        return self.normalize_state(state)


def trajectory_process(trajectory: pd.DataFrame, decision_time: float) -> pd.DataFrame:
    trajectory = trajectory.copy()
    trajectory.columns = ["frame", "id", "y", "x", "width", "height", "laneId"]
    trajectory["v_x"] = trajectory["x"].diff() / DT
    trajectory["v_y"] = trajectory["y"].diff() / DT
    trajectory["a_x"] = trajectory["v_x"].diff() / DT
    trajectory["a_y"] = trajectory["v_y"].diff() / DT
    trajectory = trajectory[trajectory["frame"] >= decision_time - 20].copy()
    trajectory = trajectory.replace([np.inf, -np.inf], np.nan)
    trajectory = trajectory.bfill().ffill().fillna(0.0)
    return trajectory


def extract_scene_arrays(pair: pd.DataFrame, horizon: int = 100, obs_len: int = 20) -> Optional[Dict[str, np.ndarray]]:
    decision_frame = float(pair["decision_frame"].values[0])
    specs = {
        "lcv": ["frame", "id_x", "y_x", "x_x", "width_x", "height_x", "laneId_x"],
        "fv": ["frame", "id_y", "y_y", "x_y", "width_y", "height_y", "laneId_y"],
        "nlv": ["frame", "id", "y", "x", "width", "height", "laneId"],
        "olv": ["frame", "id_z", "y_z", "x_z", "width_z", "height_z", "laneId_z"],
    }
    out: Dict[str, np.ndarray] = {}
    for key, cols in specs.items():
        if any(col not in pair for col in cols):
            return None
        traj = trajectory_process(pair[cols], decision_frame)
        arr = traj[["y", "x", "v_x", "v_y", "a_x", "a_y"]].values.astype(np.float32)
        if len(arr) < horizon:
            if len(arr) == 0:
                return None
            arr = np.concatenate([arr, np.repeat(arr[-1:], horizon - len(arr), axis=0)], axis=0)
        out[key] = arr[:horizon]

    base_y, base_x = out["lcv"][0, 0], out["lcv"][0, 1]
    for key in out:
        out[key] = out[key].copy()
        out[key][:, 0] -= base_y
        out[key][:, 1] -= base_x

    agents_full = np.stack([out["lcv"], out["fv"], out["nlv"], out["olv"]], axis=0).astype(np.float32)
    agents_state = agents_full[:, :, :4].astype(np.float32)
    hist = agents_state[:, :obs_len, :].astype(np.float32)
    return {
        "agents_full": agents_full,
        "agents_state": agents_state,
        "target_full": out["lcv"].astype(np.float32),
        "target_state": out["lcv"][:, :4].astype(np.float32),
        "hist": hist,
        "terminal": out["lcv"][-1, :4].astype(np.float32),
        "base_yx": np.asarray([base_y, base_x], dtype=np.float32),
    }


def build_route_lanes(current: torch.Tensor, terminal: torch.Tensor, lane_points: int = 20) -> torch.Tensor:
    """Build a pseudo-route tensor [lane, point, 4] from current to terminal."""
    if current.ndim == 1:
        current = current.unsqueeze(0)
    if terminal.ndim == 1:
        terminal = terminal.unsqueeze(0)
    alpha = torch.linspace(0.0, 1.0, lane_points, device=current.device, dtype=current.dtype).view(1, lane_points, 1)
    start_pos = current[:, None, :2]
    end_pos = terminal[:, None, :2]
    pos = start_pos * (1.0 - alpha) + end_pos * alpha
    delta = end_pos - start_pos
    direction = delta.expand(-1, lane_points, -1)
    return torch.cat([pos, direction], dim=-1).unsqueeze(1)


class NGSIMPlannerDataset(Dataset):
    def __init__(self, data_path: Path, split: str, horizon: int = 100, obs_len: int = 20, norm: Optional[PlannerNorm] = None):
        with data_path.open("rb") as f:
            data = pickle.load(f)
        data = list(data.values()) if isinstance(data, dict) else list(data)
        if split == "train":
            data = data[:150]
        elif split == "test":
            data = data[150:]
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.items = [item for pair in data if (item := extract_scene_arrays(pair, horizon, obs_len)) is not None]
        if not self.items:
            raise RuntimeError(f"No valid {split} scenes found in {data_path}")
        self.horizon = horizon
        self.obs_len = obs_len
        self.norm = norm or self.compute_norm()

    def compute_norm(self) -> PlannerNorm:
        states = torch.from_numpy(np.stack([item["agents_state"] for item in self.items], axis=0)).float()
        flat = states.reshape(-1, states.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp_min(1e-4)
        return PlannerNorm(mean, std)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        agents = torch.from_numpy(item["agents_state"]).float()
        hist = torch.from_numpy(item["hist"]).float()
        terminal = torch.from_numpy(item["terminal"]).float()
        target_state = torch.from_numpy(item["target_state"]).float()
        target_full = torch.from_numpy(item["target_full"]).float()
        agents_n = self.norm.normalize_state(agents)
        hist_n = self.norm.normalize_state(hist)
        terminal_n = self.norm.normalize_state(terminal)
        current_n = agents_n[:, 0, :]
        route_lanes = build_route_lanes(current_n[0], terminal_n)
        return {
            "agents": agents_n,
            "hist": hist_n,
            "current_states": current_n,
            "terminal": terminal_n,
            "route_lanes": route_lanes.squeeze(0),
            "target": self.norm.normalize_state(target_state),
            "target_full": target_full,
            "base_yx": torch.from_numpy(item["base_yx"]).float(),
        }


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None):
        super().__init__()
        out_features = out_features or in_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Official-style DiT block with adaLN modulation and cross attention."""

    def __init__(self, hidden_dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm4 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp2 = Mlp(hidden_dim, int(hidden_dim * mlp_ratio))
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, x: torch.Tensor, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), x, x, need_weights=False)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.cross_attn(self.norm3(x), y, y, need_weights=False)[0]
        x = self.mlp2(self.norm4(x))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class AgentHistoryEncoder(nn.Module):
    def __init__(self, obs_len: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(obs_len * state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(AGENT_COUNT, hidden_dim)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        batch, agents = hist.shape[:2]
        tokens = self.encoder(hist)
        agent_ids = torch.arange(agents, device=hist.device).unsqueeze(0).expand(batch, -1).clamp_max(AGENT_COUNT - 1)
        return tokens + self.type_embedding(agent_ids)


class RouteEncoder(nn.Module):
    def __init__(self, hidden_dim: int, lane_points: int = 20):
        super().__init__()
        self.lane_points = lane_points
        self.point_net = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.lane_net = nn.Sequential(nn.Linear(hidden_dim * lane_points, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, route_lanes: torch.Tensor) -> torch.Tensor:
        batch, lanes, points, _ = route_lanes.shape
        point_tokens = self.point_net(route_lanes)
        return self.lane_net(point_tokens.reshape(batch, lanes, points * point_tokens.shape[-1]))


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class NGSIMEncoder(nn.Module):
    def __init__(self, obs_len: int, hidden_dim: int, heads: int):
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder(obs_len, state_dim=4, hidden_dim=hidden_dim)
        self.route_encoder = RouteEncoder(hidden_dim)
        self.fusion_encoder = FusionEncoder(hidden_dim, heads=heads, layers=2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        agent_tokens = self.agent_encoder(inputs["hist"])
        route_tokens = self.route_encoder(inputs["route_lanes"])
        encoding = self.fusion_encoder(torch.cat([agent_tokens, route_tokens], dim=1))
        return {"encoding": encoding, "agent_tokens": agent_tokens, "route_tokens": route_tokens}


class VPSDELinear:
    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, eps: float = 1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_mean_coeff = -0.25 * t.pow(2) * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        while log_mean_coeff.ndim < x.ndim:
            log_mean_coeff = log_mean_coeff.unsqueeze(-1)
        mean = torch.exp(log_mean_coeff) * x
        std = torch.sqrt(torch.clamp(1.0 - torch.exp(2.0 * log_mean_coeff), min=1e-12))
        return mean, std


class NGSIMDecoder(nn.Module):
    def __init__(
        self,
        horizon: int,
        state_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
    ):
        super().__init__()
        self.horizon = horizon
        self.state_dim = state_dim
        self.sde = VPSDELinear()
        self.x_embedder = nn.Linear(horizon * state_dim, hidden_dim)
        self.agent_embed = nn.Embedding(AGENT_COUNT, hidden_dim)
        self.time_embedder = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, horizon * state_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.ada_ln[-1].weight, 0)
            nn.init.constant_(block.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, inputs: Dict[str, torch.Tensor], encoder_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = inputs["sampled_trajectories"]
        batch, agents = x.shape[:2]
        flat = x.reshape(batch, agents, self.horizon * self.state_dim)
        agent_ids = torch.arange(agents, device=x.device).unsqueeze(0).expand(batch, -1).clamp_max(AGENT_COUNT - 1)
        x_tokens = self.x_embedder(flat) + self.agent_embed(agent_ids)
        time_token = self.time_embedder(inputs["diffusion_time"])
        y = encoder_outputs["encoding"]
        for block in self.blocks:
            x_tokens = block(x_tokens, time_token, y)
        score = self.final_layer(x_tokens, time_token).reshape(batch, agents, self.horizon, self.state_dim)
        score[:, :, 0, :] = inputs["current_states"]
        return {"score": score}

    @torch.no_grad()
    def sample(
        self,
        model: "NGSIMDiffusionPlanner",
        inputs: Dict[str, torch.Tensor],
        encoder_outputs: Dict[str, torch.Tensor],
        steps: int,
    ) -> torch.Tensor:
        current = inputs["current_states"]
        batch, agents, state_dim = current.shape
        x = torch.randn(batch, agents, self.horizon, state_dim, device=current.device, dtype=current.dtype)
        x[:, :, 0, :] = current
        time_grid = torch.linspace(1.0, self.sde.eps, steps + 1, device=current.device, dtype=current.dtype)

        for idx in range(steps):
            t = time_grid[idx].expand(batch)
            t_next = time_grid[idx + 1].expand(batch)
            step_inputs = dict(inputs)
            step_inputs["sampled_trajectories"] = x
            step_inputs["diffusion_time"] = t
            x0 = self.forward(step_inputs, encoder_outputs)["score"]
            x0[:, :, 0, :] = current

            mean_t, std_t = self.sde.marginal_prob(x0[:, :, 1:, :], t)
            eps = (x[:, :, 1:, :] - mean_t) / std_t.clamp_min(1e-6)
            if idx == steps - 1:
                x[:, :, 1:, :] = x0[:, :, 1:, :]
            else:
                mean_next, std_next = self.sde.marginal_prob(x0[:, :, 1:, :], t_next)
                x[:, :, 1:, :] = mean_next + std_next * eps
            x[:, :, 0, :] = current
        return x


class NGSIMDiffusionPlanner(nn.Module):
    def __init__(
        self,
        horizon: int = 100,
        transition_dim: int = 4,
        hidden_dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        obs_len: int = 20,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.encoder = NGSIMEncoder(obs_len=obs_len, hidden_dim=hidden_dim, heads=heads)
        self.decoder = NGSIMDecoder(horizon=horizon, state_dim=transition_dim, hidden_dim=hidden_dim, depth=depth, heads=heads)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        encoder_outputs = self.encoder(inputs)
        decoder_outputs = self.decoder(inputs, encoder_outputs)
        return encoder_outputs, decoder_outputs

    @torch.no_grad()
    def sample(self, inputs: Dict[str, torch.Tensor], steps: int) -> torch.Tensor:
        encoder_outputs = self.encoder(inputs)
        return self.decoder.sample(self, inputs, encoder_outputs, steps)



def state4_to_full6(states: torch.Tensor) -> torch.Tensor:
    vy = states[:, 3]
    vx = states[:, 2]
    ax = torch.zeros_like(vx)
    ay = torch.zeros_like(vy)
    ax[1:] = (vx[1:] - vx[:-1]) / DT
    ay[1:] = (vy[1:] - vy[:-1]) / DT
    # Keep the historical column convention [y, x, vx, vy, ax, ay].
    return torch.stack([states[:, 0], states[:, 1], vx, vy, ax, ay], dim=-1)


def train_dp_model(
    data_path: Path,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    diffusion_steps: int,
    hidden_dim: int,
    depth: int,
    heads: int,
    device: torch.device,
    seed: int,
    force: bool = False,
) -> Tuple[NGSIMDiffusionPlanner, PlannerNorm]:
    if checkpoint_path.exists() and not force:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("architecture_version") == ARCHITECTURE_VERSION:
            norm = PlannerNorm(ckpt["norm_mean"].float(), ckpt["norm_std"].float()).to(device)
            model = NGSIMDiffusionPlanner(hidden_dim=ckpt["hidden_dim"], depth=ckpt["depth"], heads=ckpt["heads"]).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()
            return model, norm
        print(f"[warn] Existing checkpoint uses an old architecture. Retraining: {checkpoint_path}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set = NGSIMPlannerDataset(data_path, split="train")
    norm = train_set.norm.to(device)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    model = NGSIMDiffusionPlanner(hidden_dim=hidden_dim, depth=depth, heads=heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in tqdm(range(epochs), desc="Training DP-adapted planner", unit="epoch"):
        losses = []
        for batch in loader:
            agents = batch["agents"].to(device)
            hist = batch["hist"].to(device)
            current = batch["current_states"].to(device)
            route_lanes = batch["route_lanes"].to(device)
            batch_size_now = agents.shape[0]

            future = agents[:, :, 1:, :]
            t = torch.rand(batch_size_now, device=device).clamp_min(model.decoder.sde.eps)
            noise = torch.randn_like(future)
            mean, std = model.decoder.sde.marginal_prob(future, t)
            noised_future = mean + std * noise
            sampled = torch.cat([current.unsqueeze(2), noised_future], dim=2)

            inputs = {
                "hist": hist,
                "current_states": current,
                "route_lanes": route_lanes,
                "sampled_trajectories": sampled,
                "diffusion_time": t,
            }
            _, decoder_outputs = model(inputs)
            pred = decoder_outputs["score"][:, :, 1:, :]
            ego_loss = F.smooth_l1_loss(pred[:, 0], future[:, 0])
            neighbor_loss = F.smooth_l1_loss(pred[:, 1:], future[:, 1:])
            terminal_loss = F.smooth_l1_loss(pred[:, 0, -1], future[:, 0, -1])
            loss = ego_loss + 0.5 * neighbor_loss + 0.2 * terminal_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            tqdm.write(f"epoch {epoch + 1}/{epochs}, loss={np.mean(losses):.5f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "model": model.state_dict(),
            "norm_mean": norm.mean.detach().cpu(),
            "norm_std": norm.std.detach().cpu(),
            "hidden_dim": hidden_dim,
            "depth": depth,
            "heads": heads,
            "diffusion_steps": diffusion_steps,
            "epochs": epochs,
        },
        checkpoint_path,
    )
    model.eval()
    return model, norm


def load_goal_predictor(goal_model_path: Optional[Path], device: torch.device):
    if goal_model_path is None or not goal_model_path.exists():
        return None
    add_safe_globals([GoalPredictor])
    setattr(sys.modules["__main__"], "GoalPredictor", GoalPredictor)
    model = torch.load(goal_model_path, map_location=device, weights_only=False).to(device)
    model.eval()
    return model


def predict_goal_from_sequence_batch(batch, goal_predictor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if goal_predictor is None:
        true_x = batch[0][0, :, 3].detach().cpu()
        true_vx = batch[0][0, :, 4].detach().cpu() if batch[0].shape[-1] > 4 else torch.zeros_like(true_x)
        return true_x[-1], true_vx[-1]
    goal_state = goal_predictor(
        batch[2][0].to(device),
        batch[2][1].to(device),
        batch[2][2].to(device),
        batch[2][3].to(device),
    )
    bounds_max = BOUND_MAX.to(device)
    bounds_min = BOUND_MIN.to(device)
    goal_state = goal_state * (bounds_max - bounds_min) + bounds_min
    return goal_state[0].detach().cpu(), goal_state[1].detach().cpu()


def build_terminal_candidates_from_batch(batch, goal_x: torch.Tensor, goal_v: torch.Tensor, limit: int) -> List[torch.Tensor]:
    true_x = batch[0][0, :, 3].detach().cpu().unsqueeze(-1)
    true_y = batch[0][0, :, 2].detach().cpu().unsqueeze(-1)
    cond0 = batch[1][0].clone().float()
    cond0[:, 0] -= true_y[0, :]
    cond0[:, 1] -= true_x[0, :]
    noise_vx = goal_v + cond0[:, 2].detach().cpu()
    nearest_lane = LANE_CENTERS[int(np.abs(np.subtract.outer(LANE_CENTERS, true_y[-1])).argmin(0)[0])]
    potent_y = [nearest_lane - true_y[0]]
    potent_x = [goal_x - 2 * i for i in range(1, 10)] + [goal_x + 2 * i for i in range(10)]
    potent_vy = [0]
    potent_vx = [noise_vx]
    out = []
    for final_pos in product(product(potent_y, potent_x), product(potent_vx, potent_vy)):
        out.append(torch.tensor(list(final_pos[0] + final_pos[1]), dtype=torch.float32).reshape(4))
        if len(out) >= limit:
            break
    return out


def generate_dp_plan_data(
    model: NGSIMDiffusionPlanner,
    norm: PlannerNorm,
    data_path: Path,
    goal_model_path: Optional[Path],
    output_path: Path,
    diffusion_steps: int,
    candidate_limit: int,
    device: torch.device,
    seed: int,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_set = NGSIMPlannerDataset(data_path, split="test", norm=norm)
    seq_dataset = SequenceDataset(str(data_path), horizon=100, if_test=True)
    goal_predictor = load_goal_predictor(goal_model_path, device)
    model.eval()

    out: Dict[int, Dict[str, object]] = {}
    for scene_id in tqdm(range(min(len(test_set), len(seq_dataset))), desc="Generating DP-adapted plan_data", unit="scene"):
        item = test_set[scene_id]
        seq_batch = seq_dataset[scene_id]
        goal_x, goal_v = predict_goal_from_sequence_batch(seq_batch, goal_predictor, device)
        terminals = build_terminal_candidates_from_batch(seq_batch, goal_x, goal_v, candidate_limit)

        hist = item["hist"].unsqueeze(0).to(device)
        current = item["current_states"].unsqueeze(0).to(device)
        base_yx = item["base_yx"]
        planned = []
        for terminal_raw in terminals:
            terminal = norm.to(device).normalize_state(terminal_raw.to(device))
            route_lanes = build_route_lanes(current[0, 0], terminal).to(device)
            inputs = {
                "hist": hist,
                "current_states": current,
                "route_lanes": route_lanes,
            }
            sample_n = model.sample(inputs, steps=diffusion_steps)[0, 0].detach().cpu()
            sample = norm.denormalize_state(sample_n)
            plan = state4_to_full6(sample)
            plan[:, 0] += base_yx[0]
            plan[:, 1] += base_yx[1]
            planned.append(plan)

        target_abs = item["target_full"].detach().cpu()
        target_abs[:, 0] += base_yx[0]
        target_abs[:, 1] += base_yx[1]
        out[scene_id] = {
            "true_y": target_abs[:, 0:1],
            "true_x": target_abs[:, 1:2],
            "planned": planned,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(out, f)
    return output_path


def train_generate_dp_baseline(
    data_path: Path,
    output_dir: Path,
    goal_model_path: Optional[Path],
    epochs: int,
    batch_size: int,
    lr: float,
    diffusion_steps: int,
    sample_steps: Optional[int],
    hidden_dim: int,
    depth: int,
    heads: int,
    candidate_limit: int,
    device: torch.device,
    seed: int,
    force_train: bool = False,
    force_generate: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dp_adapted_planner.pt"
    plan_path = output_dir / "plan_data.pkl"
    sample_steps = sample_steps or diffusion_steps

    checkpoint_valid = False
    if checkpoint_path.exists() and not force_train:
        try:
            existing_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_valid = existing_ckpt.get("architecture_version") == ARCHITECTURE_VERSION
        except Exception:
            checkpoint_valid = False

    model, norm = train_dp_model(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        diffusion_steps=diffusion_steps,
        hidden_dim=hidden_dim,
        depth=depth,
        heads=heads,
        device=device,
        seed=seed,
        force=force_train,
    )
    if plan_path.exists() and not force_generate and checkpoint_valid and not force_train:
        return plan_path
    return generate_dp_plan_data(
        model=model,
        norm=norm,
        data_path=data_path,
        goal_model_path=goal_model_path,
        output_path=plan_path,
        diffusion_steps=sample_steps,
        candidate_limit=candidate_limit,
        device=device,
        seed=seed,
    )
