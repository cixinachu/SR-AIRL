"""Build physically consistent, independently screened preference pairs.

No SR-AIRL reward is used to label a pair.  The negative must score worse by
collision/TTC, lane displacement, or comfort metrics after decoding all four
vehicles, integrating one ego trajectory, and re-encoding the repository's
relative-state representation.
"""
import argparse
import pickle
import random

import torch

from reward_config import (DT, HISTORY_LEN, POS_LAT_SCALE, POS_LONG_SCALE,
                           RANK_WINDOW, VEL_MIN, VEL_SCALE)


def history(sequence):
    windows = []
    for t in range(len(sequence)):
        start = max(0, t - HISTORY_LEN + 1)
        frames = [sequence[start]] * (HISTORY_LEN - (t - start + 1)) + [sequence[k] for k in range(start, t + 1)]
        windows.append(torch.cat(frames))
    return torch.stack(windows)


def decode(raw):
    """Decode [lcv, lcv-fv, nlv-lcv, olv-lcv] into physical absolute states."""
    lcv = raw[:, 0:4].clone()
    absolute = {'lcv': lcv, 'fv': lcv - raw[:, 4:8], 'nlv': lcv + raw[:, 8:12], 'olv': lcv + raw[:, 12:16]}
    for state in absolute.values():
        state[:, 0] *= POS_LAT_SCALE
        state[:, 1] *= POS_LONG_SCALE
        state[:, 2:4] = state[:, 2:4] * VEL_SCALE + VEL_MIN
    return absolute


def encode(absolute):
    """Encode physical absolutes back into the exact 16-D repository format."""
    normalized = {}
    for name, state in absolute.items():
        value = state.clone()
        value[:, 0] /= POS_LAT_SCALE
        value[:, 1] /= POS_LONG_SCALE
        value[:, 2:4] = (value[:, 2:4] - VEL_MIN) / VEL_SCALE
        normalized[name] = value
    lcv = normalized['lcv']
    return torch.cat((lcv, lcv - normalized['fv'], normalized['nlv'] - lcv, normalized['olv'] - lcv), dim=-1)


def integrate_target(absolute, role, a_long, a_lat):
    """Constant-acceleration kinematic rollout, retaining neighbor trajectories."""
    result = {name: value.clone() for name, value in absolute.items()}
    state = result[role]
    lat, lon, v_long, v_lat = (state[0, index].item() for index in range(4))
    for t in range(len(state)):
        # Semi-implicit Euler integration gives continuous positions and speeds.
        if t == 0:
            state[t] = torch.tensor((lat, lon, v_long, v_lat), dtype=state.dtype)
            continue
        v_long += a_long * DT; v_lat += a_lat * DT
        lon += v_long * DT; lat += v_lat * DT
        state[t] = torch.tensor((lat, lon, v_long, v_lat), dtype=state.dtype)
    return result


def independent_quality(absolute, role):
    """Higher is better; uses no learned reward or discriminator information."""
    target = absolute[role]
    collision, ttc_risk = 0.0, 0.0
    for name, neighbor in absolute.items():
        if name == role:
            continue
        d_long, d_lat = neighbor[:, 1] - target[:, 1], (neighbor[:, 0] - target[:, 0]).abs()
        collision += ((d_long.abs() < 4.5) & (d_lat < 1.2)).float().sum().item()
        closing = target[:, 2] - neighbor[:, 2]
        valid_ttc = (d_long > 0.0) & (d_lat < 2.5) & (closing > 0.1)
        ttc = torch.where(valid_ttc, d_long / closing.clamp_min(0.1), torch.full_like(d_long, 99.0))
        ttc_risk += torch.exp(-ttc / 2.0).sum().item()
    acceleration = torch.diff(target[:, 2:4], dim=0, prepend=target[:1, 2:4]) / DT
    jerk = torch.diff(acceleration, dim=0, prepend=acceleration[:1]) / DT
    comfort = acceleration.square().sum(1).mean().item() + 0.05 * jerk.square().sum(1).mean().item()
    lane_offset = (target[:, 0] - target[0, 0]).abs().max().item()
    risk = 100.0 * collision + 20.0 * ttc_risk + comfort + 2.0 * lane_offset
    return -risk, {'collision_frames': collision, 'ttc_risk': ttc_risk, 'comfort': comfort, 'lane_offset': lane_offset}


def make_negative(positive_raw, role):
    """Return a screened negative, or None when no perturbation is worse."""
    expert_absolute = decode(positive_raw)
    positive_quality, _ = independent_quality(expert_absolute, role)
    # Longitudinal acceleration can create closing risk; two lateral directions
    # cover the two possible neighboring lanes.  Braking alone is never labeled
    # negative merely because it differs from the expert.
    candidates = [(3.5, 0.0), (2.5, 2.5), (2.5, -2.5), (0.0, 4.0), (0.0, -4.0)]
    best = None
    for a_long, a_lat in candidates:
        candidate = integrate_target(expert_absolute, role, a_long, a_lat)
        quality, metrics = independent_quality(candidate, role)
        if quality < positive_quality - 1.0 and (best is None or quality < best[0]):
            best = quality, encode(candidate), metrics
    if best is None:
        return None
    return best[1], positive_quality, best[0], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_path', required=True)
    parser.add_argument('--output', default='counterfactual_pairs.pt')
    parser.add_argument('--clips_per_episode', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(); random.seed(args.seed)
    with open(args.expert_path, 'rb') as file:
        data = pickle.load(file)
    states, dones = data['state'].float(), data['dones'].reshape(-1).bool()
    result = {role: {'positive': [], 'negative': [], 'quality_positive': [], 'quality_negative': [], 'metrics': []} for role in ('lcv', 'fv')}
    begin = 0
    for episode_index, end in enumerate(torch.where(dones)[0].tolist()):
        if episode_index >= 100:  # hard training-split boundary for legacy 150-scene files
            break
        episode = states[begin:end + 1]; begin = end + 1
        if len(episode) < RANK_WINDOW:
            continue
        count = min(args.clips_per_episode, len(episode) - RANK_WINDOW + 1)
        for start in random.sample(range(len(episode) - RANK_WINDOW + 1), count):
            positive_raw = episode[start:start + RANK_WINDOW]
            for role in result:
                screened = make_negative(positive_raw, role)
                if screened is None:
                    continue
                negative_raw, positive_quality, negative_quality, metrics = screened
                result[role]['positive'].append(history(positive_raw))
                result[role]['negative'].append(history(negative_raw))
                result[role]['quality_positive'].append(positive_quality)
                result[role]['quality_negative'].append(negative_quality)
                result[role]['metrics'].append(metrics)
    for role, values in result.items():
        if not values['positive']:
            raise RuntimeError(f'No independently worse {role} counterfactuals passed screening.')
        values['positive'] = torch.stack(values['positive'])
        values['negative'] = torch.stack(values['negative'])
        values['quality_positive'] = torch.tensor(values['quality_positive'])
        values['quality_negative'] = torch.tensor(values['quality_negative'])
    torch.save(result, args.output)
    print({role: values['positive'].shape for role, values in result.items()}, '->', args.output)


if __name__ == '__main__':
    main()

