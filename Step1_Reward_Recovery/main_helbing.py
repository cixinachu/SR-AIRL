"""CUDA training entry point for structured SR-AIRL reward recovery."""
import argparse
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from NGSIM_env.envs.env import NGSIMEnv
from airl import AnisotropicInteractionPotential, SocialAIRLDiscrim
from ppo_model.PPO import PPO, RolloutBuffer
from reward_config import (DIR_LAMBDA, FRAME_STATE_DIM, GAMMA, HISTORY_LEN,
                           L_LAT, L_LONG, MAX_EPISODES, RANK_BATCH_SIZE,
                           RANK_LAMBDA, RANK_TEMPERATURE, SEED)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stacked(history):
    return np.concatenate(tuple(history), axis=0)


def trajectory_reward(disc, states):
    return disc.get_reward(states.reshape(-1, states.shape[-1]))[0].reshape(states.shape[0], -1).mean(1)


def pair_loss(disc, pairs, device):
    count = min(RANK_BATCH_SIZE, len(pairs['positive']))
    index = torch.randint(len(pairs['positive']), (count,))
    positive, negative = pairs['positive'][index].to(device), pairs['negative'][index].to(device)
    margin = trajectory_reward(disc, positive) - trajectory_reward(disc, negative)
    return -F.logsigmoid(margin / RANK_TEMPERATURE).mean(), margin.mean()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_path', required=True, help='expert pickle from scenes 0..99')
    parser.add_argument('--preference_data', default=None)
    parser.add_argument('--max_episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--lambda_rank', type=float, default=RANK_LAMBDA)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_dir', default=None)
    args = parser.parse_args()
    if not os.path.isfile(args.expert_path): parser.error(f'expert data not found: {args.expert_path}')
    if not torch.cuda.is_available(): raise RuntimeError('SR-AIRL training requires CUDA.')
    set_seed(args.seed); device = torch.device('cuda'); state_dim = FRAME_STATE_DIM * HISTORY_LEN
    log_dir = args.log_dir or os.path.join(os.path.dirname(__file__), 'runs', 'sr_airl', datetime.now().strftime('%Y%m%d-%H%M%S'))
    checkpoints = os.path.join(log_dir, 'checkpoints'); os.makedirs(checkpoints, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f'CUDA: {torch.cuda.get_device_name(device)}\nTensorBoard: {log_dir}')

    env = NGSIMEnv(scene='us-101')
    expert = RolloutBuffer(); expert.add_exp(args.expert_path, use_history=True, history_len=HISTORY_LEN)
    if expert.states.shape[-1] != state_dim: raise ValueError('expert data has incompatible state dimension')
    shared = AnisotropicInteractionPotential(L_LONG, L_LAT, DIR_LAMBDA).to(device)
    disc_lcv = SocialAIRLDiscrim(state_dim, GAMMA, shared, 'lcv', HISTORY_LEN).to(device)
    disc_fv = SocialAIRLDiscrim(state_dim, GAMMA, shared, 'fv', HISTORY_LEN).to(device)
    groups = [disc_lcv.private_g.parameters(), disc_lcv.h.parameters(), [disc_lcv.alpha_raw],
              disc_fv.private_g.parameters(), disc_fv.h.parameters(), [disc_fv.alpha_raw], shared.parameters()]
    parameters = [p for g in groups for p in g]
    if len({id(p) for p in parameters}) != len(parameters): raise RuntimeError('duplicate optimizer parameter')
    optimizer = Adam(parameters, lr=3e-4)
    model_lcv = PPO(state_dim, 2, 3e-4, 1e-3, GAMMA, 80, .2, True, .2)
    model_fv = PPO(state_dim, 2, 3e-4, 1e-3, GAMMA, 80, .2, True, .2)
    pairs = torch.load(args.preference_data, map_location='cpu', weights_only=True) if args.preference_data else None
    if pairs is not None and not {'lcv', 'fv'}.issubset(pairs): raise ValueError('preference data must contain lcv and fv')

    def save(episode, final=False):
        package = {'episode': episode, 'seed': args.seed, 'history_len': HISTORY_LEN,
                   'disc_lcv': disc_lcv.state_dict(), 'disc_fv': disc_fv.state_dict(),
                   'interaction': shared.state_dict(), 'optimizer': optimizer.state_dict(),
                   'model_lcv': model_lcv.policy.state_dict(), 'model_fv': model_fv.policy.state_dict()}
        torch.save(package, os.path.join(checkpoints, 'train_state.pt'))
        torch.save(disc_lcv.state_dict(), os.path.join(checkpoints, 'disc_lcv.pt'))
        torch.save(disc_fv.state_dict(), os.path.join(checkpoints, 'disc_fv.pt'))
        torch.save(shared.state_dict(), os.path.join(checkpoints, 'interaction.pt'))
        if final: print(f'Saved final checkpoint: {checkpoints}')

    for episode in range(1, args.max_episodes + 1):
        obs = env.reset(scene_id=(episode - 1) % 100)
        history = deque([obs.copy() for _ in range(HISTORY_LEN)], maxlen=HISTORY_LEN)
        done = False; env_return = 0.
        while not done:
            state = stacked(history)
            action = np.r_[model_lcv.select_action(state) * 5, model_fv.select_action(state) * 5]
            next_obs, reward, done, _ = env.step(action)
            history.append(next_obs); next_state = stacked(history)
            for model in (model_lcv, model_fv):
                model.buffer.rewards.append(reward); model.buffer.is_terminals.append(done)
                model.buffer.next_states.append(torch.from_numpy(next_state[None]).to(device))
            env_return += reward
        writer.add_scalar('Generator/Episode_Reward', env_return, episode)
        if episode % 150:
            if episode % args.save_every == 0: save(episode)
            continue

        aggregate = {'loss': [], 'rank': [], 'margin': [], 'expert_acc': [], 'policy_acc': []}
        for _ in range(10):
            sl, al, _, dl, logl, nsl = model_lcv.buffer.sample(64)
            sf, af, _, df, logf, nsf = model_fv.buffer.sample(64)
            se, ae, _, de, nse = expert.sample(64, if_exp=True)
            sl, sf, nsl, nsf, se, nse = [v.float().to(device) for v in (sl, sf, nsl, nsf, se, nse)]
            dl, df, de = [v.reshape(-1).float().to(device) for v in (dl, df, de)]
            with torch.no_grad():
                epl, _, _ = model_lcv.policy.evaluate(se, ae.float().to(device)[:, :2] / 5)
                epf, _, _ = model_fv.policy.evaluate(se, ae.float().to(device)[:, 2:] / 5)
            lel = disc_lcv.discriminator_logits(se, de, epl, nse); lef = disc_fv.discriminator_logits(se, de, epf, nse)
            lpl = disc_lcv.discriminator_logits(sl, dl, logl.to(device), nsl); lpf = disc_fv.discriminator_logits(sf, df, logf.to(device), nsf)
            base = sum((F.binary_cross_entropy_with_logits(x, torch.ones_like(x)) for x in (lel, lef)), torch.zeros((), device=device))
            base += sum((F.binary_cross_entropy_with_logits(x, torch.zeros_like(x)) for x in (lpl, lpf)), torch.zeros((), device=device))
            rank = torch.zeros((), device=device); margin = torch.zeros((), device=device)
            if pairs is not None and args.lambda_rank:
                rl, ml = pair_loss(disc_lcv, pairs['lcv'], device); rf, mf = pair_loss(disc_fv, pairs['fv'], device)
                rank, margin = rl + rf, (ml + mf) / 2
            loss = base + args.lambda_rank * rank
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            aggregate['loss'].append(loss.item()); aggregate['rank'].append(rank.item()); aggregate['margin'].append(margin.item())
            aggregate['expert_acc'].append(((lel > 0).float().mean() + (lef > 0).float().mean()).item() / 2)
            aggregate['policy_acc'].append(((lpl < 0).float().mean() + (lpf < 0).float().mean()).item() / 2)
        for key, values in aggregate.items(): writer.add_scalar(f'Discriminator/{key}', np.mean(values), episode)

        for disc, model in ((disc_lcv, model_lcv), (disc_fv, model_fv)):
            states, actions, dones, logpis, next_states = model.buffer.get()
            states, dones, logpis, next_states = states.float().to(device), dones.float().to(device), logpis.float().to(device), next_states.float().to(device)
            signal = disc.calculate_policy_reward(states, dones, logpis, next_states).clamp(-10, 10)
            total, private, phi = disc.get_reward(states)
            name = disc.role.upper()
            writer.add_scalar(f'Recovered_Reward/{name}_total', total.mean().item(), episode)
            writer.add_scalar(f'Recovered_Reward/{name}_private', private.mean().item(), episode)
            writer.add_scalar(f'Interaction/{name}_phi', phi.mean().item(), episode)
            model.buffer.rewards = [r.detach() for r in signal]
            model.update()
        for name, value in {'alpha_lcv': disc_lcv.alpha, 'alpha_fv': disc_fv.alpha, 'L_long': shared.L_long, 'L_lat': shared.L_lat, 'dir_lambda': shared.dir_lambda}.items(): writer.add_scalar(f'Interaction/{name}', value.item(), episode)
        if episode % args.save_every == 0: save(episode)
    save(args.max_episodes, final=True); writer.close(); env.close()


if __name__ == '__main__':
    run()

