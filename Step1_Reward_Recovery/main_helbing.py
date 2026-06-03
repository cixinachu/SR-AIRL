import argparse
import glob
import torch
from torch.optim import Adam
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.serialization import add_safe_globals
from ppo_model.PPO import PPO, RolloutBuffer
from NGSIM_env.envs.env import NGSIMEnv
import numpy as np
import warnings
import os
from datetime import datetime
from collections import deque


from airl import AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet

warnings.filterwarnings('ignore')

def run  
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_dir", type=str, default=None   无,
                        help="继续训练的日志目录（包含 checkpoints 子目录）")
    parser.add_argument("--start_episode", type=int, default=None   无,
                        help="手动指定起始 episode（如无 state 文件时使用）")
    parser.add_argument("--start_step", type=int, default=None   无,
                        help="手动指定起始 time_step（如无 state 文件时使用）")
    args = parser.parse_args()

    add_safe_globals([PPO, RolloutBuffer])

    script_dir = os.path.dirname(os.path.abspath(__file__))

    #True: 启用共享社会势场
    USE_SOCIAL_POTENTIAL = True
    #False 保持单帧输入
    USE_HISTORY = True
    HISTORY_LEN = 4 

    # 正则项权重
    LAMBDA_SOC = 0.5 
    LAMBDA_EQ = 0.01  

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = "SocialAIRL" if USE_SOCIAL_POTENTIAL else "OriginalAIRL"
    base_log_dir = os.path   路径.join(script_dir, "runs", "log2","reward_recovery_new")
    if args.resume_dir is not None:
        log_dir = args.resume_dir
    else :
        run_name = f"main2_{exp_name}_{time_str}"
        log_dir = os.path   路径.join(base_log_dir, run_name)
    writer = SummaryWriter(log_dir=log_dir) 
    checkpoints_dir = os.path   路径.join(log_dir, "checkpoints")
    expert_data_path = os.path   路径.join   加入(log_dir, "expert_data.pkl")
    os.makedirs(checkpoints_dir, exist_ok=True) 
    print(f">>> TensorBoard Logging to: {log_dir}")
    # ==========================================================

    env = NGSIMEnv(scene='us-101')
    if not os.path.exists(expert_data_path):
        env.generate_experts(save_path=expert_data_path)

    STATE_DIM = 16 * (HISTORY_LEN if USE_HISTORY else 1)

    def init_history(obs):
        hist = deque(maxlen=HISTORY_LEN)
        for _ in range(HISTORY_LEN):
            hist.append(obs.copy())
        return hist

    def stack_history(hist):
        if not USE_HISTORY:
            return hist[-1]
        return np.concatenate(list(hist), axis=0)

    buffer_exp = RolloutBuffer()
    try:
        buffer_exp.add_exp(path=expert_data_path, use_history=USE_HISTORY, history_len=HISTORY_LEN)
    except TypeError:
        buffer_exp.add_exp(path=expert_data_path)

    model_lcv = PPO   进口argparse(state_dim=STATE_DIM, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                    has_continuous_action_space=True, action_std_init=0.2)
    model_fv = PPO(state_dim=STATE_DIM, action_dim=2, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, K_epochs=80, eps_clip=0.2,
                   has_continuous_action_space=True, action_std_init=0.2)

    social_phi_net = None

    if USE_SOCIAL_POTENTIAL:
        print(">>> Training Mode: Social Potential AIRL (Shared Framework)")
        social_phi_net = SocialPotentialNet(state_shape=STATE_DIM).to('cuda')

        disc_lcv = SocialAIRLDiscrim(state_shape=STATE_DIM, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')
        disc_fv = SocialAIRLDiscrim(state_shape=STATE_DIM, gamma=0.99, shared_phi_net=social_phi_net).to('cuda')

        # optim_disc = Adam([
        #     {'params': disc_lcv.parameters()},
        #     {'params': disc_fv.parameters()},
        #     {'params': social_phi_net.parameters()}
        # ], lr=3e-4)

        optim_disc = Adam([
            {'params': disc_lcv.private_g.parameters()},  # LCV 私有奖励网络
            {'params': disc_lcv.h.parameters()},          # LCV 值函数网络
            {'params': [disc_lcv.alpha]},                 # LCV 势场权重
            {'params': disc_fv.private_g.parameters()},   # FV 私有奖励网络
            {'params': disc_fv.h.parameters()},           # FV 值函数网络
            {'params': [disc_fv.alpha]},                  # FV 势场权重
            {'params': social_phi_net.parameters()}       # 共享社会势场，仅加入一次
        ], lr=3e-4)

    else: 
        print(">>> Training Mode: Original MA-AIRL (Independent Framework)")

        disc_lcv = AIRLDiscrim(state_shape=STATE_DIM, gamma=0.99).to('cuda')
        disc_fv = AIRLDiscrim(state_shape=STATE_DIM, gamma=0.99).to('cuda')

        optim_disc = Adam([
            {'params': disc_lcv.parameters()},
            {'params': disc_fv.parameters()}
        ], lr=3e-4)   ), lr = 3)

    print(f">>> torch.cuda.is_available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f">>> GPU[0]: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log_device(name, module):
        try:
            dev = next(module.parameters()).device
        except StopIteration:
            dev = "no-params"
        print(f">>> Device[{name}]: {dev}")

    log_device("ppo_lcv_policy", model_lcv.policy)
    log_device("ppo_fv_policy", model_fv.policy)
    log_device("disc_lcv", disc_lcv)
    log_device("disc_fv", disc_fv)
    if USE_SOCIAL_POTENTIAL and social_phi_net is not None:
        log_device("social_phi_net", social_phi_net)

    disc_criterion = nn.BCELoss()

    def _buffer_to_device(buf, dev):
        if buf is None:
            return buf
        buf.states = [t.to(dev) for t in buf.states]
        buf.actions = [t.to(dev) for t in buf.actions]
        buf.logprobs = [t.to(dev) for t in buf.logprobs]
        buf.state_values = [t.to(dev) for t in buf.state_values]
        buf.next_states = [t.to(dev) for t in buf.next_states]
        buf.rewards = [torch.as_tensor(r, device=dev) for r in buf.rewards]
        buf.is_terminals = [torch.as_tensor(d, device=dev) for d in buf.is_terminals]
        return buf

    time_step = 0
    i_episode = 0
    scene_id = 0
    state_path = os.path.join(checkpoints_dir, "train_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu")
        i_episode = state.get("i_episode", i_episode)
        time_step = state.get("time_step", time_step)
        scene_id = state.get("scene_id", 0)
        print(f">>> Resumed training state from {state_path}: episode={i_episode}, time_step={time_step}, scene_id={scene_id}")
    else: 
        if args.resume_dir is not None and args.start_episode is None:
            events = sorted(glob.glob(os.path.join(log_dir, "events.*")))
            if events:
                try:
                    ea = EventAccumulator(events[-1]); ea.Reload()
                    scalars = ea.Scalars('disc/loss')
                    if scalars:
                        last_step = scalars[-1].step
                        i_episode = max(i_episode, last_step)
                        print(f">>> Recovered i_episode from events: {last_step} (use {i_episode})")
                except Exception as e:
                    print(f"Warn: failed to read events for last step: {e}")
        if args.start_episode is not None:
            i_episode = args.start_episode
        if args.start_step is not None:
            time_step = args.start_step

    if args.resume_dir is not None:
        try: 
            lcv_path = os.path.join(checkpoints_dir, 'model_lcv.pt')
            fv_path = os.path.join(checkpoints_dir, 'model_fv.pt')
            disc_lcv_path = os.path.join(checkpoints_dir, 'disc_lcv.pt')
            disc_fv_path = os.path.join(checkpoints_dir, 'disc_fv.pt')
            if os.path.exists(lcv_path):
                model_lcv = torch.load(lcv_path, map_location=device, weights_only=False)
                print(f"Loaded model_lcv from {lcv_path}")
            if os.path.exists(fv_path):
                model_fv = torch.load(fv_path, map_location=device, weights_only=False)
                print(f"Loaded model_fv from {fv_path}")
            if os.path.exists(disc_lcv_path):
                disc_lcv = torch.load(disc_lcv_path, map_location=device, weights_only=False)
                print(f"Loaded disc_lcv from {disc_lcv_path}")
            if os.path.exists(disc_fv_path):
                disc_fv = torch.load(disc_fv_path, map_location=device, weights_only=False)
                print(f"Loaded disc_fv from {disc_fv_path}")
            if USE_SOCIAL_POTENTIAL and social_phi_net is not None:
                phi_path = os.path.join(checkpoints_dir, 'social_phi.pt')
                if os.path.exists(phi_path):
                    social_phi_net = torch.load(phi_path, map_location=device, weights_only=False)
                    print(f"Loaded social_phi_net from {phi_path}")
            buf_lcv = os.path.join(checkpoints_dir, 'lcv_buffer.pt')
            buf_fv = os.path.join(checkpoints_dir, 'fv_buffer.pt')
            if os.path.exists(buf_lcv):
                model_lcv.buffer = torch.load(buf_lcv, map_location='cpu', weights_only=False)
                model_lcv.buffer = _buffer_to_device(model_lcv.buffer, device)
                print(f"Loaded lcv_buffer from {buf_lcv}")
            if os.path.exists(buf_fv):
                model_fv.buffer = torch.load(buf_fv, map_location='cpu', weights_only=False)
                model_fv.buffer = _buffer_to_device(model_fv.buffer, device)
                print(f"Loaded fv_buffer from {buf_fv}")
        except Exception as e:
            print(f"Warn: failed to load resume checkpoints: {e}")

    while i_episode <= 110000:
        obs = env.reset(scene_id=scene_id)
        hist_buf = init_history(obs)  # 历史缓冲（若关闭历史，则内部只重复当前帧）
        state = stack_history(hist_buf)
        current_ep_reward = 0

        while True:
            action_lcv = model_lcv.select_action(state) * 5
            action_fv = model_fv.select_action(state) * 5
            action = np.concatenate([action_lcv, action_fv])
            next_obs, reward, done, _ = env.step(action)

            hist_buf.append(next_obs)
            next_state = stack_history(hist_buf)

            model_lcv.buffer.rewards.append(reward)
            model_lcv.buffer.is_terminals.append(done)
            model_lcv.buffer.next_states.append(torch.from_numpy(next_state.reshape(1, -1)).to('cuda'))

            model_fv.buffer.rewards.append(reward)
            model_fv.buffer.is_terminals.append(done)
            model_fv.buffer.next_states.append(torch.from_numpy(next_state.reshape(1, -1)).to('cuda'))

            time_step += 1
            current_ep_reward += reward
            state = next_state 

            if time_step % int(1e5) == 0:
                model_lcv.decay_action_std(0.05, 0.1)
                model_fv.decay_action_std(0.05, 0.1)
                writer.add_scalar('actor/action_std_lcv', model_lcv.action_std, time_step)
                writer.add_scalar('actor/action_std_fv', model_fv.action_std, time_step)
                if model_lcv.action_std == 0.1 or model_fv.action_std == 0.1:
                    print(f"[ActionStd] Reached min_action_std=0.1 | LCV: {model_lcv.action_std}, FV: {model_fv.action_std}")

            if done:
                break

        writer.add_scalar('Generator/Episode_Reward', current_ep_reward, i_episode)


        if i_episode % 150 == 0:
            epoch_loss_total = []
            epoch_loss_base = []
            epoch_loss_soc = []
            epoch_loss_eq = []
            acc_exp_lcv = [] 
            acc_exp_fv = []  
            acc_pi_lcv = []  
            acc_pi_fv = []   

            batch_size = 64
            for _ in range(10):
                states_lcv, actions_lcv, _, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.sample(batch_size)
                states_fv, actions_fv, _, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.sample(batch_size)
                states_exp, actions_exp, _, dones_exp, next_states_exp = buffer_exp.sample(batch_size, if_exp=True)
                states_exp = states_exp.float()
                actions_exp = actions_exp.float() / 5
                next_states_exp = next_states_exp.float()
                states_lcv = states_lcv.float()
                states_fv = states_fv.float()
                next_states_lcv = next_states_lcv.float()
                next_states_fv = next_states_fv.float()
                dones_lcv = dones_lcv.int().to('cuda')
                dones_fv = dones_fv.int().to('cuda')
                dones_exp = dones_exp.int().to('cuda')

                def check_probs(name, prob):
                    if torch.isnan(prob).any() or torch.isinf(prob).any():
                        raise RuntimeError(f"{name} contains NaN/Inf, min={prob.min().item()}, max={prob.max().item()}")

                def has_invalid(name, tensor):
                    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                        print(f"[Skip batch] {name} has NaN/Inf")
                        return True
                    return False

                if any([
                    has_invalid("states_lcv", states_lcv),
                    has_invalid("states_fv", states_fv),
                    has_invalid("states_exp", states_exp),
                    has_invalid("next_states_lcv", next_states_lcv),
                    has_invalid("next_states_fv", next_states_fv),
                    has_invalid("next_states_exp", next_states_exp),
                ]):
                    continue

                with torch.no_grad():
                    log_pis_exp_lcv, _, _ = model_lcv.policy.evaluate   评估(states_exp, actions_exp[:, :2])
                    log_pis_exp_fv, _, _ = model_fv.policy.evaluate(states_exp, actions_exp[:, 2:])
                    if has_invalid("log_pis_exp_lcv", log_pis_exp_lcv) or has_invalid("log_pis_exp_fv", log_pis_exp_fv):
                        continue

                prob_pi_lcv = disc_lcv(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
                prob_exp_lcv = disc_lcv(states_exp, dones_exp.squeeze(-1), log_pis_exp_lcv, next_states_exp)

                prob_pi_fv = disc_fv(states_fv, dones_fv, log_pis_fv, next_states_fv)
                prob_exp_fv = disc_fv(states_exp, dones_exp.squeeze(-1), log_pis_exp_fv, next_states_exp)

                prob_pi_lcv = torch.clamp(prob_pi_lcv, 0.05, 0.95)
                prob_exp_lcv = torch.clamp(prob_exp_lcv, 0.05, 0.95)
                prob_pi_fv = torch.clamp(prob_pi_fv, 0.05, 0.95)
                prob_exp_fv = torch.clamp(prob_exp_fv, 0.05, 0.95)

                check_probs("prob_exp_lcv", prob_exp_lcv)
                check_probs("prob_pi_lcv", prob_pi_lcv)
                check_probs("prob_exp_fv", prob_exp_fv)
                check_probs("prob_pi_fv", prob_pi_fv)

                # prob_exp_lcv = torch.clamp(prob_exp_lcv, 1e-6, 1 - 1e-6)
                # prob_pi_lcv = torch.clamp(prob_pi_lcv, 1e-6, 1 - 1e-6)
                # prob_exp_fv = torch.clamp(prob_exp_fv, 1e-6, 1 - 1e-6)
                # prob_pi_fv = torch.clamp(prob_pi_fv, 1e-6, 1 - 1e-6)

                expert_acc_lcv = ((prob_exp_lcv > 0.5).float()).mean().detach().cpu()
                expert_acc_fv = ((prob_exp_fv > 0.5).float()).mean().detach().cpu()
                learner_acc_lcv = ((prob_pi_lcv < 0.5).float()).mean().detach().cpu()
                learner_acc_fv = ((prob_pi_fv < 0.5).float()).mean().detach().cpu()

                expert_target = torch.full_like(prob_exp_lcv, 0.9, device=prob_exp_lcv.device)
                agent_target = torch.full_like(prob_pi_lcv, 0.1, device=prob_pi_lcv.device)

                #Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
                expert_loss = disc_criterion(prob_exp_lcv, expert_target) + \
                              disc_criterion(prob_exp_fv, expert_target)
                agent_loss = disc_criterion(prob_pi_lcv, agent_target) + \
                             disc_criterion(prob_pi_fv, agent_target)
            
                loss_base = (expert_loss + agent_loss)

                loss_soc = torch.tensor(0.0).to('cuda')
                loss_eq = torch.tensor(0.0).to('cuda')

                if USE_SOCIAL_POTENTIAL:
                    _, _, eps_lcv_exp = disc_lcv.get_reward(states_exp)
                    _, _, eps_fv_exp = disc_fv.get_reward(states_exp)
                    loss_soc = torch.mean(eps_lcv_exp ** 2) + torch.mean(eps_fv_exp ** 2)

                    adv_lcv_exp = torch.clamp(disc_lcv.f(states_exp, dones_exp.squeeze(-1), next_states_exp), -5.0, 5.0)
                    adv_fv_exp = torch.clamp(disc_fv.f(states_exp, dones_exp.squeeze(-1), next_states_exp), -5.0, 5.0)
                    loss_eq = torch.mean(adv_lcv_exp ** 2) + torch.mean(adv_fv_exp ** 2)

                    loss_disc = loss_base + (LAMBDA_SOC * loss_soc) + (LAMBDA_EQ * loss_eq)
                else:
                    loss_disc = loss_base

                optim_disc.zero_grad()
                loss_disc.backward()
                optim_disc.step()

                epoch_loss_total.append(loss_disc.item())
                epoch_loss_base.append(loss_base.item())
                if USE_SOCIAL_POTENTIAL:   如果USE_SOCIAL_POTENTIAL:
                    epoch_loss_soc.append(loss_soc.item())
                    epoch_loss_eq.append(loss_eq.item())
                acc_exp_lcv.append(expert_acc_lcv)
                acc_exp_fv.append(expert_acc_fv)
                acc_pi_lcv.append(learner_acc_lcv)
                acc_pi_fv.append(learner_acc_fv)

            writer.add_scalar('Discriminator/Total_Loss', np.mean(epoch_loss_total), i_episode)
            writer.add_scalar('Discriminator/Base_Loss', np.mean(epoch_loss_base), i_episode)
            if USE_SOCIAL_POTENTIAL:
                writer.add_scalar('Discriminator/Social_Loss', np.mean(epoch_loss_soc), i_episode)
                writer.add_scalar('Discriminator/Equilibrium_Loss', np.mean(epoch_loss_eq), i_episode)

            states_lcv, actions_lcv, dones_lcv, log_pis_lcv, next_states_lcv = model_lcv.buffer.get()
            states_fv, actions_fv, dones_fv, log_pis_fv, next_states_fv = model_fv.buffer.get()
            torch.save(model_lcv.buffer, os.path.join(checkpoints_dir, 'lcv_buffer.pt'))
            torch.save(model_fv.buffer, os.path.join(checkpoints_dir, 'fv_buffer.pt'))
            next_states_lcv = next_states_lcv.float()
            next_states_fv = next_states_fv.float()
            dones_lcv = dones_lcv.int().to('cuda')
            dones_fv = dones_fv.int().to('cuda')

            rewards_lcv = disc_lcv.calculate_reward(states_lcv, dones_lcv, log_pis_lcv, next_states_lcv)
            rewards_fv = disc_fv.calculate_reward(states_fv, dones_fv, log_pis_fv, next_states_fv)

            writer.add_scalar('Recovered_Reward/LCV_Mean', rewards_lcv.mean().item(), i_episode)
            writer.add_scalar('Recovered_Reward/FV_Mean', rewards_fv.mean().item(), i_episode)

            rewards_lcv = torch.clamp(rewards_lcv, -10.0, 10.0)
            rewards_fv = torch.clamp(rewards_fv, -10.0, 10.0)

            model_lcv.buffer.rewards = []
            model_fv.buffer.rewards = []
            for sub_r in rewards_lcv:
                model_lcv.buffer.rewards.append(sub_r)
            for sub_r in rewards_fv:
                model_fv.buffer.rewards.append(sub_r)

            disc_loss_mean = float(np.mean(epoch_loss_total)) if len(epoch_loss_total) > 0 else 0.0
            acc_exp_lcv_mean = float(np.mean(acc_exp_lcv)) if len(acc_exp_lcv) > 0 else 0.0
            acc_pi_lcv_mean = float(np.mean(acc_pi_lcv)) if len(acc_pi_lcv) > 0 else 0.0
            acc_exp_fv_mean = float(np.mean(acc_exp_fv)) if len(acc_exp_fv) > 0 else 0.0
            acc_pi_fv_mean = float(np.mean(acc_pi_fv)) if len(acc_pi_fv) > 0 else 0.0

            # print(f'Episode {i_episode}, Disc Loss: {np.mean(epoch_loss_total):.4f}') 
            print('Epoch disc loss {:.4}, acc exp lcv {:.4}, acc pi lcv {:.4},acc exp fv {:.4}, acc pi fv {:.4}'
                  .format(disc_loss_mean, acc_exp_lcv_mean, acc_pi_lcv_mean, acc_exp_fv_mean, acc_pi_fv_mean))

            writer.add_scalar('disc/loss', disc_loss_mean, i_episode)
            writer.add_scalar('disc/acc_exp_lcv', acc_exp_lcv_mean, i_episode)
            writer.add_scalar('disc/acc_pi_lcv', acc_pi_lcv_mean, i_episode)
            writer.add_scalar('disc/acc_exp_fv', acc_exp_fv_mean, i_episode)
            writer.add_scalar('disc/acc_pi_fv', acc_pi_fv_mean, i_episode)

            model_lcv.update()
            model_fv.update()

            print('Episode {}, reward {:.4}, disc r lcv {:.4}, disc r fv {:.4}'
                  .format(i_episode, current_ep_reward, rewards_lcv.mean(), rewards_fv.mean()))
            writer.add_scalar('disc/reward_lcv', rewards_lcv.mean().item(), i_episode)
            writer.add_scalar('disc/reward_fv', rewards_fv.mean().item(), i_episode)

        i_episode += 1
        scene_id += 1

        if scene_id > 150:
            scene_id = 0
            torch.save(disc_lcv, os.path.join(checkpoints_dir, 'disc_lcv.pt'))
            torch.save(disc_fv, os.path.join(checkpoints_dir, 'disc_fv.pt'))

            if USE_SOCIAL_POTENTIAL and social_phi_net is not None:如果USE_SOCIAL_POTENTIAL和social_phi_net不是None：
                torch.save(social_phi_net, os.path.join(checkpoints_dir, 'social_phi.pt'))
                print("Saved social_phi.pt")

            torch.save(model_lcv, os.path.join(checkpoints_dir, 'model_lcv.pt'))
            torch.save(model_fv, os.path.join(checkpoints_dir, 'model_fv.pt'))
            torch.save({
                "i_episode": i_episode,
                "time_step": time_step,
                "scene_id": scene_id
            }, state_path)
            print(f"Models saved at episode {i_episode}")

    writer.close()
    env.close()


if __name__ == '__main__':
    run()
