import os
import copy
import math
import numpy as np
import torch
import einops
import pdb
import torch.nn.functional as F

from model.diffusion import make_timesteps
from model.helpers import apply_conditioning, extract
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=50,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        writer=None,  # 新增：可选 TensorBoard Writer，与 train.py 保持同一目录
        save_last_only=True,  # 新增：仅保存最后一次模型的开关，True 时禁用中途多次保存
        use_dppo=True,
        dppo_coef=0.0,
        ppo_clip=0.2,
        dppo_gamma=0.99,
        dppo_every=10,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = writer  # 可为 None 时不写 TensorBoard
        self.save_last_only = save_last_only  # 控制是否只在训练结束保存一次
        # DPPO 超参
        self.use_dppo = use_dppo
        self.dppo_coef = float(dppo_coef)
        self.dppo_coef_max = float(dppo_coef)
        self.dppo_warmup = 2000
        self.ppo_clip = float(ppo_clip)
        self.dppo_gamma = float(dppo_gamma)
        self.dppo_every = int(dppo_every)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # ---------------- DPPO helpers ---------------- #
    def _pred_noise_from_output(self, x, t, model_out):
        diffusion = self.model
        if diffusion.predict_epsilon:
            return model_out
        coef1 = extract(diffusion.sqrt_recip_alphas_cumprod, t, x.shape)
        coef2 = extract(diffusion.sqrt_recipm1_alphas_cumprod, t, x.shape)
        return (coef1 * x - model_out) / (coef2 + 1e-8)

    def _gaussian_logprob_xprev(self, x_prev, mean, logvar):
        # log N(x_prev; mean, diag(exp(logvar))) -> (B,)
        while logvar.dim() < x_prev.dim():
            logvar = logvar.unsqueeze(-1)
        return (-0.5 * (((x_prev - mean) ** 2) / torch.exp(logvar) + logvar + math.log(2 * math.pi))
                ).flatten(1).mean(dim=1)

    @torch.no_grad()
    def _denoising_rollout_old(self, cond, old_diffusion):
        """
        用旧策略（EMA）跑一条去噪链，构造 DPPO 的 (state, action, logp_old, reward) 序列。
        state: x_t
        action: x_{t-1}
        reward: -||eps_pred - z||^2
        """
        diffusion = self.model
        device = next(old_diffusion.parameters()).device
        first_cond = next(iter(cond.values()))
        B = first_cond.shape[0]

        x_T = torch.randn(B, diffusion.horizon, diffusion.transition_dim, device=device)
        x = apply_conditioning(x_T, cond, diffusion.action_dim)

        steps = []

        for i in reversed(range(diffusion.n_timesteps)):
            t = make_timesteps(B, i, device)

            model_out = old_diffusion.model(x, cond, t)
            mean, _ = old_diffusion.mean_processor.get_mean_and_xstart(x, t, model_out)
            var, logvar = old_diffusion.var_processor.get_variance(model_out, t)
            logvar = torch.clamp(logvar, -20.0, 2.0)
            std = torch.exp(0.5 * logvar)

            z = torch.randn_like(x)
            nonzero = (t != 0).float().view(-1, *([1] * (x.dim() - 1)))
            z = z * nonzero

            x_prev = mean + std * z
            x_prev = apply_conditioning(x_prev, cond, diffusion.action_dim)

            logp_old = self._gaussian_logprob_xprev(x_prev, mean, logvar)
            eps_pred = self._pred_noise_from_output(x, t, model_out)
            r_t = -((eps_pred - z) ** 2).flatten(1).mean(dim=1)
            r_t = torch.clamp(r_t, -10.0, 0.0)

            steps.append({
                "t": t.detach(),
                "x_t": x.detach(),
                "x_prev": x_prev.detach(),
                "logp_old": logp_old.detach(),
                "reward": r_t.detach(),
            })

            x = x_prev.detach()

        return {
            "x_T": x_T.detach(),
            "steps": steps,
        }

    def _compute_adv(self, rewards):
        B, T = rewards.shape
        G = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        returns = torch.zeros_like(rewards)
        for k in reversed(range(T)):
            G = rewards[:, k] + self.dppo_gamma * G
            returns[:, k] = G
        adv = returns - returns.mean(dim=1, keepdim=True)
        adv = adv / adv.std(dim=1, keepdim=True).clamp_min(1e-3)
        return adv

    def _compute_traj_adv(self, rewards):
        """
        轨迹级优势：只用 G_0 标准化。
        rewards: (B, T)
        """
        B, T = rewards.shape
        G = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for k in reversed(range(T)):
            G = rewards[:, k] + self.dppo_gamma * G
        adv_traj = G - G.mean()
        adv_traj = adv_traj / adv_traj.std().clamp_min(1e-3)
        return adv_traj

    def _dppo_loss(self, batch, step):
        if (not self.use_dppo) or (self.dppo_coef_max <= 0):
            return None
        if (step % self.dppo_every) != 0:
            return None
        if step < self.step_start_ema:
            return None

        cond = batch.conditions
        self.ema_model.eval()
        old_diffusion = self.ema_model

        rollout = self._denoising_rollout_old(cond, old_diffusion)
        steps = rollout["steps"]

        rewards = torch.stack([s["reward"] for s in steps], dim=1)  # (B, T)
        adv_traj = self._compute_traj_adv(rewards)                  # (B,)

        diffusion = self.model

        # 旧策略整条轨迹 logprob
        logp_old_seq = torch.stack([s["logp_old"] for s in steps], dim=1)  # (B, T)
        logp_old_traj = logp_old_seq.sum(dim=1)                            # (B,)

        # 新策略 replay
        x = rollout["x_T"]
        x = apply_conditioning(x, cond, diffusion.action_dim)
        logp_new_list = []

        for k, s_k in enumerate(steps):
            t = s_k["t"]
            x_prev = s_k["x_prev"]

            model_out_new = diffusion.model(x, cond, t)
            mean_new, _ = diffusion.mean_processor.get_mean_and_xstart(x, t, model_out_new)
            _, logvar_new = diffusion.var_processor.get_variance(model_out_new, t)
            logvar_new = torch.clamp(logvar_new, -20.0, 2.0)
            logp_new = self._gaussian_logprob_xprev(x_prev, mean_new, logvar_new)
            logp_new_list.append(logp_new)

            x = x_prev.detach()

        logp_new_seq = torch.stack(logp_new_list, dim=1)  # (B, T)
        logp_new_traj = logp_new_seq.sum(dim=1)           # (B,)

        clip_log_ratio = 5.0
        log_ratio = (logp_new_traj - logp_old_traj).clamp(-clip_log_ratio, clip_log_ratio)
        ratio = torch.exp(log_ratio)
        ratio_clip = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)

        surr1 = ratio * adv_traj
        surr2 = ratio_clip * adv_traj
        ppo_loss = -torch.mean(torch.min(surr1, surr2))

        return ppo_loss

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            loss_bc_val = None
            loss_dppo_val = None
            loss_total_val = None
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)
                loss_bc, infos = self.model.loss(*batch)
                loss_dppo = self._dppo_loss(batch, self.step)
                loss_total = loss_bc
                if loss_dppo is not None:
                    coef = self.dppo_coef_max * min(
                        1.0,
                        max(0.0, (self.step - self.step_start_ema) / float(self.dppo_warmup))
                    )
                    loss_total = loss_total + coef * loss_dppo
                    loss_dppo_val = loss_dppo.item()
                loss_total_val = loss_total.item()
                loss_bc_val = loss_bc.item()

                loss_total = loss_total / self.gradient_accumulate_every
                loss_total.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # if self.step % self.save_freq == 0:  # 旧逻辑：周期性保存（已由 save_last_only 开关控制）
            #     label = self.step // self.save_freq
            #     self.save(label)
            #     # if label == 7:
            #     #     break
            if (not self.save_last_only) and self.step % self.save_freq == 0:
                label = self.step // self.save_freq
                self.save(label)  # 在 save_last_only=False 时才按周期保存

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                step_time = timer()  # 记录本次 log 周期耗时
                print_str = f'{self.step}: {loss_total_val:8.4f} | {infos_str} | '
                print_str += f'bc {loss_bc_val:8.4f}'
                if loss_dppo_val is not None:
                    print_str += f' | dppo {loss_dppo_val:8.4f}'
                print_str += f' | t: {step_time:8.4f}'
                print(print_str, flush=True)
                # 同步到 TensorBoard，与 train.py 在同一 log_dir
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', loss_total_val, self.step)
                    self.writer.add_scalar('train/loss_bc', loss_bc_val, self.step)
                    if loss_dppo_val is not None:
                        self.writer.add_scalar('train/loss_dppo', loss_dppo_val, self.step)
                    for key, val in infos.items():  # 修正缩进，确保逐项写入
                        self.writer.add_scalar(f'train/{key}', val, self.step)
                    self.writer.add_scalar('train/time_per_log', step_time, self.step)  # 记录 timer 指标

            self.step += 1

        # 仅保存最后一次模型（当 save_last_only=True 时启用）
        if self.save_last_only:
            self.save('final')  # 始终使用同名文件，自动覆盖上次训练的最终模型

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        # 确保存储目录存在，避免父目录缺失导致 torch.save 报错
        os.makedirs(self.logdir, exist_ok=True)
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
