import argparse
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import NoisyReward
from model.ddpm import GaussianDiffusionTrainer
import copy
from tqdm import tqdm
from model.related_module import *
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent
# 确保本地 utils 可被导入（避免相对路径运行时报 ModuleNotFoundError）
sys.path.append(str(base_dir))
from utils.dataset import data_loader
import random
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval(args, loader, trainer, model, fv_reward):
    model.eval()
    device = args.device
    loss_list = []
    for batch in loader:
        lcv_traj = batch[0].cpu().numpy()
        fv_traj = batch[1].cpu().numpy()
        nlv_traj = batch[2].cpu().numpy()
        olv_traj = batch[3].cpu().numpy()

        # Generate true reward
        n_lcv = normalize_obs(copy.deepcopy(lcv_traj))
        n_fv = normalize_obs(copy.deepcopy(fv_traj))
        n_nlv = normalize_obs(copy.deepcopy(nlv_traj))
        n_olv = normalize_obs(copy.deepcopy(olv_traj))

        dfv = n_lcv - n_fv
        dnlv = n_nlv - n_lcv
        dolv = n_olv - n_lcv

        state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=-1)).to(device).float()
        true_reward = fv_reward(state)

        # Noisy data
        noisy_fv, t = trainer(torch.from_numpy(n_fv).to(device))
        noisy_dfv = n_lcv - noisy_fv.cpu().numpy()
        noisy_state = torch.from_numpy(np.concatenate([n_lcv, noisy_dfv, dnlv, dolv], axis=-1)).to(device).float()
        pred_reward = model(noisy_state, t)

        loss = F.mse_loss(pred_reward, true_reward)
        loss_list.append(loss.item())

    model.train()

    return np.mean(loss_list)


def train(args):
    device = args.device

    model = NoisyReward(args.context_dim, args.T)
    ema_model = copy.deepcopy(model)
    guide = None

    # show model size
    model_size = 0
    for param in ema_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    trainer = GaussianDiffusionTrainer(model, beta_1=args.beta_1, beta_T=args.beta_T, T=args.T).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # dataset
    path = ''
    val_path = ''

    train_loader = data_loader(args.obs_len, args.pred_len, args.skip, args.batch_size, path,
                               save_path='data/sample_data.pkl', if_test='train')
    test_loader = data_loader(args.obs_len, args.pred_len_val, args.skip, args.batch_size * 4, val_path,
                              shuffle=False, save_path='data/sample_data.pkl', if_test='test')
    # 安全加载判别器奖励网络（路径指向 Step1_Reward_Recovery/disc_fv.pt）
    from torch.serialization import add_safe_globals
    sys.path.append(str(base_dir.parent / "Step1_Reward_Recovery"))  # 允许导入 AIRL 判别器定义
    from airl import AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet  # noqa: E402
    add_safe_globals([AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet])  # 仅在信任权重文件的前提下允许反序列化该类
    # 如果文件放在训练日志目录下，更新为实际保存路径
    # disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery" / "main1_20251124-212403" / "checkpoints" / "disc_fv.pt"
    # disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery" / "main1_20251210-232617" / "checkpoints" / "disc_fv.pt"
    # disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery_new" / "main2_SocialAIRL_20251211-100531" / "checkpoints" / "disc_fv.pt"
    # disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery_new" / "main2_SocialAIRL_20251211-100547" / "checkpoints" / "disc_fv.pt"
    # disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery_new" / "main2_SocialAIRL_20251211-100746" / "checkpoints" / "disc_fv.pt"
    disc_path = base_dir.parent / "Step1_Reward_Recovery" / "runs" / "log" / "reward_recovery_new" / "main2_SocialAIRL_20251211-101331" / "checkpoints" / "disc_fv.pt"

    # fv_reward = torch.load(disc_path, map_location=device, weights_only=False).g  # 原版 AIRL 判别器仅用私有奖励
    disc = torch.load(disc_path, map_location=device, weights_only=False)

    def _infer_hist_len(disc_obj):
        """
        从判别器首层 Linear 的输入维度推断历史堆叠倍数（Step1 训练时用 4 帧 -> 64 维）。
        若无法推断则退化为 1（不堆叠）。
        """
        in_dim = None
        first_linear = None
        if hasattr(disc_obj, "private_g"):
            first_linear = next((m for m in disc_obj.private_g if isinstance(m, nn.Linear)), None)
        if in_dim is None and first_linear is not None:
            in_dim = first_linear.in_features
        if in_dim is None and hasattr(disc_obj, "g"):
            first_linear = next((m for m in disc_obj.g if isinstance(m, nn.Linear)), None)
            if first_linear is not None:
                in_dim = first_linear.in_features
        if in_dim is not None and in_dim % 16 == 0:
            return max(1, in_dim // 16)
        return 1

    _disc_hist_len = _infer_hist_len(disc)
    def _stack_history_states(states, hist_len):
        """
        将 (B, T, 16) 叠成 (B, T, 16*hist_len)，前几帧用第 0 帧填充，保持与判别器训练时一致。
        """
        if hist_len <= 1:
            return states
        bsz, seq_len, feat = states.shape
        history = [states[:, 0, :]] * hist_len  # 初始重复首帧
        stacked = []
        for t in range(seq_len):
            if t > 0:
                history.pop(0)
                history.append(states[:, t, :])
            stacked.append(torch.cat(history, dim=-1))
        return torch.stack(stacked, dim=1)  # (B, T, hist_len*feat)

    def fv_reward(state):
        """
        state: (B, T, 16) -> 输出 (B, T, 1)
        兼容两类判别器：
          - SocialAIRLDiscrim：使用私有奖励 + alpha * 社会势场
          - AIRLDiscrim：仅使用 g (私有奖励)
        """
        state_hist = _stack_history_states(state, _disc_hist_len)
        flat = state_hist.reshape(-1, state_hist.shape[-1])
        if hasattr(disc, "get_reward") and hasattr(disc, "alpha"):
            r_priv, phi, _ = disc.get_reward(flat)
            # 标准化势场并削弱 alpha，防止奖励爆炸
            # phi = (phi - phi.mean()) / (phi.std())
            # phi = phi / (1.0 + torch.abs(phi))
            # 
            
            # 先对 phi 去极值
            q1, q9 = torch.quantile(phi, torch.tensor([0.01, 0.99], device=phi.device))
            phi = torch.clamp(phi, min=q1, max=q9)

            # 标准化或压缩
            
            #phi = (phi - phi.mean()) / (phi.std() + 1e-6)  # 或用 tanh/softsign
            phi = phi / (1.0 + torch.abs(phi))
            total = r_priv + (disc.alpha) * phi
            total = torch.clamp(total, -100.0, 100.0)

        elif hasattr(disc, "g"):
            total = disc.g(flat)
        else:
            total = disc(flat)
        return total.view(state_hist.shape[0], state_hist.shape[1], -1)
    best_ade = 1e5

    # 统一日志和模型保存路径（runs/planner_goal 下每次一个时间戳目录），直接复用模块级 base_dir
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "planner_goal" / f"train_{time_str}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f">>> TensorBoard logging to: {log_dir}")
    results_dir = log_dir / "results"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(args.total_epoch):
        epoch_loss = []

        for batch in train_loader:
            lcv_traj = batch[0].cpu().numpy()
            fv_traj = batch[1].cpu().numpy()
            nlv_traj = batch[2].cpu().numpy()
            olv_traj = batch[3].cpu().numpy()

            optim.zero_grad()
            cur_lr = optim.state_dict()['param_groups'][0]['lr']

            # Generate true reward
            n_lcv = normalize_obs(copy.deepcopy(lcv_traj))
            n_fv = normalize_obs(copy.deepcopy(fv_traj))
            n_nlv = normalize_obs(copy.deepcopy(nlv_traj))
            n_olv = normalize_obs(copy.deepcopy(olv_traj))

            dfv = n_lcv - n_fv
            dnlv = n_nlv - n_lcv
            dolv = n_olv - n_lcv

            state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=-1)).to(device).float()
            true_reward = fv_reward(state)

            # Noisy data
            noisy_fv, t = trainer(torch.from_numpy(n_fv).to(device))
            noisy_dfv = n_lcv - noisy_fv.cpu().numpy()
            noisy_state = torch.from_numpy(np.concatenate([n_lcv, noisy_dfv, dnlv, dolv], axis=-1)).to(device).float()
            pred_reward = model(noisy_state, t)

            loss = F.mse_loss(pred_reward, true_reward)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            epoch_loss.append(loss.item())

            # break

        if epoch % args.print_step == 0:
            print('Epoch {}, loss {:.6f}, lr {}'.format(epoch, np.mean(epoch_loss), cur_lr))

        if epoch % args.sample_step == 0:
            val_ade = eval(args, test_loader, trainer, model, fv_reward)

            if val_ade < best_ade:
                best_ade = val_ade
                print('Best model updated, testing ...')
                eval(args, test_loader, trainer, model, fv_reward)

                # 保存到当前日志目录的 results 子目录
                torch.save(ema_model, results_dir / 'best_model.pth')

        # TensorBoard 记录训练/验证指标
        writer.add_scalar('train/loss', np.mean(epoch_loss), epoch)
        if epoch % args.sample_step == 0:
            writer.add_scalar('val/ade', val_ade, epoch)
        writer.add_scalar('train/lr', optim.state_dict()['param_groups'][0]['lr'], epoch)

    writer.close()


def main(args):
    train(args)


if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')

    # Dataset
    parser.add_argument('--obs_len', type=int, default=20)
    parser.add_argument('--pred_len', type=int, default=100)
    parser.add_argument('--pred_len_val', type=int, default=100)
    parser.add_argument('--skip', type=int, default=5)

    # Backbone
    parser.add_argument('--node_feat_dim', type=int, default=0)
    parser.add_argument('--time_dim', type=int, default=64)
    parser.add_argument('--context_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)

    # Diffusion
    parser.add_argument('--beta_1', type=int, default=1e-4)
    parser.add_argument('--beta_T', type=int, default=0.05)
    parser.add_argument('--T', type=int, default=100)

    # Training
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--total_epoch', type=int, default=100000)
    parser.add_argument('--warmup', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--print_step', type=int, default=5)

    args = parser.parse_args()
    main(args)












