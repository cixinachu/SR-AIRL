import os
import sys
from pathlib import Path
import copy
import time
import pickle
import warnings

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals  # 允许反序列化自定义类
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
warnings.filterwarnings('ignore')

# ensure project root (Diff-LC) is on sys.path for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Step5_Evaluation.model.related_module import *

# 统一日志/结果目录，优先使用 0_main 传入的环境变量
log_dir = Path(os.environ.get("EVAL_LOG_DIR", Path(__file__).resolve().parent / "run" / "eval" / "debug"))
results_dir = Path(os.environ.get("EVAL_RESULTS_DIR", log_dir / "results"))
os.makedirs(results_dir, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))

device = 'cuda'

# Evaluation Module（安全加载判别器权重）
airl_dir = Path(__file__).resolve().parent.parent / "Step1_Reward_Recovery"  # 判别器类定义位置
sys.path.append(str(airl_dir))
from airl import AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet  # noqa: E402
add_safe_globals([AIRLDiscrim, SocialAIRLDiscrim, SocialPotentialNet])

# 判别器权重路径可通过环境变量覆盖，默认使用 trained_model_best
default_disc = Path(__file__).resolve().parent / "trained_model_best" / "disc_lcv.pt"
disc_path = Path(os.environ.get("EVAL_DISC_PATH", default_disc))
if not disc_path.exists():
    raise FileNotFoundError(f"未找到判别器权重文件：{disc_path}")

disc_lcv = torch.load(disc_path, map_location=device, weights_only=False)
USE_SOCIAL_REWARD = os.environ.get("EVAL_USE_SOCIAL_REWARD", "0") == "1"


def _flatten_state(x: torch.Tensor) -> torch.Tensor:
    # 将 (B, T, C) 展平为 (B*T, C)，兼容判别器输入
    return x.view(-1, x.shape[-1]) if x.dim() > 2 else x


def build_reward_fn(disc, use_social: bool):
    """
    兼容 AIRLDiscrim（g/h）与 SocialAIRLDiscrim（private_g/alpha/shared_phi_net/get_reward）。
    use_social 为 True 时叠加势场，否则只用私有奖励。
    """
    if hasattr(disc, "g"):
        return lambda s: disc.g(_flatten_state(s))

    if hasattr(disc, "get_reward"):
        def _fn(s):
            flat = _flatten_state(s)
            r_priv, phi, _ = disc.get_reward(flat)
            if use_social and hasattr(disc, "alpha"):
                return r_priv + disc.alpha * phi
            return r_priv
        return _fn

    if hasattr(disc, "private_g"):
        def _fn(s):
            flat = _flatten_state(s)
            base = disc.private_g(flat)
            if use_social and hasattr(disc, "alpha") and hasattr(disc, "shared_phi_net"):
                base = base + disc.alpha * disc.shared_phi_net(flat)
            return base
        return _fn

    # 兜底：直接调用 forward
    return lambda s: disc(_flatten_state(s))


reward_lcv = build_reward_fn(disc_lcv, USE_SOCIAL_REWARD)


def _infer_hist_len(disc_obj) -> int:
    """
    从判别器首层 Linear 输入维度推断历史堆叠倍数（单帧为 16 维）。
    若无法推断则退化为 1（不堆叠），避免影响非历史模型。
    """
    first_linear = None
    if hasattr(disc_obj, "private_g"):
        first_linear = next((m for m in disc_obj.private_g if isinstance(m, nn.Linear)), None)
    if first_linear is None and hasattr(disc_obj, "g"):
        first_linear = next((m for m in disc_obj.g if isinstance(m, nn.Linear)), None)
    if first_linear is not None and first_linear.in_features % 16 == 0:
        return max(1, first_linear.in_features // 16)
    return 1


def _stack_history_states(states: torch.Tensor, hist_len: int) -> torch.Tensor:
    """
    将 (B, T, 16) 按 hist_len 叠成 (B, T, 16*hist_len)，前几帧用首帧填充。
    hist_len<=1 时直接返回，兼容非历史判别器。
    """
    if states.dim() == 2:
        states = states.unsqueeze(0)  # 单序列补 batch 维度，形状变为 (1, T, C)
    if hist_len <= 1:
        return states 
    bsz, seq_len, feat = states.shape
    history = [states[:, 0, :]] * hist_len
    stacked = []
    for t in range(seq_len):
        if t > 0:
            history.pop(0)
            history.append(states[:, t, :])
        stacked.append(torch.cat(history, dim=-1))
    return torch.stack(stacked, dim=1)


_disc_hist_len = _infer_hist_len(disc_lcv)

# load data
pred_data = pickle.load(open(results_dir / 'pred_data.pkl', 'rb'))
plan_data = pickle.load(open(results_dir / 'plan_data.pkl', 'rb'))
trajectory_set = build_trajecotry()
best_res = {}
all_ade = []
all_fde = []
collide_count = 0
for scene_id in range(53):
    print(scene_id)
    nlv = trajectory_set['nlv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    olv = trajectory_set['olv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    true_lcv = trajectory_set['lcv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]
    true_fv = trajectory_set['fv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100, :4]

    best_reward = -1e5
    all_reward = []
    all_rmse = []
    all_final_rmse = []
    all_lcv = []

    for p in range(len(plan_data[scene_id]['planned'])):
        lcv = plan_data[scene_id]['planned'][p][:, :4].detach().cpu().numpy()
        fv = pred_data[scene_id]['predicted'][p][0].detach().cpu().numpy()

        n_lcv = normalize_obs(copy.deepcopy(lcv))
        n_fv = normalize_obs(copy.deepcopy(fv))
        n_nlv = normalize_obs(copy.deepcopy(nlv))
        n_olv = normalize_obs(copy.deepcopy(olv))

        dfv = n_lcv - n_fv
        dnlv = n_nlv - n_lcv
        dolv = n_olv - n_lcv

        start = time.time()
        state = torch.from_numpy(np.concatenate([n_lcv, dfv, dnlv, dolv], axis=1)).to(device).float()
        state = _stack_history_states(state, _disc_hist_len)

        reward = reward_lcv(state).mean()

        # Calculate rmse
        loss = torch.cat([torch.from_numpy(true_lcv[:, 1:2]), torch.from_numpy(true_lcv[:, 0:1])], dim=-1) - \
               torch.cat([torch.from_numpy(lcv[:, 1:2]), torch.from_numpy(lcv[:, 0:1])], dim=-1)
        loss = (loss ** 2).sum(dim=1).sqrt()
        rmse = loss.mean()
        final_rmse = loss[-1]

        all_reward.append(reward.item())
        all_rmse.append(rmse.item())
        all_final_rmse.append(final_rmse.item())
        all_lcv.append(plan_data[scene_id]['planned'][p].detach().cpu().numpy())


    _, idx = torch.topk(torch.from_numpy(np.array(all_reward)), k=3)
    min_idx = idx[np.array(all_final_rmse)[idx.numpy()].argmin()]
    best_reward = all_reward[min_idx]
    best_lcv = all_lcv[min_idx]
    best_rmse = all_rmse[min_idx]
    best_final_rmse = all_final_rmse[min_idx]
    best_res[scene_id] = [best_lcv,
                          trajectory_set['lcv'][scene_id][['y', 'x', 'v_x', 'v_y', 'a_x', 'a_y']].values[:100],
                          best_rmse, best_final_rmse]
    all_ade.append(best_rmse)
    all_fde.append(best_final_rmse)
    print('Best reward {:.4}, best rmse {:.4}, best final {:.4}'.format(best_reward, best_rmse, best_final_rmse))

with open(results_dir / 'final_data.pkl', 'wb') as path:
    pickle.dump(best_res, path)

# 汇总指标写入 TensorBoard
writer.add_scalar('evaluation/mean_ade', np.mean(all_ade), 0)
writer.add_scalar('evaluation/mean_fde', np.mean(all_fde), 0)
writer.add_scalar('evaluation/collide_count', collide_count, 0)
writer.close()





























