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
from airl import AIRLDiscrim, SocialAIRLDiscrim, AnisotropicInteractionPotential  # noqa: E402
add_safe_globals([AIRLDiscrim, SocialAIRLDiscrim, AnisotropicInteractionPotential])

# 判别器权重路径可通过环境变量覆盖，默认使用 trained_model_best
default_disc = Path(__file__).resolve().parent / "trained_model_best" / "disc_lcv.pt"
disc_path = Path(os.environ.get("EVAL_DISC_PATH", default_disc))
if not disc_path.exists():
    raise FileNotFoundError(f"未找到判别器权重文件：{disc_path}")

disc_lcv = torch.load(disc_path, map_location=device, weights_only=False)


def _flatten_state(x: torch.Tensor) -> torch.Tensor:
    # 将 (B, T, C) 展平为 (B*T, C)，兼容判别器输入
    return x.view(-1, x.shape[-1]) if x.dim() > 2 else x


def reward_lcv(state):
    """Return the discriminator's total reward without recombination."""
    flat_state = _flatten_state(state)
    r_total, _, _ = disc_lcv.get_reward(flat_state)
    return r_total


def _infer_hist_len(disc_obj) -> int:
    """
    从判别器首层 Linear 输入维度推断历史堆叠倍数（单帧为 16 维）。
    若无法推断则退化为 1（不堆叠），避免影响非历史模型。
    """
    if hasattr(disc_obj, "history_len"):
        return disc_obj.history_len
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

    all_reward = []
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

        with torch.no_grad():
            reward = reward_lcv(state).mean()

        all_reward.append(reward.item())
        all_lcv.append(plan_data[scene_id]['planned'][p].detach().cpu().numpy())

    # Select exclusively by mean total reward, without ground-truth errors.
    best_idx = int(np.argmax(all_reward))
    best_reward = all_reward[best_idx]
    best_lcv = all_lcv[best_idx]

    # Ground truth is used only to evaluate the selected trajectory.
    position_error = np.linalg.norm(true_lcv[:, :2] - best_lcv[:, :2], axis=1)
    best_rmse = float(position_error.mean())
    best_final_rmse = float(position_error[-1])
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






























