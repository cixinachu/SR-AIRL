import os
import sys
from pathlib import Path
import copy
import pickle
import warnings
import numpy as np
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt
import torch
from torch.serialization import add_safe_globals  # 允许反序列化自定义模型
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
from tqdm import tqdm

# ensure project root (Diff-LC) is on sys.path for package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Step5_Evaluation.model.ddpm import GaussianDiffusionSampler
from Step5_Evaluation.utils.ngsim_dataset import data_loader, TrajectoryDataset
from Step5_Evaluation.model.backbone import TransformerConcatLinear, NoisyReward  # 供权重反序列化使用
warnings.filterwarnings('ignore')


device = 'cuda'
# 允许自定义类的反序列化（权重文件可信前提下）
add_safe_globals([TransformerConcatLinear, NoisyReward])
model = torch.load('trained_model/predictor.pth', map_location=device, weights_only=False)
ema_model = copy.deepcopy(model)
guide = torch.load('trained_model/guide.pth', map_location=device, weights_only=False).to(device)
model.eval()
ema_sampler = GaussianDiffusionSampler(ema_model, beta_1=1e-4, beta_T=0.05, T=100).to(device)

test_dataset = TrajectoryDataset('',
                                 obs_len=20,
                                 pred_len=100,
                                 skip=5, save_dir='data/sample_data.pkl', if_test='test')
# Prediction
# 统一日志/结果目录，优先使用 0_main 传入的环境变量
log_dir = Path(os.environ.get("EVAL_LOG_DIR", Path(__file__).resolve().parent / "run" / "eval" / "debug"))
results_dir = Path(os.environ.get("EVAL_RESULTS_DIR", log_dir / "results"))
os.makedirs(results_dir, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))
plots_dir = results_dir / "plots"
os.makedirs(plots_dir, exist_ok=True)  # 保存可视化图片的目录
SAVE_PLOTS = True  # 开关：如需关闭图片保存可置 False

plan_path = results_dir / 'plan_data.pkl'
if not plan_path.exists():
    raise FileNotFoundError(f"缺少规划结果 {plan_path}，请先运行 1_planning.py（或 0_main.py）生成 plan_data.pkl")

with open(plan_path, 'rb') as file:
    plan_data = pickle.load(file)

for traj_id in tqdm(range(53)):
    # traj_id = 34  # 旧调试代码（停用）；默认处理全部 53 条轨迹
    plan_data[traj_id]['predicted'] = []
    hist_traj = test_dataset.hist_traj[traj_id:traj_id+1].to(device)
    hist_init = copy.deepcopy(hist_traj[:, 0:1, :2]).to(device)
    state_data = test_dataset.state_data[traj_id:traj_id+1].to(device)
    cond_x_init = plan_data[traj_id]['true_x'][0]
    cond_y_init = plan_data[traj_id]['true_y'][0]

    with torch.no_grad():
        for p in range(len(plan_data[traj_id]['planned'])):
            cond_traj = plan_data[traj_id]['planned'][p].unsqueeze(0).float()
            cond_traj[:, :, :2] -= hist_init.cpu()

            diff_pred_traj = torch.randn(1, 100, 4).to(device)
            hist_traj[:, :, :2] -= hist_init

            hist_traj = hist_traj.to(device)
            cond_traj = cond_traj.to(device)
            context = model.context(hist_traj, cond_traj)

            # Sampling
            pred_diff, _ = ema_sampler(node_loc=diff_pred_traj,
                                       context=context, state=state_data, guide=guide)
            hist_traj[:, :, :2] += hist_init
            preds = torch.cat([hist_traj[:, -1:, :2], pred_diff[:, :, :2]], dim=1)
            preds = torch.cumsum(preds, dim=1)[:, 1:, :]
            velocity = pred_diff[:, :, 2:]

            plan_data[traj_id]['predicted'].append(torch.cat([preds, velocity], dim=-1))
            # break

            # 保存预测/条件轨迹可视化到 plots 目录，避免阻塞显示
            # if SAVE_PLOTS:
            #     av_traj = (cond_traj[:, :, :2] + hist_init).detach().cpu()
            #     hv_traj = preds.detach().cpu()
            #     fig, ax = plt.subplots()
            #     ax.plot(hv_traj[0, :, 1], hv_traj[0, :, 0], color='orange', label='pred')
            #     ax.plot(av_traj[0, :, 1], av_traj[0, :, 0], color='blue', label='cond')
            #     for y in [3.6576, 7.3152, 10.9728, 14.6304, 18.288, 21.9456]:
            #         ax.axhline(y, linestyle='--', color='red', linewidth=0.8)
            #     ax.legend()
            #     ax.set_title(f"traj{traj_id}_plan{p}")
            #     fig.savefig(plots_dir / f"traj{traj_id}_plan{p}.png", dpi=150)
            #     plt.close(fig)
            if SAVE_PLOTS:
                av_traj = (cond_traj[:, :, :2] + hist_init).detach().cpu()
                hv_traj = preds.detach().cpu()
                vel = velocity.detach().cpu()

                # 基于速度的渐变色轨迹绘制（预测轨迹按照速度大小着色）
                speed = torch.norm(vel[0], dim=-1).numpy()
                hv_xy = hv_traj[0, :, [1, 0]].numpy()  # x=lon, y=lat
                points = hv_xy.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                norm = plt.Normalize(speed.min(), speed.max())
                lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=2.5, label='pred')
                lc.set_array(0.5 * (speed[:-1] + speed[1:]))  # 用段内平均速度着色

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.add_collection(lc)
                ax.plot(av_traj[0, :, 1], av_traj[0, :, 0], color='#475569', linestyle='--', linewidth=1.5, label='cond')
                for y in [3.6576, 7.3152, 10.9728, 14.6304, 18.288, 21.9456]:
                    ax.axhline(y, linestyle='--', color='#e2e8f0', linewidth=0.8)
                ax.legend()
                ax.set_title(f"traj{traj_id}_plan{p}")
                cbar = fig.colorbar(lc, ax=ax, pad=0.02)
                cbar.set_label("speed (norm)")
                fig.savefig(plots_dir / f"traj{traj_id}_plan{p}.png", dpi=150, bbox_inches='tight')
                plt.close(fig)

            # 旧版直接弹窗显示的代码（已停用，改为保存图片）
            # av_traj = (cond_traj[:, :, :2] + hist_init).detach().cpu()
            # hv_traj = preds.detach().cpu()
            # plt.plot(hv_traj[0, :, 1], hv_traj[0, :, 0], color='orange')
            # plt.plot(av_traj[0, :, 1], av_traj[0, :, 0], color='blue')
            # plt.axhline(3.6576, linestyle='--', color='red')
            # plt.axhline(7.3152, linestyle='--', color='red')
            # plt.axhline(10.9728, linestyle='--', color='red')
            # plt.axhline(14.6304, linestyle='--', color='red')
            # plt.axhline(18.288, linestyle='--', color='red')
            # plt.axhline(21.9456, linestyle='--', color='red')
            # plt.show()

with open(results_dir / 'pred_data.pkl', 'wb') as path:
    pickle.dump(plan_data, path)

# TensorBoard 简要记录
writer.add_scalar('prediction/num_traj', len(plan_data), 0)
writer.close()



















