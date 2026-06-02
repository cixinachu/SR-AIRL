import torch
import numpy as np
from model.temporal import TemporalUnet
from model.diffusion import GaussianDiffusion
from utils.training import Trainer
from utils.dataset import SequenceDataset
from utils.arrays import *
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录
from pathlib import Path
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def main():
    device = 'cuda'
    data_path = 'data/sample_data.pkl'
    train_dataset = SequenceDataset(data_path, horizon=100)

    # 统一基准路径为当前脚本所在目录，避免从不同工作目录运行时找不到日志/模型保存位置
    base_dir = Path(__file__).resolve().parent

    model = TemporalUnet(horizon=100,
                         transition_dim=4+2,
                         dim=64,
                         cond_dim=4,
                         dim_mults=(1, 2, 4, 8),
                         attention=False).to(device)

    diffusion = GaussianDiffusion(model,
                                  horizon=100,
                                  observation_dim=4,
                                  action_dim=2,
                                  n_timesteps=20,
                                  loss_type='l2',
                                  clip_denoised=False,
                                  predict_epsilon=False,
                                  ## loss weighting
                                  action_weight=10,
                                  loss_weights=None,
                                  loss_discount=1).to(device)

    # 确保模型保存目录存在，避免 torch.save 找不到父目录
    results_dir = base_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 提前创建 TensorBoard writer，便于传入 Trainer 内部日志（避免下方未定义）
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / "runs" / "planner_dppo" / f"train_{time_str}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f">>> TensorBoard logging to: {log_dir}")

    trainer = Trainer(diffusion,
                      dataset=train_dataset,
                      train_batch_size=32,
                      train_lr=2e-4,
                      gradient_accumulate_every=2,
                      ema_decay=0.995,
                      sample_freq=20000,
                      save_freq=4000,
                      label_freq=int(1E6 // 5),
                      save_parallel=False,
                      results_folder=str(results_dir),  # 使用已创建的保存目录
                      bucket=None,
                      n_reference=8,
                      writer=writer,  # 传入 TensorBoard Writer，训练内部日志与当前 log_dir 对齐
                      save_last_only=True,

                      # ---- DPPO over denoising (no Step1 reward) ----
                      use_dppo=True,
                      dppo_coef=0.02,      # 先小一点，别抢 BC
                      ppo_clip=0.2,
                      dppo_gamma=0.99,
                      dppo_every=10,       # 每 10 个 train step 做一次 PPO（省算力也更稳）
                      )  # 新增：只保存最后一次模型，中途不产生多份 checkpoint

    print('Testing forward...', end=' ', flush=True)
    batch = batchify(train_dataset[0])
    loss, _ = diffusion.loss(*batch)
    loss.backward()
    print('✓')

    # Main Loop
    n_epochs = int(1e6 // 10000)
    # TensorBoard：记录每个 epoch 开始（对齐终端输出），writer 已在上方创建

    for i in range(n_epochs):
        print(f'Epoch {i} / {n_epochs} |')
        writer.add_scalar('train/epoch', i, i)  # 与 print 对齐的 epoch 计数
        trainer.train(n_train_steps=10000)

    writer.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
