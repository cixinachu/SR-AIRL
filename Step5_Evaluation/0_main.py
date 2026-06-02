import os
import sys  # 用于在步骤失败时立即退出，确保强制顺序执行
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 记录运行状态

# 统一日志/结果目录（run/eval 下每次一个时间戳子目录）
base_dir = Path(__file__).resolve().parent
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = base_dir / "run" / "eval" / f"run_{time_str}"
results_dir = log_dir / "results"
os.makedirs(results_dir, exist_ok=True)

# 环境变量传递给子脚本，确保读写同一目录
os.environ["EVAL_LOG_DIR"] = str(log_dir)
os.environ["EVAL_RESULTS_DIR"] = str(results_dir)

writer = SummaryWriter(log_dir=str(log_dir))

print('Start Planning ...')
ret1 = os.system("python 1_planning.py")
writer.add_scalar('status/planning_exit_code', ret1, 0)
if ret1 != 0:
    print(f"Planning failed with exit code {ret1}, aborting subsequent steps.")
    writer.close()
    sys.exit(ret1)  # 规划失败则直接退出，强制顺序执行

print('Start Prediction ...')
ret2 = os.system("python 2_prediction.py")
writer.add_scalar('status/prediction_exit_code', ret2, 0)
if ret2 != 0:
    print(f"Prediction failed with exit code {ret2}, aborting evaluation.")
    writer.close()
    sys.exit(ret2)  # 预测失败则直接退出，强制顺序执行

print('Start Evaluation ...')
ret3 = os.system("python 3_evaluation.py")
writer.add_scalar('status/evaluation_exit_code', ret3, 0)

writer.close()



