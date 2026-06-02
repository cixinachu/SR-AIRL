import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from NGSIM_env.data.ngsim import *
import pickle
from pathlib import Path


def trajectory_process(trajectory, decision_time):
    """
    给原始轨迹添加速度/加速度，并截取决策帧前的片段。

    输入的 `trajectory` 列顺序会先重命名为统一格式，再做差分：
    - v_x/v_y: 位置差分 / 0.1s
    - a_x/a_y: 速度差分 / 0.1s
    返回包含 frame/id/位置/尺寸/车道号/速度/加速度的 DataFrame。
    """
    trajectory.columns = ['frame', 'id', 'y', 'x', 'width', 'height', 'laneId']
    trajectory['v_x'] = trajectory['x'].diff() / 0.1
    trajectory['v_y'] = trajectory['y'].diff() / 0.1
    trajectory['a_x'] = trajectory['v_x'].diff() / 0.1
    trajectory['a_y'] = trajectory['v_y'].diff() / 0.1

    # 只保留决策帧前的 2s（20 帧）及之后的片段
    trajectory = trajectory[trajectory['frame'] >= decision_time - 20]

    return trajectory


def build_trajecotry():
    """
    从预处理的 pair 数据构建四车轨迹集合，保留速度/加速度字段。
    返回 dict: {'lcv': [df,...], 'fv': [...], 'nlv': [...], 'olv': [...]}。
    """
    base_dir = Path(__file__).resolve().parent  # .../NGSIM_env/data
    data_path = base_dir.parent / "data" / "sample_data.pkl"
    with open(data_path, "rb") as input_file:
        data = pickle.load(input_file)

    record_trajectory = {'lcv': [], 'fv': [], 'nlv': [], 'olv': []}
    
    for pair in data:
        end_frame = pair['end_frame'].values[0]
        decision_frame = pair['decision_frame'].values[0]
        # 将位置/尺寸/车道列切片给四辆车；速度/加速度在 trajectory_process 中差分得到
        lcv = pair[['frame', 'id_x', 'y_x', 'x_x', 'width_x', 'height_x', 'laneId_x']]
        fv  = pair[['frame', 'id_y', 'y_y', 'x_y', 'width_y', 'height_y', 'laneId_y']]
        nlv = pair[['frame', 'id',   'y',   'x',   'width',  'height',  'laneId']]
        olv = pair[['frame', 'id_z', 'y_z', 'x_z', 'width_z','height_z','laneId_z']]

        lcv = trajectory_process(lcv, decision_frame)
        fv = trajectory_process(fv, decision_frame)
        nlv = trajectory_process(nlv, decision_frame)
        olv = trajectory_process(olv, decision_frame)

        if lcv.isnull().values.any() or fv.isnull().values.any() or nlv.isnull().values.any() or olv.isnull().values.any():
            continue

        record_trajectory['lcv'].append(lcv)
        record_trajectory['fv'].append(fv)
        record_trajectory['nlv'].append(nlv)
        record_trajectory['olv'].append(olv)

    return record_trajectory


if __name__ == "__main__":
    # 简易入口：重新构建轨迹并输出带速度/加速度的新 pkl
    traj = build_trajecotry()
    out_path = Path(__file__).resolve().parent / 'sample_data_with_speed.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(traj, f)
    print(f"Saved trajectories with speed/acc to {out_path}")
