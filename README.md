# SR-AIRL

This repository contains a simple research pipeline for reward recovery, diffusion-based planning, reward-guided prediction, and evaluation on vehicle trajectory data.

## Structure

- `Step1_Reward_Recovery/`: AIRL reward recovery code and NGSIM environment utilities.
- `Step2_Planner/`: diffusion planner training code.
- `Step3_Reward_Guide/`: reward-guided diffusion training code.
- `Step4_Predictor/`: trajectory prediction code.
- `Step5_Evaluation/`: planning, prediction, and evaluation scripts.
- `img/`: figures and result images.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run each step from its own folder:

```bash
cd Step1_Reward_Recovery
python main_helbing.py

cd ../Step2_Planner
python train_dppo.py

cd ../Step3_Reward_Guide
python main.py

cd ../Step4_Predictor
python main.py

cd ../Step5_Evaluation
python 0_main.py
```

Sample data files are included under the `data/` folders for each step.

## License

See `LICENSE`.
