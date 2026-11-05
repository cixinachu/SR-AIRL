"""Validate whether recovered SR-AIRL rewards rank expert states above policy states.

This script evaluates only Step1's recovered reward r_total.  It does not use
the reward-guide approximation R_c or Step2's denoising-consistency reward.
"""

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Required for loading full-module checkpoints saved by main3_helbing.py.
from airl import AIRLDiscrim, AnisotropicInteractionPotential, SocialAIRLDiscrim  # noqa: F401
from ppo_model.PPO import PPO, RolloutBuffer  # noqa: F401


DEFAULT_RUN_DIR = Path(__file__).resolve().parent / "runs" / "log2" / "reward_recovery_new"


def latest_run_dir(root: Path) -> Path:
    runs = [path for path in root.iterdir() if (path / "checkpoints" / "disc_lcv.pt").is_file()]
    if not runs:
        raise FileNotFoundError(f"No completed SR-AIRL run found under {root}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def stack_history(states: torch.Tensor, dones: torch.Tensor, history_len: int) -> torch.Tensor:
    """Match Step1's done-aware expert-state history construction."""
    if history_len <= 1:
        return states
    frames = []
    history = []
    for index, state in enumerate(states):
        if index > 0 and bool(dones[index - 1].item()):
            history = []
        history.append(state)
        history = history[-history_len:]
        frames.append(torch.cat([history[0]] * (history_len - len(history)) + history).reshape(1, -1))
    return torch.cat(frames, dim=0)


def policy_states(buffer_path: Path) -> torch.Tensor:
    buffer = torch.load(buffer_path, map_location="cpu", weights_only=False)
    if not buffer.states:
        raise ValueError(f"Policy rollout buffer is empty: {buffer_path}")
    return torch.cat(buffer.states, dim=0).float()


def score_states(discriminator, states: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    rewards = []
    discriminator.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            state_batch = states[start:start + batch_size].to(device)
            r_total, _, _ = discriminator.get_reward(state_batch)
            rewards.append(r_total.squeeze(-1).cpu())
    return torch.cat(rewards).numpy()


def summary(role: str, data: str, rewards: np.ndarray) -> dict:
    return {
        "role": role.upper(),
        "data": data,
        "count": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)),
        "median_reward": float(np.median(rewards)),
    }


def save_distribution_plot(role: str, expert: np.ndarray, policy: np.ndarray, output_path: Path) -> None:
    low, high = np.quantile(np.concatenate([expert, policy]), [0.005, 0.995])
    bins = np.linspace(low, high, 61)
    plt.figure(figsize=(8, 4.5))
    plt.hist(expert, bins=bins, density=True, histtype="step", linewidth=2.2, label="Expert")
    plt.hist(policy, bins=bins, density=True, histtype="step", linewidth=2.2, label="Policy")
    plt.xlabel("SR-AIRL recovered total reward $r_{total}$")
    plt.ylabel("Probability density")
    plt.title(f"{role.upper()}: expert vs. policy reward distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_markdown(rows: list[dict], output_path: Path) -> None:
    lines = [
        "# SR-AIRL recovered reward validation",
        "",
        "Only Step1 recovered reward `r_total` is evaluated here. `R_c` and Step2 consistency reward are excluded.",
        "",
        "| Role | Data | Count | Mean reward | Std | Median |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['data']} | {row['count']} | "
            f"{row['mean_reward']:.6f} | {row['std_reward']:.6f} | {row['median_reward']:.6f} |"
        )
    lines.append("")
    for role in ("LCV", "FV"):
        role_rows = {row["data"]: row for row in rows if row["role"] == role}
        delta = role_rows["Expert"]["mean_reward"] - role_rows["Policy"]["mean_reward"]
        lines.append(f"- {role}: expert-minus-policy mean reward = `{delta:.6f}`.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=None, help="Step1 run directory; defaults to the newest completed run.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(DEFAULT_RUN_DIR)
    checkpoint_dir = run_dir / "checkpoints"
    output_dir = run_dir / "reward_validation"
    output_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)

    with (run_dir / "expert_data.pkl").open("rb") as handle:
        expert_data = pickle.load(handle)
    expert_frames = expert_data["state"].float()

    rows = []
    for role in ("lcv", "fv"):
        discriminator = torch.load(checkpoint_dir / f"disc_{role}.pt", map_location=device, weights_only=False).to(device)
        history_len = discriminator.history_len
        expert_rewards = score_states(
            discriminator,
            stack_history(expert_frames, expert_data["dones"], history_len),
            device,
            args.batch_size,
        )
        policy_rewards = score_states(discriminator, policy_states(checkpoint_dir / f"{role}_buffer.pt"), device, args.batch_size)

        rows.extend([summary(role, "Expert", expert_rewards), summary(role, "Policy", policy_rewards)])
        save_distribution_plot(role, expert_rewards, policy_rewards, output_dir / f"reward_distribution_{role}.png")

    with (output_dir / "reward_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, output_dir / "reward_summary.md")

    print(f"Validated run: {run_dir}")
    for row in rows:
        print("{role:>3} {data:>6}: mean={mean_reward:+.6f}, std={std_reward:.6f}, median={median_reward:+.6f}, n={count}".format(**row))
    print(f"Saved table and distribution plots to: {output_dir}")


if __name__ == "__main__":
    main()

