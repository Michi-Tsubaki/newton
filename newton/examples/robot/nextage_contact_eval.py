# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import newton.examples
from newton.examples.robot.nextage_contact_env import NextageLiftPlaceEnv, PPOActorCritic, PPOConfig


def _latest_checkpoint(log_dir: Path) -> Path:
    checkpoint_files = sorted(log_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    return max(checkpoint_files, key=lambda path: int(path.stem.split("_")[-1]))


def build_parser() -> argparse.ArgumentParser:
    parser = newton.examples.create_parser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp_nextage_contact")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    viewer, args = newton.examples.init(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = Path("logs") / f"{args.exp_name}_nextage_contact_ppo"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else _latest_checkpoint(log_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    env = NextageLiftPlaceEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        headless=viewer is None,
        viewer=viewer,
        use_cuda_graph=False,
    )
    obs = torch.as_tensor(env.reset_all(), device=device)

    config = PPOConfig(**checkpoint["config"])
    model = PPOActorCritic(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for episode in range(args.num_episodes):
        obs = torch.as_tensor(env.reset_all(), device=device)
        total_reward = torch.zeros(env.num_envs, device=device)
        for _ in range(env.episode_steps):
            with torch.no_grad():
                action, _, _ = model.act(obs, deterministic=True)
            obs, reward, done, info = env.step(action.detach().cpu().numpy())
            obs = torch.as_tensor(obs, device=device)
            total_reward += torch.as_tensor(reward, device=device)
            if viewer is not None:
                env.render()
        print(
            f"episode={episode + 1} mean_reward={float(total_reward.mean().detach().cpu()):.3f} "
            f"success_rate={float(info['success'].mean().detach().cpu()):.3f}"
        )


if __name__ == "__main__":
    main()
