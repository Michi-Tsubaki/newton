# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from newton.examples.robot.nextage_contact_env import NextageLiftPlaceEnv, PPOActorCritic, PPOConfig


def _default_log_dir(exp_name: str) -> Path:
    return Path("logs") / f"{exp_name}_nextage_contact_ppo"


def _make_minibatches(batch_size: int, minibatch_size: int) -> list[torch.Tensor]:
    indices = torch.randperm(batch_size)
    return [indices[start : start + minibatch_size] for start in range(0, batch_size, minibatch_size)]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = _default_log_dir(args.exp_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = NextageLiftPlaceEnv(
        num_envs=args.num_envs,
        seed=args.seed,
        headless=True,
        use_cuda_graph=not args.no_cuda_graph,
    )
    obs = torch.as_tensor(env.reset_all(), device=device)

    config = PPOConfig(obs_dim=env.obs_dim, action_dim=env.action_dim)
    model = PPOActorCritic(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    rollout_steps = args.rollout_steps
    gamma = args.gamma
    lam = args.gae_lambda
    clip_range = args.clip_range
    value_coef = args.value_coef
    entropy_coef = args.entropy_coef
    max_grad_norm = args.max_grad_norm

    total_steps = 0
    for update in range(1, args.num_updates + 1):
        obs_buf = torch.zeros((rollout_steps, args.num_envs, env.obs_dim), device=device)
        actions_buf = torch.zeros((rollout_steps, args.num_envs, env.action_dim), device=device)
        logprobs_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
        rewards_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
        dones_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
        success_buf = torch.zeros((rollout_steps, args.num_envs), device=device)
        values_buf = torch.zeros((rollout_steps, args.num_envs), device=device)

        for step in range(rollout_steps):
            with torch.no_grad():
                action, logprob, value = model.act(obs)
            next_obs, reward, done, info = env.step(action.detach().cpu().numpy())
            next_obs = torch.as_tensor(next_obs, device=device)
            reward = torch.as_tensor(reward, device=device)
            done = torch.as_tensor(done, device=device)
            success = torch.as_tensor(info["success"], device=device)

            obs_buf[step] = obs
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            rewards_buf[step] = reward
            dones_buf[step] = done.float()
            success_buf[step] = success
            values_buf[step] = value
            obs = next_obs
            total_steps += args.num_envs

        with torch.no_grad():
            _, next_value = model.forward(obs)

        advantages = torch.zeros_like(rewards_buf)
        gae = torch.zeros(args.num_envs, device=device)
        for step in reversed(range(rollout_steps)):
            next_nonterminal = 1.0 - dones_buf[step]
            next_values = next_value if step == rollout_steps - 1 else values_buf[step + 1]
            delta = rewards_buf[step] + gamma * next_values * next_nonterminal - values_buf[step]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[step] = gae
        returns = advantages + values_buf

        b_obs = obs_buf.reshape(-1, env.obs_dim)
        b_actions = actions_buf.reshape(-1, env.action_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1.0e-8)

        batch_size = b_obs.shape[0]
        minibatch_size = max(1, batch_size // args.num_minibatches)
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0

        for _ in range(args.ppo_epochs):
            for mb_inds in _make_minibatches(batch_size, minibatch_size):
                new_logprob, entropy, value = model.evaluate_actions(b_obs[mb_inds], b_actions[mb_inds])
                log_ratio = new_logprob - b_logprobs[mb_inds]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * b_advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * b_advantages[mb_inds]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = value
                if args.clip_value_loss:
                    value_clipped = b_values[mb_inds] + torch.clamp(
                        value_pred - b_values[mb_inds], -clip_range, clip_range
                    )
                    value_loss = 0.5 * torch.max(
                        (value_pred - b_returns[mb_inds]).square(),
                        (value_clipped - b_returns[mb_inds]).square(),
                    ).mean()
                else:
                    value_loss = 0.5 * (value_pred - b_returns[mb_inds]).square().mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                last_policy_loss = float(policy_loss.detach().cpu())
                last_value_loss = float(value_loss.detach().cpu())
                last_entropy = float(entropy.mean().detach().cpu())

        avg_reward = float(rewards_buf.mean().detach().cpu())
        avg_success = float(success_buf.max(dim=0).values.mean().detach().cpu())
        print(
            f"update={update:04d} steps={total_steps} reward={avg_reward:.3f} success={avg_success:.3f} "
            f"policy_loss={last_policy_loss:.3f} value_loss={last_value_loss:.3f} entropy={last_entropy:.3f}"
        )

        if update % args.save_interval == 0 or update == args.num_updates:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "args": vars(args),
            }
            torch.save(checkpoint, log_dir / f"checkpoint_{update:05d}.pt")

        obs = env.reset_all()
        obs = torch.as_tensor(obs, device=device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp_nextage_contact")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_updates", type=int, default=500)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)
    parser.add_argument("--value_coef", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=25)
    parser.add_argument("--clip_value_loss", action="store_true", default=True)
    parser.add_argument("--no_cuda_graph", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
