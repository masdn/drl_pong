import os
import json
import time
import glob
from datetime import datetime

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

import pong_plots


# Register ALE Atari environments with Gymnasium (needed for ALE/Pong-v5, etc.)
gym.register_envs(ale_py)

script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_pong_env(
    env_name: str,
    seed: int,
    render_mode: str | None = None,
    frame_skip: int = 1,
    frame_stack: int = 4,
    grayscale: bool = True,
    scale_obs: bool = True,
):
    """
    Create a Pong environment with standard Atari preprocessing.

    Uses Gymnasium's AtariPreprocessing and FrameStack wrappers to:
    - Downsample to 84x84
    - Convert to grayscale (optional)
    - Frame-skip
    - Scale observations to [0, 1] (optional)
    - Stack the last `frame_stack` frames along the channel dimension
    """

    def _thunk():
        env = gym.make(env_name, render_mode=render_mode)
        env.reset(seed=seed)
        env.action_space.seed(seed)

        env = AtariPreprocessing(
            env,
            frame_skip=frame_skip,
            grayscale_obs=grayscale,
            scale_obs=scale_obs,
        )
        if frame_stack > 1:
            env = FrameStackObservation(env, frame_stack)
        return env

    return _thunk()


class ActorCriticNet(nn.Module):
    """
    CNN-based Actor-Critic network for Atari Pong.

    - Convolutional torso processes stacked frames.
    - Shared fully connected layer.
    - Policy head outputs action logits.
    - Value head outputs a scalar state value.
    """

    def __init__(self, input_shape, num_actions, fc_size: int = 512):
        super().__init__()

        # input_shape is (C, H, W)
        c, h, w = input_shape
        assert h == 84 and w == 84, (
            f"Expected 84x84 observations after preprocessing, got {h}x{w}"
        )

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # For 84x84 input with the above conv stack, the feature map is 7x7x64.
        conv_output_size = 64 * 7 * 7

        self.fc = nn.Linear(conv_output_size, fc_size)
        self.policy_head = nn.Linear(fc_size, num_actions)
        self.value_head = nn.Linear(fc_size, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for Atari Pong using a single environment.

    This implementation uses full-episode rollouts:
    - Collects (state, action, reward, value, log_prob) for an episode.
    - Computes discounted returns and advantages (return - value).
    - Updates the shared actor-critic network with:
        loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    """

    def __init__(self, config: dict):
        self.cfg = config

        # Environment & preprocessing settings
        env_name = config["env_name"]
        seed = config.get("seed", 0)
        frame_skip = config.get("frame_skip", 4)
        frame_stack = config.get("frame_stack", 4)
        grayscale = config.get("grayscale", True)
        scale_obs = config.get("scale_obs", True)

        # Training hyperparameters
        self.gamma = config.get("gamma", 0.99)
        # n-step horizon for returns/advantages; 0 or None = full-episode
        self.n_step = config.get("n_step", 0)
        self.learning_rate = config.get("learning_rate", 2.5e-4)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.log_interval = config.get("log_interval", 10)
        self.render_every = config.get("render_every", 0)  # 0 = never

        # Create environments
        self.env = make_pong_env(
            env_name=env_name,
            seed=seed,
            render_mode=None,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            grayscale=grayscale,
            scale_obs=scale_obs,
        )

        self.render_env = None
        if self.render_every and self.render_every > 0:
            self.render_env = make_pong_env(
                env_name=env_name,
                seed=seed,
                render_mode="human",
                frame_skip=frame_skip,
                frame_stack=frame_stack,
                grayscale=grayscale,
                scale_obs=scale_obs,
            )

        obs_shape = self.env.observation_space.shape  # (C, H, W)
        num_actions = self.env.action_space.n

        print(f"Observation shape: {obs_shape}")
        print(f"Number of actions: {num_actions}")

        fc_size = config.get("fc_size", 512)
        self.model = ActorCriticNet(obs_shape, num_actions, fc_size=fc_size).to(device)
        # Use RMSprop as in many classic Atari A2C implementations
        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

    def select_action(self, state):
        """
        Given a single observation (LazyFrames or np.array), return:
        - action (int)
        - log_prob of the action
        - value estimate (tensor)
        - entropy of the policy distribution
        """
        # FrameStack returns LazyFrames; convert to numpy array
        state_arr = np.array(state, copy=False)

        if state_arr.ndim == 3:
            # (C, H, W)
            state_tensor = torch.from_numpy(state_arr).float().unsqueeze(0).to(device)
        else:
            raise ValueError(f"Unexpected state shape: {state_arr.shape}")

        logits, value = self.model(state_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return (
            int(action.item()),
            log_prob.squeeze(0),
            value.squeeze(0),
            entropy.squeeze(0),
        )

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Compute discounted returns and advantages for a single episode.

        rewards: list of scalars (len T)
        values: list of value tensors (len T)
        """
        T = len(rewards)
        returns = torch.zeros(T, dtype=torch.float32, device=device)
        values_tensor = torch.stack(values).squeeze(-1)  # (T,)

        # If n_step <= 0, fall back to full-episode returns
        if not self.n_step or self.n_step <= 0:
            R = 0.0
            for t in reversed(range(T)):
                R = rewards[t] + self.gamma * R
                returns[t] = R
        else:
            n = int(self.n_step)
            for t in range(T):
                R = 0.0
                discount = 1.0
                done_in_window = False

                # Accumulate up to n rewards or until a done is hit
                for k in range(n):
                    idx = t + k
                    if idx >= T:
                        break
                    R += discount * rewards[idx]
                    discount *= self.gamma
                    if dones[idx]:
                        done_in_window = True
                        break

                # Bootstrap from V(s_{t+n}) if the episode hasn't ended within the window
                idx_bootstrap = t + n
                if (not done_in_window) and (idx_bootstrap < T):
                    R += discount * values_tensor[idx_bootstrap].detach()

                returns[t] = R

        advantages = returns - values_tensor
        return returns, advantages

    def update(self, log_probs, values, rewards, entropies, dones):
        """
        Perform a single A2C update given one episode of experience.
        """
        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)

        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        # Keep a copy of the *raw* advantages for the value loss
        raw_advantages = advantages

        # Normalize advantages for stability (policy loss only), guarding against
        # single-sample episodes where std() can become NaN.
        adv_mean = advantages.mean()
        if advantages.numel() > 1:
            adv_std = advantages.std(unbiased=False)
        else:
            adv_std = torch.tensor(1.0, device=advantages.device)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        policy_loss = -(log_probs_tensor * advantages.detach()).mean()
        # Critic should learn on the true scale of (returns - values), not normalized
        value_loss = raw_advantages.pow(2).mean()
        entropy_mean = entropies_tensor.mean()

        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy_mean
        )

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            loss.item(),
            policy_loss.item(),
            value_loss.item(),
            entropy_mean.item(),
        )

    def train(self, num_episodes: int, max_steps_per_episode: int):
        """
        Train the A2C agent for a given number of episodes.

        Returns:
            all_episode_rewards: list of per-episode returns.
            total_steps: total environment steps collected across training.
        """
        all_episode_rewards: list[float] = []
        total_steps: int = 0

        for episode in range(1, num_episodes + 1):
            # Choose whether to render this episode
            env = self.env
            render = False
            if self.render_env is not None and self.render_every:
                if episode % self.render_every == 0:
                    env = self.render_env
                    render = True

            state, _ = env.reset()

            episode_reward = 0.0
            episode_steps = 0

            # --- n-step update mode ---
            if self.n_step and self.n_step > 0:
                while episode_steps < max_steps_per_episode:
                    log_probs = []
                    values = []
                    rewards = []
                    entropies = []
                    dones = []

                    # Collect up to n_step transitions
                    for _ in range(self.n_step):
                        action, log_prob, value, entropy = self.select_action(state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = bool(terminated or truncated)

                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(float(reward))
                        entropies.append(entropy)
                        dones.append(done)

                        episode_reward += float(reward)
                        episode_steps += 1
                        total_steps += 1
                        state = next_state

                        if render:
                            env.render()

                        if done or episode_steps >= max_steps_per_episode:
                            break

                    # Bootstrap value from next state (0 if terminal)
                    with torch.no_grad():
                        if done or episode_steps >= max_steps_per_episode:
                            next_value = torch.zeros(1, device=device)
                        else:
                            state_arr = np.array(state, copy=False)
                            state_tensor = (
                                torch.from_numpy(state_arr)
                                .float()
                                .unsqueeze(0)
                                .to(device)
                            )
                            _, v_next = self.model(state_tensor)
                            next_value = v_next.squeeze(0)

                    # Compute n-step returns for this chunk
                    R = next_value
                    returns = []
                    for r, d in zip(reversed(rewards), reversed(dones)):
                        r_t = torch.tensor(r, dtype=torch.float32, device=device)
                        mask = 0.0 if d else 1.0
                        R = r_t + self.gamma * R * mask
                        returns.insert(0, R)

                    returns_tensor = torch.stack(returns)
                    values_tensor = torch.stack(values).squeeze(-1)
                    advantages = returns_tensor - values_tensor

                    # Use the same update logic as in self.update, but with our chunk
                    log_probs_tensor = torch.stack(log_probs)
                    entropies_tensor = torch.stack(entropies)

                    raw_advantages = advantages
                    adv_mean = advantages.mean()
                    if advantages.numel() > 1:
                        adv_std = advantages.std(unbiased=False)
                    else:
                        adv_std = torch.tensor(1.0, device=advantages.device)
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                    policy_loss = -(log_probs_tensor * advantages.detach()).mean()
                    value_loss = raw_advantages.pow(2).mean()
                    entropy_mean = entropies_tensor.mean()

                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_mean
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm is not None and self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()

                    if done or episode_steps >= max_steps_per_episode:
                        break

            # --- full-episode (Monte Carlo) update mode ---
            else:
                log_probs = []
                values = []
                rewards = []
                entropies = []
                dones = []

                for _ in range(max_steps_per_episode):
                    action, log_prob, value, entropy = self.select_action(state)

                    next_state, reward, terminated, truncated, _ = env.step(action)

                    done = bool(terminated or truncated)

                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(float(reward))
                    entropies.append(entropy)
                    dones.append(done)

                    episode_reward += float(reward)
                    state = next_state
                    episode_steps += 1
                    total_steps += 1

                    if render:
                        env.render()

                    if done:
                        break

                (
                    total_loss,
                    policy_loss,
                    value_loss,
                    entropy_mean,
                ) = self.update(log_probs, values, rewards, entropies, dones)

            all_episode_rewards.append(episode_reward)

            if episode % self.log_interval == 0:
                avg_reward = np.mean(all_episode_rewards[-50:])
                # In n-step mode, we don't have per-episode loss stats; print reward info.
                if self.n_step and self.n_step > 0:
                    print(
                        f"Episode {episode}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg(50): {avg_reward:.2f}",
                        flush=True,
                    )
                else:
                    print(
                        f"Episode {episode}/{num_episodes} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Avg(50): {avg_reward:.2f} | "
                        f"Loss: {total_loss:.4f} | "
                        f"Policy: {policy_loss:.4f} | "
                        f"Value: {value_loss:.4f} | "
                        f"Entropy: {entropy_mean:.4f}",
                        flush=True,
                    )

        return all_episode_rewards, total_steps


def run_from_config(config_path: str):
    """
    Load a JSON config and run A2C training for Pong.
    """
    config_name = os.path.splitext(os.path.basename(config_path))[0]


    print(f"\n{'#' * 80}")
    print(f"# Processing config: {config_name}")
    print(f"# Processing from device: {device}")
    print(f"{'#' * 80}\n")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    print("Loaded A2C config:")
    print(json.dumps(cfg, indent=2))
    print()

    # Set random seeds for reproducibility
    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_episodes = cfg.get("num_episodes", 1000)
    max_steps = cfg.get("max_steps", 10000)

    print(f"\n{'=' * 60}")
    print(f"A2C Training on {cfg['env_name']}")
    print(f"Config name: {config_name}")
    print(f"Episodes: {num_episodes}, Max Steps/Episode: {max_steps}")
    print(f"{'=' * 60}\n")

    agent = A2CAgent(cfg)

    start_time = time.time()
    episode_rewards, total_steps = agent.train(
        num_episodes=num_episodes, max_steps_per_episode=max_steps
    )
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print(
        f"Training completed in {elapsed_time:.2f} seconds "
        f"({elapsed_time / 60:.2f} minutes)"
    )
    print(f"{'=' * 60}\n")

    # Compute summary statistics
    average_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    min_reward = float(np.min(episode_rewards))

    print(f"Average Reward: {average_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")

    # Common timestamp for results + checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model checkpoint
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode": num_episodes,
        "total_steps": int(total_steps),
        "config": cfg,
    }
    checkpoint_path = os.path.join(
        checkpoints_dir, f"a2c_{config_name}_{timestamp}.pt"
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Save results in a structure similar to your Lunar Lander scripts
    results_base_dir = os.path.join(script_dir, "results", config_name)
    os.makedirs(results_base_dir, exist_ok=True)
    folder_name = f"a2c_avg_{average_reward:.2f}_{timestamp}"
    results_dir = os.path.join(results_base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nSaving results to: {results_dir}")

    # Copy config file
    config_dest = os.path.join(results_dir, f"{config_name}.json")
    with open(config_path, "r") as src, open(config_dest, "w") as dst:
        dst.write(src.read())
    print(f"Config file saved to: {config_dest}")

    # Save rewards
    rewards_file = os.path.join(results_dir, "episode_rewards.txt")
    np.savetxt(rewards_file, np.array(episode_rewards, dtype=np.float32), fmt="%.6f")

    # Save summary
    summary_file = os.path.join(results_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("A2C Training Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Algorithm: A2C (Advantage Actor-Critic)\n")
        f.write(f"Config: {config_name}\n")
        f.write(f"Environment: {cfg['env_name']}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Max Steps per Episode: {max_steps}\n")
        f.write(
            f"Training Time: {elapsed_time:.2f} seconds "
            f"({elapsed_time / 60:.2f} minutes)\n\n"
        )
        f.write(f"Average Reward: {average_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")
        f.write(f"Max Reward: {max_reward:.2f}\n")
        f.write(f"Min Reward: {min_reward:.2f}\n\n")
        f.write("Key Hyperparameters:\n")
        f.write(f"  Learning Rate: {cfg.get('learning_rate', 2.5e-4)}\n")
        f.write(f"  Gamma: {cfg.get('gamma', 0.99)}\n")
        f.write(f"  Entropy Coef: {cfg.get('entropy_coef', 0.01)}\n")
        f.write(f"  Value Loss Coef: {cfg.get('value_loss_coef', 0.5)}\n")
        f.write(f"  Max Grad Norm: {cfg.get('max_grad_norm', 0.5)}\n")
        f.write(f"  FC Size: {cfg.get('fc_size', 512)}\n")
        f.write(f"  Frame Skip: {cfg.get('frame_skip', 4)}\n")
        f.write(f"  Frame Stack: {cfg.get('frame_stack', 4)}\n")
        f.write(f"  Grayscale: {cfg.get('grayscale', True)}\n")
        f.write(f"  Scale Obs: {cfg.get('scale_obs', True)}\n")

    print(f"Summary saved to: {summary_file}")

    # Create training plots (same style as Lunar Lander algorithms)
    pong_plots.plot_training_curves(
        results_dir=results_dir,
        episode_rewards=episode_rewards,
        cfg=cfg,
        config_name=config_name,
        algo_name="A2C",
    )

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 60}\n")

    return {
        "config_name": config_name,
        "average_reward": average_reward,
        "std_reward": std_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "elapsed_time": elapsed_time,
    }


if __name__ == "__main__":
    # Find all A2C Pong configs in the project root that match this pattern
    config_pattern = os.path.join(script_dir, "global_config-pong-a2c*.json")
    config_files = sorted(glob.glob(config_pattern))

    if not config_files:
        print(f"No config files found matching '{os.path.basename(config_pattern)}'")
        print("Create at least one JSON config, e.g. global_config-pong-a2c.json")
        raise SystemExit(1)

    print(f"\n{'=' * 80}")
    print(f"Found {len(config_files)} configuration file(s) to process:")
    for cf in config_files:
        print(f"  - {os.path.basename(cf)}")
    print(f"{'=' * 80}\n")

    all_results = []
    for config_path in config_files:
        result = run_from_config(config_path)
        all_results.append(result)

    # Final summary across all configs
    print(f"\n{'#' * 80}")
    print("# A2C Pong - All Configurations Summary")
    print(f"{'#' * 80}\n")
    print(f"{'Config Name':<40} {'Avg Reward':>12} {'Std':>10} {'Time (min)':>12}")
    print("-" * 80)
    for result in all_results:
        print(
            f"{result['config_name']:<40} "
            f"{result['average_reward']:>12.2f} "
            f"{result['std_reward']:>10.2f} "
            f"{result['elapsed_time'] / 60:>12.2f}"
        )

    print(f"\n{'=' * 80}")
    print("All A2C runs complete! Results saved in results/ directory.")
    print(f"{'=' * 80}\n")


