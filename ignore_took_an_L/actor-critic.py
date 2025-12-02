import os
import json
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


# Register ALE Atari environments with Gymnasium
gym.register_envs(ale_py)

script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_pong_env(
    env_name: str,
    seed: int,
    render_mode: str | None = None,
    frame_skip: int = 4,
    frame_stack: int = 4,
    grayscale: bool = True,
    scale_obs: bool = True,
):
    """Create a Pong environment with Atari preprocessing and frame stacking."""

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
    CNN-based Actor-Critic network whose architecture is driven by config:

    - `conv_layers`: list of dicts with out_channels, kernel_size, stride.
    - `fc_hid_layers`: list of fully-connected hidden sizes (e.g. [512]).
    """

    def __init__(self, input_shape, num_actions, conv_layers, fc_hid_layers):
        super().__init__()

        c, h, w = input_shape
        assert h == 84 and w == 84, f"Expected 84x84 input, got {h}x{w}"

        conv_modules = []
        in_channels = c
        for i, layer_cfg in enumerate(conv_layers):
            out_channels = layer_cfg["out_channels"]
            kernel_size = layer_cfg["kernel_size"]
            stride = layer_cfg["stride"]
            conv_modules.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            )
            conv_modules.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*conv_modules)

        # Infer conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy).view(1, -1).shape[1]

        fc_modules = []
        prev_dim = conv_out
        for hidden_dim in fc_hid_layers:
            fc_modules.append(nn.Linear(prev_dim, hidden_dim))
            fc_modules.append(nn.ReLU())
            prev_dim = hidden_dim
        self.fc = nn.Sequential(*fc_modules)

        self.policy_head = nn.Linear(prev_dim, num_actions)
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class ActorCriticAgent:
    """
    Advantage Actor-Critic agent with n-step returns and configurable
    training frequency, driven by a JSON config.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        env_name = cfg["env_name"]
        seed = cfg.get("seed", 0)

        frame_skip = cfg.get("frame_skip", 4)
        frame_stack = cfg.get("frame_stack", 4)
        grayscale = cfg.get("grayscale", True)
        scale_obs = cfg.get("scale_obs", True)

        # Hyperparameters
        self.gamma = cfg.get("gamma", 0.99)
        self.n_step = cfg.get("num_step_returns", 5)
        self.train_freq = cfg.get("training_frequency", self.n_step)
        self.learning_rate = cfg.get("learning_rate", 7e-4)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.get("max_grad_norm", 40.0)
        self.log_interval = cfg.get("log_interval", 10)
        self.render_every = cfg.get("render_every", 0)
        self.max_frame = cfg.get("max_frame", None)
        self.reward_scale = cfg.get("reward_scale", None)
        self.num_envs = cfg.get("num_envs", 1)
        self.log_frequency = cfg.get("log_frequency", None)
        self.eval_frequency = cfg.get("eval_frequency", None)

        if self.num_envs != 1:
            print(
                f"Warning: config requests num_envs={self.num_envs}, "
                "but this implementation currently uses a single env.",
                flush=True,
            )

        # Environments
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

        obs_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n

        print(f"Observation shape: {obs_shape}")
        print(f"Number of actions: {num_actions}")

        conv_layers = cfg.get("conv_layers", [])
        fc_hid_layers = cfg.get("fc_hid_layers", [512])

        self.model = ActorCriticNet(
            obs_shape, num_actions, conv_layers, fc_hid_layers
        ).to(device)

        self.optimizer = optim.RMSprop(
            self.model.parameters(),
            lr=self.learning_rate,
            alpha=0.99,
            eps=1e-5,
        )

    def select_action(self, state):
        state_arr = np.array(state, copy=False)
        state_tensor = (
            torch.from_numpy(state_arr).float().unsqueeze(0).to(device)
        )  # (1, C, H, W)

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

    def train(self, num_episodes: int, max_steps_per_episode: int):
        all_episode_rewards: list[float] = []
        total_steps = 0

        next_log_step = self.log_frequency if self.log_frequency else None

        for episode in range(1, num_episodes + 1):
            env = self.env
            render = False
            if self.render_env is not None and self.render_every:
                if episode % self.render_every == 0:
                    env = self.render_env
                    render = True

            state, _ = env.reset()
            done = False

            episode_reward = 0.0
            episode_steps = 0
            last_entropy_mean = 0.0

            while not done and episode_steps < max_steps_per_episode:
                log_probs = []
                values = []
                rewards = []
                entropies = []
                dones = []

                # Collect up to training_frequency steps
                for _ in range(self.train_freq):
                    action, log_prob, value, entropy = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)

                    # Optional reward scaling (e.g., "sign")
                    if self.reward_scale == "sign":
                        reward = float(np.sign(reward))

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

                # Step-based logging, if configured
                if next_log_step is not None and total_steps >= next_log_step:
                    avg_reward = (
                        np.mean(all_episode_rewards[-50:])
                        if all_episode_rewards
                        else episode_reward
                    )
                    print(
                        f"[StepLog] steps={total_steps} | "
                        f"episode={episode} | "
                        f"reward={episode_reward:.2f} | "
                        f"Avg(50)={avg_reward:.2f}",
                        flush=True,
                    )
                    next_log_step += self.log_frequency

                # Global frame budget
                if self.max_frame is not None and total_steps >= self.max_frame:
                    done = True

                # Bootstrap from next state
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

                # Compute returns (n-step truncated within this chunk)
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

                log_probs_tensor = torch.stack(log_probs)
                entropies_tensor = torch.stack(entropies)

                # Advantage normalization with clamped std
                raw_advantages = advantages
                adv_mean = advantages.mean()
                if advantages.numel() > 1:
                    adv_std = advantages.std(unbiased=False)
                    adv_std = torch.clamp(adv_std, min=0.1)
                else:
                    adv_std = torch.tensor(1.0, device=advantages.device)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                policy_loss = -(log_probs_tensor * advantages.detach()).mean()
                value_loss = raw_advantages.pow(2).mean()
                entropy_mean = entropies_tensor.mean()
                last_entropy_mean = float(entropy_mean.item())

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

            all_episode_rewards.append(episode_reward)

            if episode % self.log_interval == 0:
                avg_reward = np.mean(all_episode_rewards[-50:])
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Avg(50): {avg_reward:.2f} | "
                    f"Entropy: {last_entropy_mean:.4f}",
                    flush=True,
                )

        return all_episode_rewards, total_steps


def run_from_config(config_path: str):
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    print(f"\n{'#' * 80}")
    print(f"# Actor-Critic - Processing config: {config_name}")
    print(f"# Device: {device}")
    print(f"{'#' * 80}\n")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    print("Loaded config:")
    print(json.dumps(cfg, indent=2))
    print()

    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    num_episodes = cfg.get("num_episodes", 1000)
    max_steps = cfg.get("max_steps", 10000)

    print(f"\n{'=' * 60}")
    print(f"Actor-Critic Training on {cfg['env_name']}")
    print(f"Config name: {config_name}")
    print(f"Episodes: {num_episodes}, Max Steps/Episode: {max_steps}")
    print(f"{'=' * 60}\n")

    agent = ActorCriticAgent(cfg)

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
    print(f"Total steps: {total_steps}")
    print(f"{'=' * 60}\n")

    average_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    max_reward = float(np.max(episode_rewards))
    min_reward = float(np.min(episode_rewards))

    print(f"Average Reward: {average_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")

    # Save results and config
    results_base_dir = os.path.join(script_dir, "results", config_name)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"actor_critic_avg_{average_reward:.2f}_{timestamp}"
    results_dir = os.path.join(results_base_dir, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nSaving results to: {results_dir}")

    # Copy config
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
        f.write("Actor-Critic Training Summary\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Algorithm: Actor-Critic (n-step)\n")
        f.write(f"Config: {config_name}\n")
        f.write(f"Environment: {cfg['env_name']}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"Max Steps per Episode: {max_steps}\n")
        f.write(
            f"Training Time: {elapsed_time:.2f} seconds "
            f"({elapsed_time / 60:.2f} minutes)\n\n"
        )
        f.write(f"Total Steps: {total_steps}\n\n")
        f.write(f"Average Reward: {average_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")
        f.write(f"Max Reward: {max_reward:.2f}\n")
        f.write(f"Min Reward: {min_reward:.2f}\n\n")
        f.write("Key Hyperparameters:\n")
        f.write(f"  Learning Rate: {cfg.get('learning_rate', 7e-4)}\n")
        f.write(f"  Gamma: {cfg.get('gamma', 0.99)}\n")
        f.write(f"  Num Step Returns: {cfg.get('num_step_returns', 11)}\n")
        f.write(f"  Training Frequency: {cfg.get('training_frequency', 5)}\n")
        f.write(f"  Entropy Coef: {cfg.get('entropy_coef', 0.01)}\n")
        f.write(f"  Value Loss Coef: {cfg.get('value_loss_coef', 0.5)}\n")
        f.write(f"  Max Grad Norm: {cfg.get('max_grad_norm', 40.0)}\n")
        f.write(f"  FC Layers: {cfg.get('fc_hid_layers', [512])}\n")
        f.write(f"  Conv Layers: {cfg.get('conv_layers', [])}\n")

    print(f"Summary saved to: {summary_file}")
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # For now just look for a single config file
    config_path = os.path.join(script_dir, "global_config-pong-actor-critic.json")
    if not os.path.isfile(config_path):
        print(
            "Config file 'global_config-pong-actor-critic.json' not found.\n"
            "Please create it or adjust the path in actor-critic.py."
        )
        raise SystemExit(1)

    run_from_config(config_path)


