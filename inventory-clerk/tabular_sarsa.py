"""
Pure tabular SARSA training for the Retail Store Clerk environment.

This version uses a NumPy Q-table with one entry per discrete state-action
pair. It is useful as a baseline against the neural SARSA and A2C agents.
"""

import os
import json
import glob
import time
from datetime import datetime
import shutil

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import retail_store_env  # noqa: F401  - registers the environment


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SARSAAgentTabular:
    """
    Tabular SARSA agent for discrete state and action spaces.
    Uses a simple Q-table instead of function approximation.
    """

    def __init__(self, config: dict):
        # Allow toggling dynamic customers via config (default: enabled)
        enable_customers = config.get("enable_customers", True)

        self.env = gym.make(
            config["env_name"],
            render_mode=None,
            enable_customers=enable_customers,
        )
        self.render_env = gym.make(
            config["env_name"],
            render_mode="human",
            enable_customers=enable_customers,
        )

        # Environment exposes a Dict observation but provides a compact
        # integer state index via env.unwrapped.state_index_size and
        # info["state_index"].
        self.state_dim = self.env.unwrapped.state_index_size
        self.action_dim = self.env.action_space.n

        # Q-table: [state_dim, action_dim]
        self.q_table = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)

        # Hyperparameters
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.epsilon = config["initial_epsilon"]
        self.min_epsilon = config["min_epsilon"]
        self.decay_rate = config["decay_rate"]
        self.use_boltzmann = config.get("use_boltzmann", False)
        self.display_episodes = config.get("display_episodes", 2001)

        print(f"State space size: {self.state_dim}")
        print(f"Action space size: {self.action_dim}")
        print(
            f"Using {'Boltzmann' if self.use_boltzmann else 'Epsilon-greedy'} exploration"
        )

    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy or Boltzmann exploration."""
        q_values = self.q_table[state]  # shape [action_dim]

        if self.use_boltzmann:
            # Boltzmann exploration
            temperature = max(self.epsilon, 0.1)
            q_shifted = q_values - np.max(q_values)
            exp_q = np.exp(q_shifted / temperature)
            probs = exp_q / np.sum(exp_q)
            action = np.random.choice(self.action_dim, p=probs)
        else:
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = int(np.argmax(q_values))

        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> float:
        """
        Standard 1-step SARSA update:
        Q(s,a) <- Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ].
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            next_q = self.q_table[next_state, next_action]
            target = reward + self.gamma * next_q

        td_error = target - current_q
        self.q_table[state, action] += self.learning_rate * td_error

        return float(abs(td_error))

    def train(self, num_episodes: int, max_steps: int):
        """Train the agent using SARSA algorithm."""
        all_episode_rewards = []
        success_count = 0

        start_time = time.time()
        print_every = 10

        for episode in range(num_episodes):
            # Use render environment for display episodes
            env = (
                self.render_env
                if (episode + 1) % self.display_episodes == 0
                else self.env
            )

            # Reset environment
            obs, info = env.reset()
            # Use the compact integer state index provided by the environment.
            state = info.get("state_index", obs)

            # Select initial action
            action = self.select_action(state)

            episode_total_reward = 0.0
            episode_had_full_success = False

            # Run episode
            for step in range(max_steps):
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = info.get("state_index", next_obs)

                next_action = self.select_action(next_state)

                episode_total_reward += reward

                # SARSA update
                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                    next_action,
                    terminated or truncated,
                )

                state = next_state
                action = next_action

                if (episode + 1) % self.display_episodes == 0:
                    env.render()

                if terminated or truncated:
                    if info.get("all_items_stocked"):
                        success_count += 1
                        episode_had_full_success = True
                    break

            # Decay epsilon
            self.epsilon = max(
                self.min_epsilon, self.epsilon * np.exp(-self.decay_rate)
            )

            all_episode_rewards.append(episode_total_reward)

            # If this episode successfully stocked all items, run a separate
            # greedy rendering episode using the current Q-table so we can
            # visually inspect the behavior.
            if episode_had_full_success:
                print(
                    "All items stocked in this episode; rendering a greedy "
                    "demo episode with the learned Q-table..."
                )
                self._render_greedy_episode(max_steps)

            # Progress print every 10 episodes with ETA
            if (
                (episode + 1) % print_every == 0
                or episode == 0
                or (episode + 1) == num_episodes
            ):
                avg_reward = float(np.mean(all_episode_rewards[-50:]))
                success_rate = success_count / (episode + 1) * 100.0

                elapsed = time.time() - start_time
                completed = episode + 1
                remaining = max(num_episodes - completed, 0)
                eta_seconds = (elapsed / completed) * remaining if completed > 0 else 0.0
                eta_minutes = eta_seconds / 60.0

                print(
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_total_reward:.2f} | "
                    f"Avg (50): {avg_reward:.2f} | "
                    f"ε: {self.epsilon:.4f} | "
                    f"Success Rate: {success_rate:.1f}% | "
                    f"ETA: {eta_minutes:.2f} min",
                    flush=True,
                )

        return all_episode_rewards, success_count

    def _render_greedy_episode(self, max_steps: int):
        """
        Run a single greedy episode using the current Q-table in the
        dedicated render environment, rendering every step.
        This does not update the Q-table (pure evaluation).
        """
        env = self.render_env
        obs, info = env.reset()
        state = info.get("state_index", obs)

        for _ in range(max_steps):
            q_values = self.q_table[state]
            action = int(np.argmax(q_values))
            next_obs, _, terminated, truncated, info = env.step(action)
            env.render()

            state = info.get("state_index", next_obs)
            if terminated or truncated:
                break


if __name__ == "__main__":
    # Default configuration for Retail Store environment
    default_config = {
        "env_name": "RetailStore-v0",
        "num_episodes": 1000,
        "max_steps": 600,
        "gamma": 0.99,
        "learning_rate": 0.1,
        "initial_epsilon": 1.0,
        "min_epsilon": 0.01,
        "decay_rate": 0.005,
        "use_boltzmann": False,
        "display_episodes": 1001,
        "enable_customers": True,
        "seed": 42,
    }

    # Look for config files specific to retail store, or use default
    config_files = sorted(
        glob.glob(os.path.join(SCRIPT_DIR, "retail_store_config*.json"))
    )

    if not config_files:
        print("No retail store config files found. Using default configuration.")
        print("You can create retail_store_config.json to customize settings.\n")

        # Create default config file
        default_config_path = os.path.join(
            SCRIPT_DIR, "retail_store_config_default.json"
        )
        with open(default_config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        print(f"Default config saved to: {default_config_path}\n")

        config_files = [default_config_path]

    print(f"\n{'='*80}")
    print(f"Found {len(config_files)} configuration file(s) to process:")
    for cf in config_files:
        print(f"  - {os.path.basename(cf)}")
    print(f"{'='*80}\n")

    all_run_results = []

    # Process each config file
    for config_idx, config_path in enumerate(config_files):
        config_name = os.path.splitext(os.path.basename(config_path))[0]

        print(f"\n{'#'*80}")
        print(f"# Processing config {config_idx+1}/{len(config_files)}: {config_name}")
        print(f"{'#'*80}\n")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Set random seeds
        np.random.seed(cfg["seed"])

        print(f"\n{'='*60}")
        print(f"SARSA Training on {cfg['env_name']}")
        print(f"Config: {config_name}")
        print(f"{'='*60}\n")

        agent = SARSAAgentTabular(cfg)

        start_time = time.time()
        episode_rewards, success_count = agent.train(
            cfg["num_episodes"], cfg["max_steps"]
        )
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(
            f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        )
        print(f"{'='*60}\n")

        # Compute statistics
        average_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        max_reward = float(np.max(episode_rewards))
        min_reward = float(np.min(episode_rewards))
        success_rate = success_count / cfg["num_episodes"] * 100.0

        print(f"Average Reward: {average_reward:.2f}")
        print(f"Std Reward: {std_reward:.2f}")
        print(f"Max Reward: {max_reward:.2f}")
        print(f"Min Reward: {min_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")

        all_run_results.append(
            {
                "config_name": config_name,
                "average_reward": average_reward,
                "std_reward": std_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "success_rate": success_rate,
                "elapsed_time": elapsed_time,
            }
        )

        # Create results directory
        results_base_dir = os.path.join(SCRIPT_DIR, "results", "retail_store", config_name)
        os.makedirs(results_base_dir, exist_ok=True)

        # Create folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = (
            f"tabular_sarsa_avg_{average_reward:.2f}_success_{success_rate:.1f}_{timestamp}"
        )
        results_dir = os.path.join(results_base_dir, folder_name)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\nSaving results to: {results_dir}")

        # Copy config file
        config_dest = os.path.join(results_dir, f"{config_name}.json")
        shutil.copy(config_path, config_dest)

        # Save rewards
        rewards_file = os.path.join(results_dir, "episode_rewards.txt")
        np.savetxt(rewards_file, episode_rewards, fmt="%.6f")

        # Save Q-table
        qtable_file = os.path.join(results_dir, "q_table.npy")
        np.save(qtable_file, agent.q_table)
        print(f"Q-table saved to: {qtable_file}")

        # Save summary
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Retail Store Tabular SARSA Training Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Algorithm: SARSA (Tabular)\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Environment: {cfg['env_name']}\n")
            f.write(f"Episodes: {cfg['num_episodes']}\n")
            f.write(f"Max Steps: {cfg['max_steps']}\n")
            f.write(
                f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n\n"
            )
            f.write("Results:\n")
            f.write(f"  Average Reward: {average_reward:.2f}\n")
            f.write(f"  Std Reward: {std_reward:.2f}\n")
            f.write(f"  Max Reward: {max_reward:.2f}\n")
            f.write(f"  Min Reward: {min_reward:.2f}\n")
            f.write(f"  Success Rate: {success_rate:.2f}%\n")
            f.write(f"  Successful Episodes: {success_count}/{cfg['num_episodes']}\n\n")
            f.write("Hyperparameters:\n")
            f.write(f"  Learning Rate: {cfg['learning_rate']}\n")
            f.write(f"  Gamma: {cfg['gamma']}\n")
            f.write(f"  Initial Epsilon: {cfg['initial_epsilon']}\n")
            f.write(f"  Min Epsilon: {cfg['min_epsilon']}\n")
            f.write(f"  Decay Rate: {cfg['decay_rate']}\n")
            f.write(f"  Use Boltzmann: {cfg.get('use_boltzmann', False)}\n")

        # Create plots similar to the neural SARSA script
        episodes = np.arange(1, len(episode_rewards) + 1)

        plt.figure(figsize=(14, 5))

        # Episode rewards
        plt.subplot(1, 3, 1)
        plt.plot(
            episodes,
            episode_rewards,
            alpha=0.4,
            linewidth=0.8,
            label="Episode Reward",
        )
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(f"Tabular SARSA on Retail Store\n{config_name}", fontsize=11)
        plt.grid(alpha=0.3)
        plt.legend()

        # Moving average
        plt.subplot(1, 3, 2)
        window = 50
        if len(episode_rewards) >= window:
            weights = np.ones(window) / window
            moving_avg = np.convolve(episode_rewards, weights, mode="valid")
            ma_episodes = np.arange(window, len(episode_rewards) + 1)
            plt.plot(ma_episodes, moving_avg, linewidth=2, color="orange")
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Average Reward", fontsize=12)
            plt.title(f"{window}-Episode Moving Average", fontsize=11)
            plt.grid(alpha=0.3)

        # Cumulative success rate
        plt.subplot(1, 3, 3)
        successes = [1 if r >= 90.0 else 0 for r in episode_rewards]
        cumulative_success_rate = np.cumsum(successes) / episodes * 100.0
        plt.plot(episodes, cumulative_success_rate, linewidth=2, color="green")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.title("Cumulative Success Rate", fontsize=11)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(results_dir, "training_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Training plot saved to: {plot_file}")
        plt.close()

        # Learning curve with confidence interval
        plt.figure(figsize=(10, 6))
        if len(episode_rewards) >= window:
            moving_avg = []
            moving_std = []
            for i in range(window, len(episode_rewards) + 1):
                window_rewards = episode_rewards[i - window : i]
                moving_avg.append(np.mean(window_rewards))
                moving_std.append(np.std(window_rewards))

            moving_avg = np.array(moving_avg)
            moving_std = np.array(moving_std)
            ma_episodes = np.arange(window, len(episode_rewards) + 1)

            plt.plot(
                ma_episodes, moving_avg, linewidth=2, color="blue", label="Mean"
            )
            plt.fill_between(
                ma_episodes,
                moving_avg - moving_std,
                moving_avg + moving_std,
                alpha=0.2,
                color="blue",
                label="±1 Std",
            )
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Average Reward", fontsize=12)
            plt.title(
                f"Tabular SARSA Learning Curve ({window}-Episode Window)\n{config_name}",
                fontsize=13,
            )
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)

            learning_curve_file = os.path.join(results_dir, "learning_curve.png")
            plt.savefig(learning_curve_file, dpi=300, bbox_inches="tight")
            print(f"Learning curve saved to: {learning_curve_file}")
            plt.close()

        print(f"\n{'='*60}")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*60}\n")

    # Final summary across all configs
    print(f"\n{'#'*80}")
    print("# Retail Store Tabular SARSA - All Configurations Summary")
    print(f"{'#'*80}\n")
    print(
        f"{'Config Name':<40} {'Avg Reward':>12} {'Success %':>11} {'Time (min)':>12}"
    )
    print(f"{'-'*80}")
    for result in all_run_results:
        print(
            f"{result['config_name']:<40} "
            f"{result['average_reward']:>12.2f} "
            f"{result['success_rate']:>10.1f}% "
            f"{result['elapsed_time']/60:>12.2f}"
        )

    print(f"\n{'='*80}")
    print(
        "All Tabular SARSA runs complete! Results saved in results/retail_store/ directory."
    )
    print(f"{'='*80}\n")


