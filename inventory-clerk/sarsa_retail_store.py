"""
Neural SARSA (deep SARSA) training for the Retail Store Clerk environment.

This version uses a PyTorch neural network to approximate the action-value
function Q(s, a), and runs on GPU (CUDA) when available.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import time
import glob
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import retail_store_env  # Register the environment

script_dir = os.path.dirname(os.path.abspath(__file__))


class QNetwork(nn.Module):
    """
    Simple feedforward Q-network for discrete state indices and actions.
    We embed the integer state index into a learned vector and pass it
    through a small MLP to produce Q-values for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(state_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

        # Optional: initialize weights for stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: tensor of shape [batch] with integer state indices.
        Returns:
            q_values: tensor of shape [batch, action_dim]
        """
        x = self.embedding(states.long())
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values


class SARSAAgentTabular:
    """
    Neural SARSA agent for discrete state and action spaces.
    Uses a PyTorch Q-network instead of a tabular Q-table, and can
    train on GPU when available.
    """
    
    def __init__(self, config):
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
        
        # Select device: use GPU (cuda) if available, otherwise CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get state and action dimensions. The environment now exposes a Dict
        # observation, but also provides a compact integer state index via
        # env.unwrapped.state_index_size and info["state_index"].
        self.state_dim = self.env.unwrapped.state_index_size
        self.action_dim = self.env.action_space.n

        # Q-network and optimizer
        hidden_size = config.get("hidden_size", 128)
        self.q_network = QNetwork(self.state_dim, self.action_dim, hidden_size).to(
            self.device
        )
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config["learning_rate"]
        )
        
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
        print(f"Using {'Boltzmann' if self.use_boltzmann else 'Epsilon-greedy'} exploration")
        print(f"Using device: {self.device}")
    
    def select_action(self, state):
        """Select action using epsilon-greedy or Boltzmann exploration."""
        state_t = torch.tensor([state], device=self.device, dtype=torch.long)
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze(0)  # [action_dim]
        
        if self.use_boltzmann:
            # Boltzmann exploration using torch operations.
            temperature = max(self.epsilon, 0.1)
            q_shifted = q_values - torch.max(q_values)
            exp_q = torch.exp(q_shifted / temperature)
            probs = exp_q / torch.sum(exp_q)
            action = torch.multinomial(probs, 1).item()
        else:
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = torch.argmax(q_values).item()
        
        return action
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update with function approximation:
        Q(s,a;θ) <- Q(s,a;θ) + α * [r + γ Q(s',a';θ) - Q(s,a;θ)]
        """
        state_t = torch.tensor([state], device=self.device, dtype=torch.long)
        next_state_t = torch.tensor([next_state], device=self.device, dtype=torch.long)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        # Current Q(s,a)
        q_values = self.q_network(state_t)  # [1, action_dim]
        q_sa = q_values[0, action]
        
        # Target r + gamma * Q(s',a')
        with torch.no_grad():
            next_q_values = self.q_network(next_state_t)  # [1, action_dim]
            next_q_sap = next_q_values[0, next_action]
            target = reward_t if done else reward_t + self.gamma * next_q_sap
        
        loss = (q_sa - target).pow(2)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return float(loss.item())
    
    def train(self, num_episodes, max_steps):
        """Train the agent using SARSA algorithm."""
        all_episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            # Use render environment for display episodes
            env = self.render_env if (episode + 1) % self.display_episodes == 0 else self.env
            
            # Reset environment
            obs, info = env.reset()
            # Use the compact integer state index provided by the environment.
            state = info.get("state_index")
            
            # Select initial action
            action = self.select_action(state)
            
            episode_total_reward = 0
            
            # Run episode
            for step in range(max_steps):
                # Take action
                next_obs, reward, done, truncated, info = env.step(action)
                next_state = info.get("state_index")
                
                # Select next action
                next_action = self.select_action(next_state)
                
                episode_total_reward += reward
                
                # SARSA update
                self.update(state, action, reward, next_state, next_action, done or truncated)
                
                # Move to next state and action
                state = next_state
                action = next_action
                
                # Render if display episode
                if (episode + 1) % self.display_episodes == 0:
                    env.render()
                
                # Check if episode ended
                if done or truncated:
                    # Count episode as success when all items have been stocked.
                    # The environment exposes this via info["all_items_stocked"].
                    if info.get("all_items_stocked"):
                        success_count += 1
                    break
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate))
            
            # Store episode reward
            all_episode_rewards.append(episode_total_reward)
            
            # Print progress
            avg_reward = np.mean(all_episode_rewards[-50:])
            success_rate = success_count / (episode + 1) * 100
            
            print(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_total_reward:.2f} | "
                f"Avg (50): {avg_reward:.2f} | "
                f"ε: {self.epsilon:.4f} | "
                f"Success Rate: {success_rate:.1f}%",
                flush=True,
            )
        
        return all_episode_rewards, success_count


# --- Main execution ---
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
        "seed": 42
    }
    
    # Look for config files specific to retail store, or use default
    config_files = sorted(glob.glob(os.path.join(script_dir, "retail_store_config*.json")))
    
    if not config_files:
        print("No retail store config files found. Using default configuration.")
        print("You can create retail_store_config.json to customize settings.\n")
        
        # Create default config file
        default_config_path = os.path.join(script_dir, "retail_store_config_default.json")
        with open(default_config_path, "w") as f:
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
        
        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)
        
        # Set random seeds
        np.random.seed(cfg["seed"])
        
        print(f"\n{'='*60}")
        print(f"SARSA Training on {cfg['env_name']}")
        print(f"Config: {config_name}")
        print(f"{'='*60}\n")
        
        # Create agent
        agent = SARSAAgentTabular(cfg)
        
        # Train agent
        start_time = time.time()
        episode_rewards, success_count = agent.train(cfg["num_episodes"], cfg["max_steps"])
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"{'='*60}\n")
        
        # Compute statistics
        average_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        max_reward = np.max(episode_rewards)
        min_reward = np.min(episode_rewards)
        success_rate = success_count / cfg["num_episodes"] * 100
        
        print(f"Average Reward: {average_reward:.2f}")
        print(f"Std Reward: {std_reward:.2f}")
        print(f"Max Reward: {max_reward:.2f}")
        print(f"Min Reward: {min_reward:.2f}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        # Store results (for final multi-config summary)
        all_run_results.append({
            "config_name": config_name,
            "average_reward": average_reward,
            "std_reward": std_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
        })
        
        # Create results directory
        results_base_dir = os.path.join(script_dir, "results", "retail_store", config_name)
        os.makedirs(results_base_dir, exist_ok=True)
        
        # Create folder with timestamp and allow user to customize the name.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_folder_name = f"sarsa_avg_{average_reward:.2f}_success_{success_rate:.1f}_{timestamp}"
        user_folder_name = input(
            f'What do you want to name the folder that is going in results? '
            f'(press Enter for default "{default_folder_name}"): '
        ).strip()
        folder_name = user_folder_name or default_folder_name
        results_dir = os.path.join(results_base_dir, folder_name)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\nSaving results to: {results_dir}")
        
        # Copy config file
        config_dest = os.path.join(results_dir, f"{config_name}.json")
        shutil.copy(config_path, config_dest)
        
        # Save rewards data
        rewards_file = os.path.join(results_dir, "episode_rewards.txt")
        np.savetxt(rewards_file, episode_rewards, fmt='%.6f')
        
        # Save a snapshot of the Q-function for analysis / demo:
        # evaluate the network on all discrete states and store the resulting
        # Q-table as a NumPy array, and also save the network weights.
        qtable_file = os.path.join(results_dir, "q_table.npy")
        with torch.no_grad():
            all_states = torch.arange(
                agent.state_dim, device=agent.device, dtype=torch.long
            )
            all_q = agent.q_network(all_states).cpu().numpy()
        np.save(qtable_file, all_q)
        print(f"Q-table snapshot saved to: {qtable_file}")

        model_file = os.path.join(results_dir, "q_network.pt")
        torch.save(agent.q_network.state_dict(), model_file)
        print(f"Q-network weights saved to: {model_file}")
        
        # Save summary
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"Retail Store SARSA Training Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Algorithm: SARSA (Tabular)\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Environment: {cfg['env_name']}\n")
            f.write(f"Episodes: {cfg['num_episodes']}\n")
            f.write(f"Max Steps: {cfg['max_steps']}\n")
            f.write(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n\n")
            f.write(f"Results:\n")
            f.write(f"  Average Reward: {average_reward:.2f}\n")
            f.write(f"  Std Reward: {std_reward:.2f}\n")
            f.write(f"  Max Reward: {max_reward:.2f}\n")
            f.write(f"  Min Reward: {min_reward:.2f}\n")
            f.write(f"  Success Rate: {success_rate:.2f}%\n")
            f.write(f"  Successful Episodes: {success_count}/{cfg['num_episodes']}\n\n")
            f.write(f"Hyperparameters:\n")
            f.write(f"  Learning Rate: {cfg['learning_rate']}\n")
            f.write(f"  Gamma: {cfg['gamma']}\n")
            f.write(f"  Initial Epsilon: {cfg['initial_epsilon']}\n")
            f.write(f"  Min Epsilon: {cfg['min_epsilon']}\n")
            f.write(f"  Decay Rate: {cfg['decay_rate']}\n")
            f.write(f"  Use Boltzmann: {cfg.get('use_boltzmann', False)}\n")
        
        print(f"Summary saved to: {summary_file}")
        
        # Create plots
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        # Plot 1: Episode rewards with moving average
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(episodes, episode_rewards, alpha=0.4, linewidth=0.8, label='Episode Reward')
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(f"SARSA on Retail Store\n{config_name}", fontsize=11)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Plot 2: Moving average
        plt.subplot(1, 3, 2)
        window = 50
        if len(episode_rewards) >= window:
            weights = np.ones(window) / window
            moving_avg = np.convolve(episode_rewards, weights, mode='valid')
            ma_episodes = np.arange(window, len(episode_rewards) + 1)
            plt.plot(ma_episodes, moving_avg, linewidth=2, color='orange')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Average Reward", fontsize=12)
            plt.title(f"{window}-Episode Moving Average", fontsize=11)
            plt.grid(alpha=0.3)
        
        # Plot 3: Success rate over time
        plt.subplot(1, 3, 3)
        # Calculate cumulative success rate
        successes = [1 if r >= 900 else 0 for r in episode_rewards]
        cumulative_success_rate = np.cumsum(successes) / episodes * 100
        plt.plot(episodes, cumulative_success_rate, linewidth=2, color='green')
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Success Rate (%)", fontsize=12)
        plt.title("Cumulative Success Rate", fontsize=11)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, "training_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_file}")
        plt.close()
        
        # Plot learning curve with confidence interval
        plt.figure(figsize=(10, 6))
        window = 50
        if len(episode_rewards) >= window:
            moving_avg = []
            moving_std = []
            for i in range(window, len(episode_rewards) + 1):
                window_rewards = episode_rewards[i-window:i]
                moving_avg.append(np.mean(window_rewards))
                moving_std.append(np.std(window_rewards))
            
            moving_avg = np.array(moving_avg)
            moving_std = np.array(moving_std)
            ma_episodes = np.arange(window, len(episode_rewards) + 1)
            
            plt.plot(ma_episodes, moving_avg, linewidth=2, color='blue', label='Mean')
            plt.fill_between(ma_episodes, 
                             moving_avg - moving_std, 
                             moving_avg + moving_std, 
                             alpha=0.2, color='blue', label='±1 Std')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Average Reward", fontsize=12)
            plt.title(f"Learning Curve ({window}-Episode Window)\n{config_name}", fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            
            # Save learning curve
            learning_curve_file = os.path.join(results_dir, "learning_curve.png")
            plt.savefig(learning_curve_file, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to: {learning_curve_file}")
            plt.close()
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*60}\n")
    
    # Print final summary
    print(f"\n{'#'*80}")
    print(f"# Retail Store SARSA - All Configurations Summary")
    print(f"{'#'*80}\n")
    
    print(f"{'Config Name':<40} {'Avg Reward':>12} {'Success %':>11} {'Time (min)':>12}")
    print(f"{'-'*80}")
    for result in all_run_results:
        print(f"{result['config_name']:<40} {result['average_reward']:>12.2f} "
              f"{result['success_rate']:>10.1f}% {result['elapsed_time']/60:>12.2f}")
    
    print(f"\n{'='*80}")
    print(f"All SARSA runs complete! Results saved in results/retail_store/ directory.")
    print(f"{'='*80}\n")


