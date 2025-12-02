import os, json, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import gymnasium as gym
import numpy as np, matplotlib.pyplot as plt
import shutil
from datetime import datetime
import time
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Policy network definition with softmax output ---
class PolicyNet(nn.Module):
    """
    Policy network that outputs action probabilities via a softmax layer.
    Architecture is configurable via hidden_layers and activation_function parameters.
    """
    def __init__(self, state_dim, action_dim, hidden_layers, activation_function):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        
        self.layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, action_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = self.act_fn(layer(x))
        
        # Output action probabilities via softmax
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1)


# --- REINFORCE Agent (Policy Gradient) ---
class REINFORCEAgent:
    """
    REINFORCE (Policy Gradient) algorithm implementation.
    Supports:
    - Epsilon-greedy or Boltzmann sampling for exploration
    - Optional baseline to reduce variance
    - Configurable network architecture
    - Gamma (discount factor) and learning rate scheduling
    """
    def __init__(self, config):
        self.env = gym.make(config["env_name"], render_mode=None)
        self.render_env = gym.make(config["env_name"], render_mode="human")
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Hyperparameters from config
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.epsilon = config["initial_epsilon"]
        self.min_epsilon = config["min_epsilon"]
        self.decay_rate = config["decay_rate"]
        self.use_boltzmann = config.get("use_boltzmann", False)
        self.use_baseline = config.get("use_baseline", True)
        self.display_episodes = config.get("display_episodes", 2001)
        
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Using {'Boltzmann' if self.use_boltzmann else 'Epsilon-greedy'} exploration")
        print(f"Using baseline: {self.use_baseline}")
        
        # Policy network
        self.policy = PolicyNet(
            self.state_dim, 
            self.action_dim, 
            config["hidden_layers"], 
            config["activation_function"]
        ).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Optional baseline network (value function approximator)
        if self.use_baseline:
            self.baseline = nn.Linear(self.state_dim, 1).to(device)
            self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=self.learning_rate)
    
    def select_action(self, state):
        """
        Select action using either epsilon-greedy or Boltzmann sampling.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs = self.policy(state_tensor).cpu().numpy().flatten()
        
        if self.use_boltzmann:
            # Boltzmann exploration: sample from softmax distribution with temperature
            temperature = max(self.epsilon, 0.1)
            # Apply temperature scaling
            log_probs = np.log(np.clip(action_probs, 1e-10, 1.0))
            scaled_probs = np.exp(log_probs / temperature)
            scaled_probs = scaled_probs / np.sum(scaled_probs)
            action = np.random.choice(self.action_dim, p=scaled_probs)
        else:
            # Epsilon-greedy exploration
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                action = np.argmax(action_probs)
        
        return action
    
    def compute_returns(self, rewards):
        """
        Compute discounted returns (G_t) for each timestep.
        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        """
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(device)
    
    def update(self, log_probs, rewards, states):
        """
        Update policy using REINFORCE algorithm with optional baseline.
        """
        # Compute returns
        returns = self.compute_returns(rewards)
        
        # Compute advantages
        if self.use_baseline:
            # Use baseline to reduce variance
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            baseline_values = self.baseline(states_tensor).squeeze()
            advantages = returns - baseline_values.detach()
            
            # Update baseline (value function)
            baseline_loss = F.mse_loss(baseline_values, returns)
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()
        else:
            # Use returns directly as advantages
            advantages = returns
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy gradient loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
    
    def train(self, num_episodes, max_steps):
        """
        Train the agent using REINFORCE algorithm.
        """
        all_episode_rewards = []
        
        for episode in range(num_episodes):
            # Use render environment for display episodes
            env = self.render_env if (episode + 1) % self.display_episodes == 0 else self.env
            
            # Reset environment
            state, _ = env.reset()
            
            # Episode storage
            episode_log_probs = []
            episode_rewards = []
            episode_states = []
            episode_total_reward = 0
            
            # Run episode
            for step in range(max_steps):
                # Select action
                action = self.select_action(state)
                
                # Get action probability and log probability
                state_tensor = torch.FloatTensor(state).to(device)
                action_probs = self.policy(state_tensor)
                log_prob = torch.log(torch.clamp(action_probs[action], min=1e-10))
                episode_log_probs.append(log_prob)
                
                # Take action in environment
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition
                episode_rewards.append(reward)
                episode_states.append(state)
                episode_total_reward += reward
                
                # Move to next state
                state = next_state
                
                # Render if display episode
                if (episode + 1) % self.display_episodes == 0:
                    env.render()
                
                # Check if episode ended
                if done or truncated:
                    break
            
            # Update policy after episode
            self.update(episode_log_probs, episode_rewards, episode_states)
            
            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate))
            
            # Store episode reward
            all_episode_rewards.append(episode_total_reward)
            
            # Print progress
            avg_reward = np.mean(all_episode_rewards[-50:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {episode_total_reward:.2f} | "
                  f"Avg (50): {avg_reward:.2f} | "
                  f"ε: {self.epsilon:.4f}", flush=True)
        
        return all_episode_rewards


# --- Main execution ---
if __name__ == "__main__":
    # Find all config files
    config_files = sorted(glob.glob(os.path.join(script_dir, "global_config*.json")))
    
    if not config_files:
        print("No config files found matching 'global_config*.json'")
        exit(1)
    
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
        
        # Set random seeds for reproducibility
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg["seed"])
        
        print(f"\n{'='*60}")
        print(f"REINFORCE Training on {cfg['env_name']}")
        print(f"Config: {config_name}")
        print(f"{'='*60}\n")
        
        # Create agent
        agent = REINFORCEAgent(cfg)
        
        # Train agent
        start_time = time.time()
        episode_rewards = agent.train(cfg["num_episodes"], cfg["max_steps"])
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
        
        print(f"Average Reward: {average_reward:.2f}")
        print(f"Std Reward: {std_reward:.2f}")
        print(f"Max Reward: {max_reward:.2f}")
        print(f"Min Reward: {min_reward:.2f}")
        
        # Store results for summary
        all_run_results.append({
            'config_name': config_name,
            'average_reward': average_reward,
            'std_reward': std_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'elapsed_time': elapsed_time
        })
        
        # Create results directory organized by config name
        results_base_dir = os.path.join(script_dir, "results", config_name)
        os.makedirs(results_base_dir, exist_ok=True)
        
        # Create folder name with average reward and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"reinforce_avg_{average_reward:.2f}_{timestamp}"
        results_dir = os.path.join(results_base_dir, folder_name)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\nSaving results to: {results_dir}")
        
        # Copy config file to results directory
        config_dest = os.path.join(results_dir, f"{config_name}.json")
        shutil.copy(config_path, config_dest)
        print(f"Config file saved to: {config_dest}")
        
        # Save rewards data
        rewards_file = os.path.join(results_dir, "episode_rewards.txt")
        np.savetxt(rewards_file, episode_rewards, fmt='%.6f')
        
        # Save summary statistics
        summary_file = os.path.join(results_dir, "summary.txt")
        with open(summary_file, "w") as f:
            f.write(f"REINFORCE Training Summary\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Config: {config_name}\n")
            f.write(f"Environment: {cfg['env_name']}\n")
            f.write(f"Episodes: {cfg['num_episodes']}\n")
            f.write(f"Max Steps: {cfg['max_steps']}\n")
            f.write(f"Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n\n")
            f.write(f"Average Reward: {average_reward:.2f}\n")
            f.write(f"Std Reward: {std_reward:.2f}\n")
            f.write(f"Max Reward: {max_reward:.2f}\n")
            f.write(f"Min Reward: {min_reward:.2f}\n\n")
            f.write(f"Hyperparameters:\n")
            f.write(f"  Learning Rate: {cfg['learning_rate']}\n")
            f.write(f"  Gamma: {cfg['gamma']}\n")
            f.write(f"  Initial Epsilon: {cfg['initial_epsilon']}\n")
            f.write(f"  Min Epsilon: {cfg['min_epsilon']}\n")
            f.write(f"  Decay Rate: {cfg['decay_rate']}\n")
            f.write(f"  Hidden Layers: {cfg['hidden_layers']}\n")
            f.write(f"  Activation: {cfg['activation_function']}\n")
            f.write(f"  Use Baseline: {cfg.get('use_baseline', True)}\n")
            f.write(f"  Use Boltzmann: {cfg.get('use_boltzmann', False)}\n")
        
        print(f"Summary saved to: {summary_file}")
        
        # Create plots
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        # Plot 1: Episode rewards
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(episodes, episode_rewards, alpha=0.6, linewidth=0.8)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.title(f"REINFORCE on {cfg['env_name']}\n{config_name}\n({'Boltzmann' if cfg.get('use_boltzmann', False) else 'Epsilon-greedy'} exploration)", fontsize=11)
        plt.grid(alpha=0.3)
        
        # Plot 2: Moving average
        plt.subplot(1, 2, 2)
        window = 20
        if len(episode_rewards) >= window:
            weights = np.ones(window) / window
            moving_avg = np.convolve(episode_rewards, weights, mode='valid')
            ma_episodes = np.arange(window, len(episode_rewards) + 1)
            plt.plot(ma_episodes, moving_avg, linewidth=2, color='orange')
            plt.xlabel("Episode", fontsize=12)
            plt.ylabel("Average Reward", fontsize=12)
            plt.title(f"{window}-Episode Moving Average", fontsize=13)
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, "training_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: {plot_file}")
        plt.close()
        
        # Plot 3: Learning curve with confidence interval
        plt.figure(figsize=(10, 6))
        window = 20
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
    
    # Print final summary of all runs
    print(f"\n{'#'*80}")
    print(f"# REINFORCE - All Configurations Summary")
    print(f"{'#'*80}\n")
    
    print(f"{'Config Name':<40} {'Avg Reward':>12} {'Std':>10} {'Time (min)':>12}")
    print(f"{'-'*80}")
    for result in all_run_results:
        print(f"{result['config_name']:<40} {result['average_reward']:>12.2f} {result['std_reward']:>10.2f} {result['elapsed_time']/60:>12.2f}")
    
    print(f"\n{'='*80}")
    print(f"All REINFORCE runs complete! Results saved in results/ directory.")
    print(f"{'='*80}\n")

