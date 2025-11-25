"""
Evaluate a trained SARSA agent on the Retail Store environment.
Loads a saved Q-table and runs evaluation episodes.
"""

import gymnasium as gym
import retail_store_env
import numpy as np
import argparse
import time
import os


def load_qtable(qtable_path):
    """Load Q-table from file."""
    if not os.path.exists(qtable_path):
        raise FileNotFoundError(f"Q-table not found: {qtable_path}")
    return np.load(qtable_path)


def evaluate_agent(q_table, num_episodes=10, render=True, delay=0.1):
    """
    Evaluate a trained agent using greedy policy from Q-table.
    
    Args:
        q_table: Trained Q-table (numpy array)
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
        delay: Delay between steps (for visualization)
    
    Returns:
        Dictionary with evaluation statistics
    """
    render_mode = "human" if render else None
    env = gym.make('RetailStore-v0', render_mode=render_mode)
    
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    item_names = [
        "Dairy", "Frozen Foods", "Produce", "Bakery",
        "Canned Goods", "Beverages", "Cleaning Supplies", "Personal Care"
    ]
    
    print("\n" + "="*70)
    print("Evaluating Trained Agent on Retail Store Environment")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Render: {'Yes' if render else 'No'}")
    print("="*70 + "\n")
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        print(f"Item to stock: {info['current_item']} ({item_names[info['current_item']]})")
        print(f"Target location: {info['target_location']}")
        print(f"Starting position: {info['agent_pos']}")
        print(f"Initial distance: {info['distance_to_target']} blocks\n")
        
        while not done and steps < 200:
            # Greedy action selection (no exploration)
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            # Print significant events
            action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            if reward > 100 or reward < -100:
                print(f"  Step {steps}: {action_names[action]:8s} -> "
                      f"Pos {info['agent_pos']}, Reward: {reward:+7.0f}, "
                      f"Distance: {info['distance_to_target']}")
            
            state = next_state
            
            if render and not done:
                time.sleep(delay)
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        if episode_reward >= 900:  # Consider success if got the big reward
            success_count += 1
            print(f"\n✅ SUCCESS! Completed in {steps} steps")
        else:
            print(f"\n❌ Failed. Total reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps Taken: {steps}")
        print(f"  Final Position: {info['agent_pos']}")
        
        if render:
            time.sleep(1.5)  # Pause between episodes
    
    env.close()
    
    # Compute statistics
    stats = {
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'avg_steps': np.mean(episode_steps),
        'success_count': success_count,
        'success_rate': success_count / num_episodes * 100
    }
    
    # Print final statistics
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Episodes Evaluated: {num_episodes}")
    print(f"Successful Episodes: {success_count}/{num_episodes} ({stats['success_rate']:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Average Reward: {stats['avg_reward']:.2f}")
    print(f"  Std Deviation: {stats['std_reward']:.2f}")
    print(f"  Max Reward: {stats['max_reward']:.2f}")
    print(f"  Min Reward: {stats['min_reward']:.2f}")
    print(f"\nStep Statistics:")
    print(f"  Average Steps: {stats['avg_steps']:.1f}")
    print(f"  Steps per Success: {np.mean([s for i, s in enumerate(episode_steps) if episode_rewards[i] >= 900]):.1f}")
    print("="*70 + "\n")
    
    return stats


def find_latest_qtable():
    """Find the most recently saved Q-table."""
    results_dir = "results/retail_store"
    
    if not os.path.exists(results_dir):
        return None
    
    # Find all Q-table files
    qtable_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "q_table.npy":
                qtable_files.append(os.path.join(root, file))
    
    if not qtable_files:
        return None
    
    # Return most recent
    qtable_files.sort(key=os.path.getmtime, reverse=True)
    return qtable_files[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Retail Store agent")
    parser.add_argument("--qtable", type=str, default=None,
                        help="Path to Q-table file (.npy). If not provided, uses most recent.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between steps in seconds (default: 0.1)")
    
    args = parser.parse_args()
    
    # Find Q-table
    if args.qtable is None:
        print("No Q-table specified. Looking for most recent...")
        qtable_path = find_latest_qtable()
        if qtable_path is None:
            print("\nError: No trained Q-table found in results/retail_store/")
            print("Please train an agent first using: python sarsa_retail_store.py")
            exit(1)
        print(f"Found Q-table: {qtable_path}\n")
    else:
        qtable_path = args.qtable
    
    # Load Q-table
    try:
        q_table = load_qtable(qtable_path)
        print(f"Loaded Q-table with shape: {q_table.shape}")
    except Exception as e:
        print(f"Error loading Q-table: {e}")
        exit(1)
    
    # Evaluate
    stats = evaluate_agent(
        q_table,
        num_episodes=args.episodes,
        render=not args.no_render,
        delay=args.delay
    )
    
    # Save evaluation results
    eval_dir = os.path.dirname(qtable_path)
    eval_file = os.path.join(eval_dir, "evaluation_results.txt")
    
    with open(eval_file, "w") as f:
        f.write("Retail Store Agent Evaluation Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Q-table: {qtable_path}\n")
        f.write(f"Evaluation Episodes: {stats['num_episodes']}\n\n")
        f.write(f"Success Rate: {stats['success_rate']:.2f}%\n")
        f.write(f"Successful Episodes: {stats['success_count']}/{stats['num_episodes']}\n\n")
        f.write(f"Reward Statistics:\n")
        f.write(f"  Average: {stats['avg_reward']:.2f}\n")
        f.write(f"  Std Dev: {stats['std_reward']:.2f}\n")
        f.write(f"  Max: {stats['max_reward']:.2f}\n")
        f.write(f"  Min: {stats['min_reward']:.2f}\n\n")
        f.write(f"Step Statistics:\n")
        f.write(f"  Average Steps: {stats['avg_steps']:.1f}\n\n")
        f.write("Per-Episode Results:\n")
        for i, (reward, steps) in enumerate(zip(stats['episode_rewards'], stats['episode_steps'])):
            status = "SUCCESS" if reward >= 900 else "FAILED"
            f.write(f"  Episode {i+1}: {status:7s} | Reward: {reward:7.2f} | Steps: {steps:3d}\n")
    
    print(f"Evaluation results saved to: {eval_file}")


