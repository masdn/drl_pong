"""
Demo script for the Retail Store Clerk environment.
Tests the environment with random actions and visualizes the agent's behavior.
"""

import gymnasium as gym
import retail_store_env  # This imports and registers the environment
import time


def demo_random_agent(num_episodes: int = 5, render: bool = True):
    """
    Demonstrate the environment with a random agent.
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render the environment
    """
    render_mode = "human" if render else None
    env = gym.make('RetailStore-v0', render_mode=render_mode)
    
    print("=" * 60)
    print("Retail Store Clerk Environment Demo")
    print("=" * 60)
    print(f"Grid Size: {env.unwrapped.grid_size}x{env.unwrapped.grid_size}")
    print(f"Number of Items: {env.unwrapped.num_items}")
    print(f"Back Office Location: {env.unwrapped.office_location}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print("=" * 60)
    print("\nItem Locations:")
    item_names = [
        "Dairy", "Frozen Foods", "Produce", "Bakery",
        "Canned Goods", "Beverages", "Cleaning Supplies", "Personal Care"
    ]
    for item_id, location in env.unwrapped.item_locations.items():
        print(f"  Item {item_id} ({item_names[item_id]}): {location}")
    print("=" * 60)
    print("\nActions:")
    print("  0: Move South (down)")
    print("  1: Move North (up)")
    print("  2: Move East (right)")
    print("  3: Move West (left)")
    print("  4: Pickup box")
    print("  5: Dropoff item")
    print("=" * 60)
    print("\nReward Structure:")
    print("  +1000: Exact correct location")
    print("  +2: Within 4 blocks (Manhattan distance)")
    print("  -10: Within 8 blocks")
    print("  -500: More than 8 blocks away")
    print("  -1: Each step (encourages efficiency)")
    print("=" * 60)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        print(f"Starting position: {info['agent_pos']}")
        print(f"Current item to stock: {info['current_item']} ({item_names[info['current_item']]})")
        print(f"Target location: {info['target_location']}")
        print(f"Initial distance: {info['distance_to_target']} blocks")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            
            if reward != -1:  # Only print non-step rewards
                print(f"Step {steps}: Action={action_names[action]}, "
                      f"Pos={info['agent_pos']}, Reward={reward:.1f}, "
                      f"Distance={info['distance_to_target']}")
            
            if render and not done:
                time.sleep(0.1)  # Slow down for visualization
        
        print(f"\nEpisode {episode + 1} finished!")
        print(f"Total steps: {steps}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Final position: {info['agent_pos']}")
        print(f"Success: {'Yes' if reward == 1000 else 'No'}")
        
        if render:
            time.sleep(2)  # Pause between episodes
    
    env.close()
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def test_manual_navigation():
    """
    Test the environment by manually navigating to a known location.
    This demonstrates that an agent can successfully complete the task.
    """
    env = gym.make('RetailStore-v0')
    observation, info = env.reset()
    
    print("\n" + "=" * 60)
    print("Manual Navigation Test")
    print("=" * 60)
    print(f"Target item: {info['current_item']}")
    print(f"Target location: {info['target_location']}")
    print(f"Starting position: {info['agent_pos']}")
    
    # Simple navigation: move towards target using Manhattan distance
    done = False
    episode_reward = 0
    steps = 0
    max_steps = 50
    
    while not done and steps < max_steps:
        current_pos = info['agent_pos']
        target_pos = info['target_location']
        
        # Choose action to move towards target
        if current_pos[0] > target_pos[0]:  # Need to go up
            action = 1  # North
        elif current_pos[0] < target_pos[0]:  # Need to go down
            action = 0  # South
        elif current_pos[1] > target_pos[1]:  # Need to go left
            action = 3  # West
        elif current_pos[1] < target_pos[1]:  # Need to go right
            action = 2  # East
        else:
            # At target location, drop off
            action = 5  # Dropoff
        
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1
        
        if reward > 100 or reward < -100:
            action_names = ["South", "North", "East", "West", "Pickup", "Dropoff"]
            print(f"Step {steps}: Action={action_names[action]}, "
                  f"Pos={info['agent_pos']}, Reward={reward:.1f}")
    
    print(f"\nNavigation complete!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {episode_reward:.2f}")
    print(f"Success: {'Yes' if episode_reward > 500 else 'No'}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    import sys
    
    # Check if rendering is requested
    render = "--no-render" not in sys.argv
    
    # Run manual navigation test first to verify environment works
    test_manual_navigation()
    
    # Run random agent demo
    print("\nNow testing with random actions...")
    demo_random_agent(num_episodes=3, render=render)


