# Retail Store Clerk Environment

A custom OpenAI Gymnasium environment simulating a new retail store clerk learning where to stock items. The environment is modeled after the classic Taxi environment with discrete state and action spaces.

## Environment Description

### Scenario
A new clerk at a retail store must learn where different types of items belong on the store floor. The clerk starts at the back office (bottom-middle of the store) with a box containing one item. At first, the clerk has no knowledge of where items should go, but through trial and error, gradually learns the correct locations for each item type.

### Grid Layout
- **Grid Size**: 10x10 by default
- **Back Office**: Located at position (9, 5) - bottom-middle of the store
- **8 Item Types**: Each with a specific location on the store floor
  - Item 0: Dairy (top-left)
  - Item 1: Frozen Foods (top-right)
  - Item 2: Produce (left-middle)
  - Item 3: Bakery (right-middle)
  - Item 4: Canned Goods (center-lower)
  - Item 5: Beverages (top-center)
  - Item 6: Cleaning Supplies (left-lower)
  - Item 7: Personal Care (right-lower)

### Action Space
The environment has 6 discrete actions (same as Taxi environment):
- **0**: Move South (down)
- **1**: Move North (up)
- **2**: Move East (right)
- **3**: Move West (left)
- **4**: Pickup box (return to office for new item)
- **5**: Dropoff item (place item at current location)

### Observation Space
The observation is a discrete integer that encodes:
- Agent's position (row, col)
- Current item type (0-7)
- Whether the agent is carrying an item

**State encoding formula**: 
```
state = row * grid_size * (num_items + 1) + col * (num_items + 1) + item_encoding
```
where `item_encoding = current_item` if carrying, else `num_items`

### Reward Structure
The reward system is distance-based using Manhattan distance:

- **+1000**: Item placed at the exact correct location (success!)
- **+2**: Item placed within 4 blocks of the correct location
- **-10**: Item placed within 8 blocks of the correct location
- **-500**: Item placed more than 8 blocks away from the correct location
- **-1**: Each step taken (encourages efficiency)

### Episode Termination
An episode terminates when:
- The agent successfully places an item at the correct location (+1000 reward), OR
- The maximum number of steps (200 by default) is reached

## Installation

### Requirements
```bash
pip install gymnasium numpy pygame matplotlib
```

Or use the existing virtual environment:
```bash
source venv_cpu/bin/activate
```

### Files
- `retail_store_env.py` - The custom Gymnasium environment
- `demo_retail_store.py` - Demo script showing random and manual agents
- `sarsa_retail_store.py` - SARSA training script for the environment
- `retail_store_config_default.json` - Default configuration (auto-generated)

## Usage

### 1. Demo the Environment

Run a demo with random actions:
```bash
python demo_retail_store.py
```

Run without rendering (faster):
```bash
python demo_retail_store.py --no-render
```

The demo will:
- Show random agent behavior
- Test manual navigation to verify the environment works
- Display the grid with colored item locations
- Print detailed episode information

### 2. Train an Agent

Train using SARSA (tabular Q-learning):
```bash
python sarsa_retail_store.py
```

This will:
- Create a default configuration if none exists
- Train the agent for 1000 episodes
- Save results, plots, and Q-table
- Display training progress and final statistics

### 3. Use in Your Own Code

```python
import gymnasium as gym
import retail_store_env  # This registers the environment

# Create environment
env = gym.make('RetailStore-v0', render_mode='human')

# Reset and run
observation, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Your agent's policy here
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

## Configuration

Create a custom configuration file (e.g., `retail_store_config_custom.json`):

```json
{
  "env_name": "RetailStore-v0",
  "num_episodes": 1000,
  "max_steps": 200,
  "gamma": 0.99,
  "learning_rate": 0.1,
  "initial_epsilon": 1.0,
  "min_epsilon": 0.01,
  "decay_rate": 0.005,
  "use_boltzmann": false,
  "display_episodes": 1001,
  "seed": 42
}
```

### Key Parameters
- **gamma**: Discount factor (0.99 = value future rewards highly)
- **learning_rate**: How quickly to update Q-values (0.1 is typical for tabular)
- **initial_epsilon**: Starting exploration rate (1.0 = pure exploration)
- **min_epsilon**: Minimum exploration rate (0.01 = 1% random actions)
- **decay_rate**: How quickly epsilon decays (higher = faster decay)
- **use_boltzmann**: Use Boltzmann exploration instead of epsilon-greedy
- **display_episodes**: Render every Nth episode (set to 1001 to disable)

## Learning Challenges

This environment presents several learning challenges:

1. **Sparse Rewards**: The agent only gets large positive rewards when placing items correctly
2. **Exploration**: Must explore the entire grid to discover item locations
3. **Credit Assignment**: Agent must learn which states lead to successful placement
4. **Multiple Goals**: 8 different item locations to learn
5. **Negative Rewards**: Penalties for incorrect placement discourage random guessing

## Expected Performance

With proper hyperparameters, a SARSA agent should:
- Learn to navigate efficiently within 200-400 episodes
- Achieve >80% success rate within 500-800 episodes
- Reach >95% success rate by 1000 episodes
- Learn optimal paths to all 8 item locations

Success is measured by:
- **Success Rate**: Percentage of episodes ending with +1000 reward
- **Average Reward**: Should increase from negative to ~900+ when trained
- **Steps per Episode**: Should decrease as agent learns efficient paths

## Visualization

When rendering is enabled (`render_mode='human'`):
- **Gray square**: Back office (starting location)
- **Colored squares**: Item destination locations (labeled 0-7)
- **Blue circle**: Agent without item
- **Red circle**: Agent carrying item (shows item number)
- **Grid lines**: Store floor layout

## Extending the Environment

You can customize the environment by:

1. **Change grid size**: Pass `grid_size` parameter when creating environment
   ```python
   env = gym.make('RetailStore-v0', grid_size=15)
   ```

2. **Add more items**: Modify `item_locations` dictionary in `RetailStoreEnv.__init__()`

3. **Adjust reward structure**: Modify the `step()` method's reward logic

4. **Change starting position**: Modify `office_location` in `__init__()`

## Comparison to Taxi Environment

### Similarities
- Discrete state and action spaces
- Pickup and dropoff actions
- Grid-based navigation
- Learning where to deliver items

### Differences
- **Multiple item types** (8) vs single passenger delivery
- **Distance-based rewards** vs binary success/failure
- **Simpler pickup** (start with item) vs passenger location
- **10x10 grid** vs 5x5 grid
- **Exploration focus** (learning locations) vs navigation optimization

## Troubleshooting

### ImportError: No module named 'retail_store_env'
Make sure you're running scripts from the same directory as `retail_store_env.py`

### Pygame display issues
Run with `--no-render` flag to disable visualization:
```bash
python demo_retail_store.py --no-render
```

### Agent not learning
- Increase `num_episodes` (try 2000+)
- Adjust `learning_rate` (try 0.05-0.2 range)
- Reduce `decay_rate` for longer exploration
- Check that rewards are being received (look at episode summaries)

### Training too slow
- Disable rendering by setting `display_episodes` to a large number
- Reduce `num_episodes` for faster testing
- Use `--no-render` flag in demo script

## Results and Analysis

Training results are saved to `results/retail_store/[config_name]/` with:
- `summary.txt` - Training statistics and hyperparameters
- `episode_rewards.txt` - Reward for each episode
- `q_table.npy` - Learned Q-table (can be loaded with `np.load()`)
- `training_plot.png` - Episode rewards, moving average, and success rate
- `learning_curve.png` - Mean reward with confidence intervals

## Future Enhancements

Possible extensions to the environment:
- Multiple items per episode (stocking multiple products)
- Dynamic item locations (items move between episodes)
- Obstacles (shelves, displays that block movement)
- Time pressure (decreasing rewards over time)
- Inventory management (limited storage at each location)
- Co-workers (multi-agent environment)

## License

This environment follows the Gymnasium/OpenAI Gym API conventions and is provided as an educational example.

## Credits

Created for CS491 - Assignment 2
Inspired by the classic Taxi-v3 environment from OpenAI Gym


