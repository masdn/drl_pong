# Retail Store Environment - Quick Start Guide

## 🎯 What You Have

A custom OpenAI Gymnasium environment where an AI agent learns to stock items in a retail store!

**Key Features:**
- ✅ 10x10 grid store layout
- ✅ 8 different item types to learn
- ✅ Distance-based reward system
- ✅ Same action space as classic Taxi environment
- ✅ Visual rendering with Pygame
- ✅ Ready-to-use training scripts

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Virtual Environment
```bash
cd /Users/student/Documents/cs491-assignment2
source venv_cpu/bin/activate
```

### Step 2: Test the Environment
```bash
# Run demo with visualization (if you have display)
python demo_retail_store.py

# Or run without visualization (faster)
python demo_retail_store.py --no-render
```

### Step 3: Train an Agent
```bash
# This will train for 1000 episodes and save results
python sarsa_retail_store.py
```

## 📁 Files Created

### Core Environment
- **`retail_store_env.py`** - The custom Gymnasium environment class
  - Grid-based retail store
  - 6 actions: move (4 directions), pickup, dropoff
  - Distance-based rewards
  - Discrete state space

### Demo & Testing
- **`demo_retail_store.py`** - Test script with examples
  - Random agent demo
  - Manual navigation test
  - Shows how environment works

### Training
- **`sarsa_retail_store.py`** - SARSA training script
  - Tabular Q-learning (discrete state space)
  - Automatic config generation
  - Saves results, plots, and Q-table
  - Success rate tracking

### Documentation
- **`RETAIL_STORE_README.md`** - Full documentation
- **`RETAIL_STORE_QUICKSTART.md`** - This file
- **`retail_store_config_default.json`** - Default hyperparameters (auto-generated)

## 🎮 How It Works

### The Scenario
A new clerk must learn where 8 different item types belong in the store:
- Dairy, Frozen Foods, Produce, Bakery
- Canned Goods, Beverages, Cleaning Supplies, Personal Care

### Agent Actions
```
0: Move South (down)
1: Move North (up)
2: Move East (right)
3: Move West (left)
4: Pickup box (get new item from office)
5: Dropoff item (try to place at current location)
```

### Reward System
```
+1000  ← Perfect! Item at exact correct location ✅
+2     ← Close (within 4 blocks)
-10    ← Getting there (within 8 blocks)
-500   ← Too far away (> 8 blocks)
-1     ← Each step (encourages efficiency)
```

## 📊 Expected Results

With default settings (1000 episodes), you should see:

**Early Training (Episodes 1-200)**
- Agent explores randomly
- Many -500 penalties
- Average reward: -300 to -100

**Mid Training (Episodes 200-600)**
- Agent starts finding correct locations
- More +2 and -10 rewards
- Average reward: -50 to +200

**Late Training (Episodes 600-1000)**
- Agent knows most/all locations
- Many +1000 successes
- Average reward: +800 to +950
- Success rate: 80-95%

## 🔧 Customization

### Change Training Length
Edit `retail_store_config_default.json`:
```json
{
  "num_episodes": 2000,  // Train longer
  "max_steps": 200       // Steps per episode
}
```

### Adjust Learning
```json
{
  "learning_rate": 0.15,    // How fast to learn (0.05-0.2)
  "gamma": 0.99,            // Value future rewards (0.95-0.99)
  "initial_epsilon": 1.0,   // Start exploration (0.8-1.0)
  "decay_rate": 0.003       // Exploration decay (0.001-0.01)
}
```

### Try Different Exploration
```json
{
  "use_boltzmann": true  // Use Boltzmann instead of epsilon-greedy
}
```

## 📈 View Results

After training, check `results/retail_store/retail_store_config_default/`:

```
sarsa_avg_XXX_success_XX_TIMESTAMP/
├── summary.txt              ← Statistics and hyperparameters
├── episode_rewards.txt      ← Reward per episode
├── q_table.npy             ← Learned Q-values (load with np.load)
├── training_plot.png       ← Visualizations
└── learning_curve.png      ← Learning progress
```

## 🐛 Troubleshooting

### "No module named 'retail_store_env'"
**Fix**: Make sure you're in the correct directory
```bash
cd /Users/student/Documents/cs491-assignment2
python demo_retail_store.py
```

### Pygame/Display Issues
**Fix**: Run without rendering
```bash
python demo_retail_store.py --no-render
```

### Agent Not Learning
**Try these:**
1. Train longer: Set `num_episodes` to 2000+
2. Increase learning rate: Try 0.15 or 0.2
3. Slower exploration decay: Set `decay_rate` to 0.002 or 0.001

### Training Too Slow
**Speed up:**
1. Disable rendering: Set `display_episodes` to 100000 in config
2. Train with fewer episodes initially: Set `num_episodes` to 500 for testing

## 💡 Usage in Your Code

```python
import gymnasium as gym
import retail_store_env  # Registers the environment

# Create environment
env = gym.make('RetailStore-v0')

# Training loop
for episode in range(100):
    state, info = env.reset()
    done = False
    
    while not done:
        # Your agent's policy here
        action = select_action(state)  
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update your agent here
        update_policy(state, action, reward, next_state)
        state = next_state
    
    print(f"Episode {episode}: Reward = {info['total_reward']}")

env.close()
```

## 🎓 Learning Resources

### Reinforcement Learning Concepts Used
- **SARSA**: On-policy TD control (learns while following current policy)
- **Q-Learning**: Action-value function approximation
- **Epsilon-Greedy**: Balance exploration vs exploitation
- **Discount Factor (γ)**: How much to value future rewards
- **Temporal Difference**: Learn from each step, not just episode end

### Compare to Other Environments
- **Similar to**: Taxi-v3 (OpenAI Gym)
- **Simpler than**: LunarLander (continuous state)
- **More complex than**: FrozenLake (single goal)

## 🔄 Next Steps

1. **Run demo**: See how environment works
2. **Train baseline**: Run with default settings
3. **Experiment**: Try different hyperparameters
4. **Analyze**: Compare different configurations
5. **Extend**: Add your own features or algorithms

## 📞 Support

- Read `RETAIL_STORE_README.md` for detailed documentation
- Check your existing `sarsa_lunar.py` for comparison
- Look at `demo_retail_store.py` for usage examples

## 🌟 Features Not in Taxi Environment

1. **Distance-based rewards** - Gradual feedback instead of binary
2. **Multiple item types** - 8 different locations to learn
3. **Larger grid** - 10x10 vs 5x5
4. **Simpler pickup** - Start with item, no passenger location
5. **Exploration focus** - Must discover all 8 locations

## ✅ Success Criteria

Your agent is learning well when:
- ✅ Success rate > 80% by episode 800
- ✅ Average reward > +900 in final episodes
- ✅ Can navigate to all 8 item locations
- ✅ Takes <15 steps per successful episode

Good luck training your retail clerk! 🛒📦


