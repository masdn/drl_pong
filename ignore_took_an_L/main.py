import gymnasium as gym
import ale_py

gym.register_envs(ale_py)   # <-- this is the key line

env = gym.make("ALE/Pong-v5")

print(env.action_space, env.observation_space)