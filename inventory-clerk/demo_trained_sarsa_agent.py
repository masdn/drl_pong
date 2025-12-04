"""
Run a trained tabular SARSA agent on the RetailStore environment and render it.

This script:
  - Finds a saved `q_table.npy` from `results/retail_store/...` (or uses a
    user‑specified run directory)
  - Loads the corresponding config JSON to get env_name / max_steps
  - Runs a purely greedy policy (argmax over Q) with rendering enabled
"""

import os
import sys
import glob
import time
from typing import Tuple

import numpy as np
import gymnasium as gym
import retail_store_env  # noqa: F401  - registers the environment


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_latest_run() -> Tuple[str, str]:
    """
    Find the most recent run folder that contains a q_table.npy.

    Returns:
        (run_dir, q_table_path)
    Raises:
        FileNotFoundError if no saved runs are found.
    """
    base_results = os.path.join(SCRIPT_DIR, "results", "retail_store")
    pattern = os.path.join(base_results, "*", "*", "q_table.npy")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"No saved q_table.npy found under {base_results}. "
            "Run sarsa_retail_store.py first."
        )

    # Pick the most recently modified q_table
    latest_q = max(candidates, key=os.path.getmtime)
    run_dir = os.path.dirname(latest_q)
    return run_dir, latest_q


def load_config_for_run(run_dir: str) -> dict:
    """Load the config JSON that was copied into the run directory."""
    json_files = glob.glob(os.path.join(run_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No config JSON found in run directory: {run_dir}"
        )
    # Typically there is only one; take the first.
    config_path = json_files[0]
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def run_trained_agent(run_dir: str, num_episodes: int = 5, sleep: float = 0.1):
    """Run a greedy policy from a saved Q-table and render episodes."""
    q_table_path = os.path.join(run_dir, "q_table.npy")
    if not os.path.exists(q_table_path):
        raise FileNotFoundError(f"q_table.npy not found in {run_dir}")

    q_table = np.load(q_table_path)
    
    cfg = load_config_for_run(run_dir)
    env_name = cfg.get("env_name", "RetailStore-v0")
    max_steps = cfg.get("max_steps", 200)
    enable_customers = cfg.get("enable_customers", True)
    
    print(f"Using run directory: {run_dir}")
    print(f"Loaded Q-table shape: {q_table.shape}")
    print(f"Environment: {env_name}, max_steps: {max_steps}, enable_customers={enable_customers}")
    
    env = gym.make(env_name, render_mode="human", enable_customers=enable_customers)
    
    # Sanity-check that the Q-table's state dimension matches the environment's
    # current discrete state index size. If they don't match, this usually means
    # the environment definition changed after training and the old Q-table is
    # no longer compatible.
    try:
        expected_state_dim = env.unwrapped.state_index_size
    except AttributeError:
        # Older versions of the env used a simple Discrete observation_space.
        expected_state_dim = env.observation_space.n
    
    if q_table.shape[0] != expected_state_dim:
        print(
            f"\nWARNING: Q-table state dimension ({q_table.shape[0]}) does not "
            f"match environment state index size ({expected_state_dim}).\n"
            "This Q-table was likely trained with an older version of the "
            "environment. Please retrain the SARSA agent before using this "
            "demo, or switch to a run directory that matches the current env."
        )
        env.close()
        return

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            # Use the compact integer state index if provided; otherwise fall
            # back to the raw observation for older env versions.
            state = info.get("state_index", obs)
            state = int(state)
            done = False
            total_reward = 0.0
            steps = 0

            print(f"\nEpisode {episode + 1}/{num_episodes}")

            while not done and steps < max_steps:
                # Greedy action from Q-table
                action = int(np.argmax(q_table[state]))
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = info.get("state_index", next_obs)
                next_state = int(next_state)

                total_reward += reward
                steps += 1
                done = terminated or truncated

                # Render is automatic in this env when render_mode='human'
                time.sleep(sleep)
                
                state = next_state

            print(f"  Steps: {steps}, Total reward: {total_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    # Optional argument: path to a specific run directory.
    # If omitted, we pick the most recent run automatically.
    if len(sys.argv) > 1:
        run_dir_arg = sys.argv[1]
        if not os.path.isabs(run_dir_arg):
            run_dir_arg = os.path.join(SCRIPT_DIR, run_dir_arg)
        run_dir, _ = os.path.split(run_dir_arg.rstrip("/"))
        # If the arg is already a directory that contains q_table.npy, keep it.
        if os.path.isdir(run_dir_arg) and os.path.exists(
            os.path.join(run_dir_arg, "q_table.npy")
        ):
            run_dir = run_dir_arg
        else:
            # Assume user pointed directly at q_table.npy
            if os.path.basename(run_dir_arg) == "q_table.npy":
                run_dir = os.path.dirname(run_dir_arg)
            else:
                raise FileNotFoundError(
                    "Argument should be a run directory containing q_table.npy "
                    "or the path to q_table.npy itself."
                )
    else:
        run_dir, _ = find_latest_run()

    run_trained_agent(run_dir)


