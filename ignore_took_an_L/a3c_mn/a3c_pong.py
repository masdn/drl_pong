import json
import os

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
import torch.multiprocessing as mp

from model import ActorCriticModel
import sharedRMS
from a3c_train import train
from a3c_test import test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_pong_env(
    env_name: str,
    seed: int,
    render_mode: str | None = None,
    frame_skip: int = 1,
    frame_stack: int = 4,
    grayscale: bool = True,
    scale_obs: bool = True,
):
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



def get_from_config(config_path: str = "config.json"):
    """
    Load configuration from a JSON file and return the config dictionary.
    
    Args:
        config_path: Path to the JSON config file (default: "config.json")
        
    Returns:
        dict: Configuration dictionary containing all settings
    """
    # If no path separator, assume it's in the same directory as this file
    if "/" not in config_path and "\\" not in config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config


if __name__ == "__main__":
    # Ensure safe start method for multiprocessing with Gym and CUDA
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Start method may already be set; ignore.
        pass

    config = get_from_config()

    print(
        f"Starting A3C Pong | env={config['env_name']} | "
        f"num_processes={config['num_processes']} | "
        f"num_steps={config['num_steps']} | lr={config['lr']}",
        flush=True,
    )

    env = make_pong_env(
        env_name=config["env_name"],
        seed=config["seed"],
        render_mode=None,
        frame_skip=config["frame_skip"],
        frame_stack=config["frame_stack"],
        grayscale=config["grayscale"],
        scale_obs=config["scale_obs"],
    )
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    torch.manual_seed(config["seed"])

    shared_model = ActorCriticModel(obs_shape, num_actions)
    shared_model.share_memory()

    optimizer = sharedRMS.SharedRMSprop(shared_model.parameters(), lr=config["lr"])
    optimizer.share_memory()

    # Multiprocessing setup: one test process plus multiple training workers.
    # NOTE: you still need to implement `test` and `train` functions elsewhere.
    processes = []

    counter = mp.Value("i", 0)
    lock = mp.Lock()

    # Test process (e.g., periodic evaluation using the shared model)
    p = mp.Process(
        target=test,
        args=(
            config["num_processes"],
            config,
            shared_model,
            counter,
        ),
    )
    p.start()
    processes.append(p)

    # Training worker processes
    for rank in range(config["num_processes"]):
        p = mp.Process(
            target=train,
            args=(
                rank,
                config,
                shared_model,
                counter,
                lock,
                optimizer,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
