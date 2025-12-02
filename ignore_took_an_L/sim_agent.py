import argparse
import os

import numpy as np
import torch

from a2c import ActorCriticNet, make_pong_env, device


def find_latest_checkpoint(checkpoints_dir: str) -> str | None:
    """
    Find the most recent .pt file in the checkpoints directory.
    Returns the full path or None if no checkpoints exist.
    """
    if not os.path.isdir(checkpoints_dir):
        return None

    candidates = [
        os.path.join(checkpoints_dir, f)
        for f in os.listdir(checkpoints_dir)
        if f.endswith(".pt")
    ]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def build_model_and_env(checkpoint: dict, render_mode: str = "human"):
    """
    Recreate the Pong environment and model from a saved checkpoint config.
    """
    cfg = checkpoint["config"]

    env_name = cfg["env_name"]
    frame_skip = cfg.get("frame_skip", 1)
    frame_stack = cfg.get("frame_stack", 4)
    grayscale = cfg.get("grayscale", True)
    scale_obs = cfg.get("scale_obs", True)
    seed = cfg.get("seed", 0)

    # Evaluation environment with rendering
    env = make_pong_env(
        env_name=env_name,
        seed=seed,
        render_mode=render_mode,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        grayscale=grayscale,
        scale_obs=scale_obs,
    )

    obs_shape = env.observation_space.shape  # (C, H, W)
    num_actions = env.action_space.n
    fc_size = cfg.get("fc_size", 256)

    model = ActorCriticNet(obs_shape, num_actions, fc_size=fc_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return env, model, cfg


def run_simulation(
    checkpoint_path: str,
    episodes: int = 5,
    max_steps: int | None = None,
    render_mode: str = "human",
):
    """
    Load a saved A2C Pong checkpoint and run a purely evaluative simulation
    (no training), rendering the gameplay.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    env, model, cfg = build_model_and_env(checkpoint, render_mode=render_mode)

    if max_steps is None:
        max_steps = cfg.get("max_steps", 10000)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            state_arr = np.array(state, copy=False)
            state_tensor = (
                torch.from_numpy(state_arr).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                logits, value = model(state_tensor)
                # Greedy action selection for evaluation
                action = int(torch.argmax(logits, dim=-1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()

            total_reward += float(reward)
            steps += 1
            state = next_state
            done = bool(terminated or truncated)

        print(
            f"Episode {ep}/{episodes} finished | "
            f"Reward: {total_reward:.2f} | Steps: {steps}"
        )

    env.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(script_dir, "checkpoints")

    parser = argparse.ArgumentParser(
        description="Run a saved A2C Pong agent from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a specific checkpoint .pt file. "
            "If omitted, the most recent file in checkpoints/ is used."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max steps per episode (defaults to config max_steps).",
    )

    args = parser.parse_args()

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(checkpoints_dir)

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        print(
            "No checkpoint found. "
            "Make sure you have run a2c.py and that a .pt file exists in checkpoints/ "
            "or provide --checkpoint explicitly."
        )
        raise SystemExit(1)

    run_simulation(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render_mode="human",
    )


if __name__ == "__main__":
    main()


