import time
from collections import deque

import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from model import ActorCriticModel


def make_env_from_config(cfg: dict, rank: int):
    """Create an Atari env consistent with the training configuration."""
    seed = cfg["seed"] + rank
    env = gym.make(cfg["env_name"], render_mode=None)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    env = AtariPreprocessing(
        env,
        frame_skip=cfg.get("frame_skip", 4),
        grayscale_obs=cfg.get("grayscale", True),
        scale_obs=cfg.get("scale_obs", True),
    )
    if cfg.get("frame_stack", 4) > 1:
        env = FrameStackObservation(env, cfg.get("frame_stack", 4))
    return env


def test(rank, cfg, shared_model, counter):
    """
    Evaluation worker, adapted from the reference A3C test loop.

    Args:
        rank: integer rank / process id (first arg passed from main).
        cfg: configuration dict (from config.json).
        shared_model: global shared ActorCriticModel.
        counter: shared mp.Value counting total environment steps.
    """
    torch.manual_seed(cfg["seed"] + rank)

    env = make_env_from_config(cfg, rank)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    model = ActorCriticModel(obs_shape, num_actions)
    model.eval()

    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0.0
    done = True

    start_time = time.time()

    # quick hack to prevent the agent from getting stuck
    actions = deque(maxlen=100)
    episode_length = 0

    while True:
        episode_length += 1

        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            logits, value, (hx, cx) = model(state.unsqueeze(0), hx, cx)

        prob = F.softmax(logits, dim=-1)
        action = prob.max(1, keepdim=True)[1].cpu().numpy()

        next_state, reward, terminated, truncated, _ = env.step(int(action[0, 0]))
        done = bool(terminated or truncated)
        done = done or episode_length >= cfg["max_episode_length"]
        reward_sum += reward

        # quick hack to prevent the agent from getting stuck
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            elapsed = time.time() - start_time
            fps = counter.value / max(elapsed, 1e-6)
            print(
                "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed)),
                    counter.value,
                    fps,
                    reward_sum,
                    episode_length,
                ),
                flush=True,
            )
            reward_sum = 0.0
            episode_length = 0
            actions.clear()
            next_state, _ = env.reset()
            time.sleep(60)

        state = torch.from_numpy(next_state)


