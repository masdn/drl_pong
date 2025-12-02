import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from collections import deque

from model import ActorCriticModel


def make_env_from_config(cfg: dict, rank: int):
    """Create a single Atari env using the same settings as in config."""
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


def ensure_shared_grads(model: torch.nn.Module, shared_model: torch.nn.Module):
    """Copy gradients from local model to shared model (A3C style)."""
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, cfg, shared_model, counter, lock, optimizer=None):
    """
    A3C worker training loop, modeled after the reference implementation
    but adapted to this project's ActorCriticModel and Gymnasium API.
    """
    # cfg is a dict loaded from config.json
    torch.manual_seed(cfg["seed"] + rank)

    env = make_env_from_config(cfg, rank)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Local copy of the model (on CPU)
    model = ActorCriticModel(obs_shape, num_actions)

    # Per-worker optimizer fallback if none is shared in main
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=cfg["lr"])

    model.train()

    state, _ = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    episode_reward = 0.0
    episode_idx = 0
    recent_rewards = deque(maxlen=50)
    while True:
        # Sync local weights with the shared model
        model.load_state_dict(shared_model.state_dict())

        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(cfg["num_steps"]):
            episode_length += 1

            logits, value, (hx, cx) = model(state.unsqueeze(0), hx, cx)

            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            next_state, reward, terminated, truncated, _ = env.step(
                int(action.item())
            )
            done = bool(terminated or truncated)
            done = done or episode_length >= cfg["max_episode_length"]
            reward = max(min(reward, 1), -1)
            episode_reward += reward

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                next_state, _ = env.reset()

            state = torch.from_numpy(next_state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            _, value, _ = model(state.unsqueeze(0), hx, cx)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        gamma = cfg["gamma"]
        gae_lambda = cfg["gae_lambda"]
        entropy_coef = cfg["entropy_coef"]
        value_loss_coef = cfg["value_loss_coef"]
        max_grad_norm = cfg["max_grad_norm"]

        # Track average entropy over this rollout for logging
        if entropies:
            entropy_mean = torch.stack(entropies).mean().item()
        else:
            entropy_mean = 0.0

        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation (GAE)
            delta_t = rewards[i] + gamma * values[i + 1] - values[i]
            gae = gae * gamma * gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        # Lightweight per-episode logging similar to the reference implementation
        if done:
            episode_idx += 1
            recent_rewards.append(episode_reward)
            avg50 = float(np.mean(recent_rewards)) if recent_rewards else episode_reward

            print(
                f"Episode {episode_idx:5d} | Reward: {episode_reward:6.2f} | "
                f"Avg(50): {avg50:6.2f} | Entropy: {entropy_mean:7.4f}",
                flush=True,
            )

            episode_reward = 0.0

