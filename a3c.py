import os
import json
import time
import threading  # kept for legacy A3CWorker class (no longer used in main path)
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from a2c import make_pong_env, ActorCriticNet, device


class GlobalCounters:
    """Simple mutable counters shared across worker threads."""

    def __init__(self):
        self.episode = 0
        self.total_steps = 0
        self.lock = threading.Lock()


def compute_returns_and_advantages(rewards, values, dones, gamma: float, n_step: int):
    """
    n-step / full-episode return and advantage computation (CPU version),
    structurally similar to the A2C helper in `a2c.py`.
    """
    T = len(rewards)

    # Make sure we stay on the same device as the value estimates (CPU or CUDA)
    device_ = values[0].device
    returns = torch.zeros(T, dtype=torch.float32, device=device_)
    values_tensor = torch.stack(values).squeeze(-1).to(device_)  # (T,)

    if not n_step or n_step <= 0:
        R = 0.0
        for t in reversed(range(T)):
            R = rewards[t] + gamma * R
            returns[t] = R
    else:
        n = int(n_step)
        for t in range(T):
            R = 0.0
            discount = 1.0
            done_in_window = False

            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                R += discount * rewards[idx]
                discount *= gamma
                if dones[idx]:
                    done_in_window = True
                    break

            idx_bootstrap = t + n
            if (not done_in_window) and (idx_bootstrap < T):
                R += discount * values_tensor[idx_bootstrap].detach()

            returns[t] = R

    advantages = returns - values_tensor
    return returns, advantages


def a3c_worker_process(
    worker_id: int,
    cfg: dict,
    global_model: ActorCriticNet,
    episode_counter: mp.Value,
    step_counter: mp.Value,
    update_lock: mp.Lock,
    stats_queue: mp.Queue,
):
    """
    Multiprocessing A3C worker.

    Each process:
    - Has its own env and local model.
    - Pulls episode indices from a shared counter.
    - Runs t_max rollouts, computes losses, and applies grads to the shared model.
    """

    # Per-process seeding
    base_seed = cfg.get("seed", 0) + 1000 * worker_id
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    env_name = cfg["env_name"]
    frame_skip = cfg.get("frame_skip", 4)
    frame_stack = cfg.get("frame_stack", 4)
    grayscale = cfg.get("grayscale", True)
    scale_obs = cfg.get("scale_obs", True)

    env = make_pong_env(
        env_name=env_name,
        seed=base_seed,
        render_mode=None,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        grayscale=grayscale,
        scale_obs=scale_obs,
    )

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    fc_size = cfg.get("fc_size", 512)

    local_model = ActorCriticNet(obs_shape, num_actions, fc_size=fc_size).to(device)
    local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.RMSprop(
        global_model.parameters(),
        lr=cfg.get("learning_rate", 2.5e-4),
        alpha=0.99,
        eps=1e-5,
    )

    gamma = cfg.get("gamma", 0.99)
    n_step = cfg.get("n_step", 0)
    entropy_coef = cfg.get("entropy_coef", 0.01)
    value_loss_coef = cfg.get("value_loss_coef", 0.5)
    max_grad_norm = cfg.get("max_grad_norm", 0.5)
    t_max = cfg.get("t_max", 5)
    max_steps_per_episode = cfg.get("max_steps", 10000)

    while True:
        # Get next global episode index
        with episode_counter.get_lock():
            if episode_counter.value >= cfg["num_episodes"]:
                break
            episode_idx = episode_counter.value  # 0-based
            episode_counter.value += 1

        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps_in_episode = 0
        last_entropy_mean = 0.0

        while not done and steps_in_episode < max_steps_per_episode:
            log_probs = []
            values = []
            rewards = []
            entropies = []
            dones = []

            # Collect up to t_max steps or until episode ends
            for t in range(t_max):
                state_arr = np.array(state, copy=False)
                state_tensor = (
                    torch.from_numpy(state_arr).float().unsqueeze(0).to(device)
                )

                logits, value = local_model(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                next_state, reward, terminated, truncated, _ = env.step(
                    int(action.item())
                )
                done = bool(terminated or truncated)

                log_probs.append(log_prob.squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(float(reward))
                entropies.append(entropy.squeeze(0))
                dones.append(done)

                episode_reward += float(reward)
                steps_in_episode += 1
                with step_counter.get_lock():
                    step_counter.value += 1
                state = next_state

                if done or steps_in_episode >= max_steps_per_episode:
                    break

            if len(rewards) > 0:
                # Compute returns and advantages on same device as values
                returns, advantages = compute_returns_and_advantages(
                    rewards, values, dones, gamma, n_step
                )

                # Occasional debug: log raw advantage stats from worker 0
                if (episode_idx + 1) % 1000 == 0 and worker_id == 0:
                    adv_mean_dbg = advantages.mean().item()
                    if advantages.numel() > 1:
                        adv_std_dbg = advantages.std(unbiased=False).item()
                    else:
                        adv_std_dbg = 0.0
                    print(
                        f"[DebugAdv] ep {episode_idx + 1} | adv_mean={adv_mean_dbg:.3e} "
                        f"adv_std={adv_std_dbg:.3e}",
                        flush=True,
                    )

                log_probs_tensor = torch.stack(log_probs)
                entropies_tensor = torch.stack(entropies)

                raw_advantages = advantages.clone()
                # Normalize advantages for stability, but guard against tiny batches / NaNs
                adv_mean = advantages.mean()
                if advantages.numel() > 1:
                    adv_std = advantages.std(unbiased=False)
                else:
                    adv_std = torch.tensor(1.0, device=advantages.device)
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)

                policy_loss = -(
                    log_probs_tensor * advantages.detach()
                ).mean()
                value_loss = raw_advantages.pow(2).mean()
                entropy_mean = entropies_tensor.mean()
                last_entropy_mean = float(entropy_mean.item())

                loss = (
                    policy_loss
                    + value_loss_coef * value_loss
                    - entropy_coef * entropy_mean
                )

                # Clear grads and backprop on local model
                optimizer.zero_grad()
                for p in local_model.parameters():
                    if p.grad is not None:
                        p.grad = None

                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

                # Optional: occasional gradient norm debug from worker 0
                if (episode_idx + 1) % 1000 == 0 and worker_id == 0:
                    total_norm_sq = 0.0
                    for p in local_model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm_sq += param_norm.item() ** 2
                    grad_norm = total_norm_sq ** 0.5
                    print(f"[Debug] Worker {worker_id} grad_norm={grad_norm:.4f}", flush=True)

                # Copy local grads to shared global model and step optimizer
                with update_lock:
                    for global_param, local_param in zip(
                        global_model.parameters(), local_model.parameters()
                    ):
                        if local_param.grad is not None:
                            if global_param.grad is None:
                                global_param.grad = local_param.grad.clone()
                            else:
                                global_param.grad.copy_(local_param.grad)
                    optimizer.step()
                    # Sync local parameters from updated global model
                    local_model.load_state_dict(global_model.state_dict())

        # Push episode stats to parent (episode_idx is 0-based; +1 for human-readable)
        stats_queue.put(
            {
                "ep": int(episode_idx + 1),
                "worker": int(worker_id),
                "reward": float(episode_reward),
                "length": int(steps_in_episode),
                "entropy": float(last_entropy_mean),
            }
        )

    env.close()


def stats_aggregator(num_episodes: int, stats_queue: mp.Queue, log_interval: int = 50):
    """
    Consume per-episode stats from all workers and print a clean global log.

    Prints one line every `log_interval` global episodes with:
    - last episode reward
    - global Avg100 reward
    - episode length
    - last entropy value
    """
    rewards = []
    last_stats = None

    for global_ep in range(1, num_episodes + 1):
        stats = stats_queue.get()
        last_stats = stats
        rewards.append(stats["reward"])

        if global_ep % log_interval == 0:
            if len(rewards) >= 100:
                avg100 = float(np.mean(rewards[-100:]))
            else:
                avg100 = float(stats["reward"])

            print(
                f"[Global Episode {global_ep}/{num_episodes}] "
                f"Last: {stats['reward']:.2f} (W{stats['worker']}) | "
                f"Avg100: {avg100:.2f} | "
                f"Len: {stats['length']} | "
                f"Entropy: {stats['entropy']:.3f}",
                flush=True,
            )

    rewards_array = np.array(rewards, dtype=np.float32) if rewards else np.array([])
    return rewards_array, last_stats


def run_a3c_from_config(config_path: str):
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    print(f"\n{'#' * 80}")
    print(f"# A3C - Processing config: {config_name}")
    print(f"{'#' * 80}\n")

    with open(config_path, "r") as f:
        cfg = json.load(f)

    print("Loaded A3C config:")
    print(json.dumps(cfg, indent=2))
    print()

    np.random.seed(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))

    # Build a dummy env to infer shapes
    env = make_pong_env(
        env_name=cfg["env_name"],
        seed=cfg.get("seed", 0),
        render_mode=None,
        frame_skip=cfg.get("frame_skip", 4),
        frame_stack=cfg.get("frame_stack", 4),
        grayscale=cfg.get("grayscale", True),
        scale_obs=cfg.get("scale_obs", True),
    )
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    env.close()

    fc_size = cfg.get("fc_size", 512)
    global_model = ActorCriticNet(obs_shape, num_actions, fc_size=fc_size).to(device)
    global_model.share_memory()

    num_workers = cfg.get("num_workers", 4)

    # Shared counters and lock for multiprocessing
    episode_counter = mp.Value("i", 0)
    step_counter = mp.Value("i", 0)
    update_lock = mp.Lock()
    stats_queue = mp.Queue()

    print(
        f"Starting A3C with {num_workers} worker processes for "
        f"{cfg['num_episodes']} total episodes.\n"
    )

    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=a3c_worker_process,
            args=(
                worker_id,
                cfg,
                global_model,
                episode_counter,
                step_counter,
                update_lock,
                stats_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Aggregate stats and print global logs while workers run
    log_interval = cfg.get("log_interval", 50)
    rewards_array, last_stats = stats_aggregator(
        cfg["num_episodes"], stats_queue, log_interval=log_interval
    )

    for p in processes:
        p.join()

    average_reward = float(rewards_array.mean()) if rewards_array.size > 0 else 0.0
    std_reward = float(rewards_array.std()) if rewards_array.size > 0 else 0.0
    max_reward = float(rewards_array.max()) if rewards_array.size > 0 else 0.0
    min_reward = float(rewards_array.min()) if rewards_array.size > 0 else 0.0

    elapsed_time = 0.0  # could track with a timer if desired

    print(f"\n{'=' * 60}")
    print(f"A3C Training completed.")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Total Episodes: {episode_counter.value}")
    print(f"Total Steps: {step_counter.value}")
    print(f"{'=' * 60}\n")

    # Optionally, save a checkpoint for the global model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoints_dir, f"a3c_{config_name}_{timestamp}.pt"
    )
    checkpoint = {
        "model_state_dict": global_model.state_dict(),
        "config": cfg,
        "average_reward": average_reward,
        "total_steps": int(step_counter.value),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Global A3C checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    # Use 'spawn' for safety with CUDA and Gym/ALE
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # Start method may already be set; ignore.
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "global_config-pong-a3c.json")

    if not os.path.isfile(config_path):
        print(
            "Config file 'global_config-pong-a3c.json' not found. "
            "Create it based on 'global_config-pong-a2c.json' and add A3C-specific keys "
            "like 'num_workers' and 't_max'."
        )
        raise SystemExit(1)

    run_a3c_from_config(config_path)


