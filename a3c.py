import os
import json
import time
import threading
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

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


class A3CWorker(threading.Thread):
    """
    A3C worker thread.

    Each worker:
    - Has its own env and local copy of the model.
    - Runs episodes, computes policy/value losses and entropy bonus.
    - Applies gradients to the shared global model (with a lock) and syncs
      its local parameters from the global model.
    """

    def __init__(
        self,
        worker_id: int,
        cfg: dict,
        global_model: ActorCriticNet,
        optimizer: optim.Optimizer,
        counters: GlobalCounters,
        rewards_list: list,
        rewards_lock: threading.Lock,
    ):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.cfg = cfg
        self.global_model = global_model
        self.optimizer = optimizer
        self.counters = counters
        self.rewards_list = rewards_list
        self.rewards_lock = rewards_lock

        env_name = cfg["env_name"]
        base_seed = cfg.get("seed", 0) + 1000 * worker_id
        frame_skip = cfg.get("frame_skip", 4)
        frame_stack = cfg.get("frame_stack", 4)
        grayscale = cfg.get("grayscale", True)
        scale_obs = cfg.get("scale_obs", True)

        # Local environment for this worker
        self.env = make_pong_env(
            env_name=env_name,
            seed=base_seed,
            render_mode=None,
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            grayscale=grayscale,
            scale_obs=scale_obs,
        )

        obs_shape = self.env.observation_space.shape
        num_actions = self.env.action_space.n
        fc_size = cfg.get("fc_size", 512)

        # Local model starts as a copy of the global model
        self.local_model = ActorCriticNet(obs_shape, num_actions, fc_size=fc_size).to(
            device
        )
        self.local_model.load_state_dict(self.global_model.state_dict())

        # Hyperparameters
        self.gamma = cfg.get("gamma", 0.99)
        self.n_step = cfg.get("n_step", 0)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.t_max = cfg.get("t_max", 5)
        self.max_steps_per_episode = cfg.get("max_steps", 10000)

    def run(self):
        while True:
            # Get next episode index or exit
            with self.counters.lock:
                if self.counters.episode >= self.cfg["num_episodes"]:
                    break
                episode_idx = self.counters.episode
                self.counters.episode += 1

            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            steps_in_episode = 0

            while not done and steps_in_episode < self.max_steps_per_episode:
                log_probs = []
                values = []
                rewards = []
                entropies = []
                dones = []

                # Collect up to t_max steps or until episode ends
                for t in range(self.t_max):
                    state_arr = np.array(state, copy=False)
                    state_tensor = (
                        torch.from_numpy(state_arr)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )

                    logits, value = self.local_model(state_tensor)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()

                    next_state, reward, terminated, truncated, _ = self.env.step(
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
                    self.counters.total_steps += 1
                    state = next_state

                    if done or steps_in_episode >= self.max_steps_per_episode:
                        break

                if len(rewards) > 0:
                    # Compute returns and advantages (CPU tensors)
                    returns, advantages = compute_returns_and_advantages(
                        rewards, values, dones, self.gamma, self.n_step
                    )

                    log_probs_tensor = torch.stack(log_probs)
                    entropies_tensor = torch.stack(entropies)

                    raw_advantages = advantages.clone()
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    policy_loss = -(
                        log_probs_tensor * advantages.detach()
                    ).mean()
                    value_loss = raw_advantages.pow(2).mean()
                    entropy_mean = entropies_tensor.mean()

                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_mean
                    )

                    # Apply gradients to shared global model
                    # 1) Clear global grads
                    self.optimizer.zero_grad()
                    # 2) Clear local grads (otherwise they accumulate across updates)
                    for p in self.local_model.parameters():
                        if p.grad is not None:
                            p.grad = None

                    # 3) Backprop on local model and clip its gradients
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.local_model.parameters(), self.max_grad_norm
                    )

                    # 4) Copy grads from local to global and take an optimizer step
                    with self.counters.lock:
                        for global_param, local_param in zip(
                            self.global_model.parameters(),
                            self.local_model.parameters(),
                        ):
                            if local_param.grad is not None:
                                if global_param.grad is None:
                                    global_param.grad = local_param.grad.clone()
                                else:
                                    global_param.grad.copy_(local_param.grad)
                        self.optimizer.step()
                        # Sync local parameters from updated global model
                        self.local_model.load_state_dict(
                            self.global_model.state_dict()
                        )

            # Record episode reward
            with self.rewards_lock:
                self.rewards_list.append(episode_reward)

            if episode_idx % self.cfg.get("log_interval", 10) == 0:
                avg_reward = np.mean(self.rewards_list[-50:])
                print(
                    f"[Worker {self.worker_id}] Episode {episode_idx}/{self.cfg['num_episodes']} "
                    f"| Reward: {episode_reward:.2f} | Avg(50): {avg_reward:.2f}",
                    flush=True,
                )


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

    optimizer = optim.RMSprop(
        global_model.parameters(),
        lr=cfg.get("learning_rate", 2.5e-4),
        alpha=0.99,
        eps=1e-5,
    )

    num_workers = cfg.get("num_workers", 4)
    counters = GlobalCounters()
    rewards_list = []
    rewards_lock = threading.Lock()

    print(
        f"Starting A3C with {num_workers} worker threads for "
        f"{cfg['num_episodes']} total episodes.\n"
    )

    workers = []
    for worker_id in range(num_workers):
        w = A3CWorker(
            worker_id=worker_id,
            cfg=cfg,
            global_model=global_model,
            optimizer=optimizer,
            counters=counters,
            rewards_list=rewards_list,
            rewards_lock=rewards_lock,
        )
        w.start()
        workers.append(w)

    for w in workers:
        w.join()

    # Training finished
    average_reward = float(np.mean(rewards_list)) if rewards_list else 0.0
    std_reward = float(np.std(rewards_list)) if rewards_list else 0.0
    max_reward = float(np.max(rewards_list)) if rewards_list else 0.0
    min_reward = float(np.min(rewards_list)) if rewards_list else 0.0

    elapsed_time = 0.0  # could track with a timer if desired

    print(f"\n{'=' * 60}")
    print(f"A3C Training completed.")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Std Reward: {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Total Episodes: {len(rewards_list)}")
    print(f"Total Steps: {counters.total_steps}")
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
        "total_steps": int(counters.total_steps),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Global A3C checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
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


