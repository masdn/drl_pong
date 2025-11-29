import os

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(
    results_dir: str,
    episode_rewards,
    cfg: dict,
    config_name: str,
    algo_name: str = "A2C",
    window: int = 20,
):
    """
    Create the same style of plots used in the Lunar Lander SARSA/REINFORCE scripts:

    1) Raw episode rewards over time.
    2) Moving-average (window) of rewards.
    3) Moving-average with +/- 1 std dev band.

    All plots are saved into `results_dir` as PNG files.
    """
    episode_rewards = np.asarray(episode_rewards, dtype=np.float32)
    episodes = np.arange(1, len(episode_rewards) + 1)

    # --- Plot 1 & 2: raw rewards + moving average ---
    plt.figure(figsize=(12, 5))

    # Plot 1: raw episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(episodes, episode_rewards, alpha=0.6, linewidth=0.8)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.title(
        f"{algo_name} on {cfg['env_name']}\n{config_name}",
        fontsize=11,
    )
    plt.grid(alpha=0.3)

    # Plot 2: moving average
    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= window:
        weights = np.ones(window, dtype=np.float32) / float(window)
        moving_avg = np.convolve(episode_rewards, weights, mode="valid")
        ma_episodes = np.arange(window, len(episode_rewards) + 1)
        plt.plot(ma_episodes, moving_avg, linewidth=2, color="orange")
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"{window}-Episode Moving Average", fontsize=13)
        plt.grid(alpha=0.3)

    plt.tight_layout()

    plot_file = os.path.join(results_dir, "training_plot.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"Training plot saved to: {plot_file}")
    plt.close()

    # --- Plot 3: learning curve with confidence interval ---
    plt.figure(figsize=(10, 6))
    if len(episode_rewards) >= window:
        moving_avg = []
        moving_std = []
        for i in range(window, len(episode_rewards) + 1):
            window_rewards = episode_rewards[i - window : i]
            moving_avg.append(np.mean(window_rewards))
            moving_std.append(np.std(window_rewards))

        moving_avg = np.array(moving_avg, dtype=np.float32)
        moving_std = np.array(moving_std, dtype=np.float32)
        ma_episodes = np.arange(window, len(episode_rewards) + 1)

        plt.plot(ma_episodes, moving_avg, linewidth=2, color="blue", label="Mean")
        plt.fill_between(
            ma_episodes,
            moving_avg - moving_std,
            moving_avg + moving_std,
            alpha=0.2,
            color="blue",
            label="±1 Std",
        )
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"Learning Curve ({window}-Episode Window)\n{config_name}", fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)

        learning_curve_file = os.path.join(results_dir, "learning_curve.png")
        plt.savefig(learning_curve_file, dpi=300, bbox_inches="tight")
        print(f"Learning curve saved to: {learning_curve_file}")
        plt.close()


