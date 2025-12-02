### Implemented A2C for Pong in `a2c.py`

- Built a **CNN-based `ActorCriticNet`** suited for Atari Pong (84×84 stacked frames, policy + value heads).
- Implemented a **single-env `A2CAgent`** that:
  - Uses Gymnasium + ALE with `AtariPreprocessing` and `FrameStack` wrappers for `ALE/Pong-v5`.
  - Collects full-episode trajectories, computes discounted returns and advantages, and performs A2C updates (policy loss + value loss − entropy bonus).
  - Logs episode reward, 50-episode moving average, losses, and entropy during training.
- Added a **config-driven main** that:
  - Scans for `global_config-pong-a2c*.json` in the project root.
  - For each config, sets seeds, runs training, and saves rewards and a summary to a results folder (mirroring your Lunar Lander setup).


