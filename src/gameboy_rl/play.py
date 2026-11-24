"""Replay a checkpoint locally (headless PyBoy, one env) into an mp4 — Pokémon's trainer records no eval videos."""
from __future__ import annotations

import logging
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from gameboy_rl.config import Settings
from gameboy_rl.env import make_env

logger = logging.getLogger(__name__)

# One agent step = 24 emulator frames (0.4 s of game time); 8 fps plays back at ~3x real speed.
VIDEO_FPS = 8


def _screen(env) -> np.ndarray:
    return np.ascontiguousarray(env.unwrapped.pyboy.screen.ndarray[:, :, :3])


def play(cfg: Settings, model, steps: int, video_path: Path, deterministic: bool = False) -> dict:
    """Run `model` for `steps` agent steps from cfg.init_state and write every step's screen to video_path."""
    session = (cfg.run_dir / "play_session").resolve()  # the upstream env chdirs into v2/
    session.mkdir(parents=True, exist_ok=True)
    venv = DummyVecEnv([make_env(0, cfg, session)])
    raw = venv.envs[0]
    frames, info = [], {}
    try:
        obs = venv.reset()
        frames.append(_screen(raw))
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, infos = venv.step(action)
            info = infos[0]
            frames.append(_screen(raw))
            if done[0]:
                break
        badges = int(bin(raw.unwrapped.read_m(0xD356)).count("1"))
    finally:
        venv.close()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(video_path), frames, fps=VIDEO_FPS, macro_block_size=16, format="FFMPEG")
    stats = {"steps": len(frames) - 1, "badges": badges, "video": str(video_path)}
    logger.info("replay: %s", stats)
    return stats
