"""Wraps PWhiddy's v2 RedGymEnv (imported unmodified from v2/) for our trainer.

Upstream quirks handled here, not by editing v2/:
- red_gym_env_v2 opens "events.json" relative to the cwd -> construct it inside v2/.
- its modules import each other as top-level names -> v2/ goes on sys.path.
- StreamWrapper broadcasts to PWhiddy's public websocket map -> only when cfg.stream is True.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Iterator

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from gameboy_rl.config import V2_DIR, Settings

logger = logging.getLogger(__name__)

# baseline_fast_v2.py env_config, minus the paths and episode length that come from Settings.
UPSTREAM_ENV_CONFIG = {
    "headless": True, "save_final_state": False, "early_stop": False, "action_freq": 24,
    "print_rewards": True, "save_video": False, "fast_video": True, "debug": False,
    "reward_scale": 0.5, "explore_weight": 0.25,
}


@contextlib.contextmanager
def _in_v2_dir() -> Iterator[None]:
    if str(V2_DIR) not in sys.path:
        sys.path.insert(0, str(V2_DIR))
    prev = os.getcwd()
    os.chdir(V2_DIR)
    try:
        yield
    finally:
        os.chdir(prev)


def env_config(cfg: Settings, session_path: Path) -> dict:
    return {**UPSTREAM_ENV_CONFIG, "init_state": str(cfg.init_state_path), "gb_path": str(cfg.rom_file),
            "max_steps": cfg.ep_length, "session_path": Path(session_path).resolve()}


def make_red_env(cfg: Settings, session_path: Path) -> gym.Env:
    """A bare RedGymEnv (upstream class, upstream reward logic)."""
    if not cfg.rom_file.exists():
        raise FileNotFoundError(f"ROM not found: {cfg.rom_file}")
    if not cfg.init_state_path.exists():
        raise FileNotFoundError(f"init state not found: {cfg.init_state_path}")
    Path(session_path).mkdir(parents=True, exist_ok=True)
    with _in_v2_dir():
        from red_gym_env_v2 import RedGymEnv

        return RedGymEnv(env_config(cfg, session_path))


class ButtonsWrapper(gym.Wrapper):
    """Adds info["buttons"] = one-hot of the Discrete action, so HealthMonitor's button-spam rule works."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        buttons = np.zeros(self.action_space.n, dtype=np.int8)
        buttons[int(action)] = 1
        info = dict(info, buttons=buttons)
        return obs, reward, terminated, truncated, info


class AgentStatsTail(gym.Wrapper):
    """Exposes only the newest entry of RedGymEnv.agent_stats to the trainer process.

    PWhiddy's TensorboardCallback does get_attr("agent_stats") and keeps stats[-1]; unwrapped, that pickles
    every env's whole per-step list (up to ep_length dicts each) across the pipe at each episode end.
    SubprocVecEnv.get_attr resolves the name on the outermost wrapper, so this property answers instead.
    """

    @property
    def agent_stats(self) -> list[dict]:
        return self.env.unwrapped.agent_stats[-1:]


def import_upstream(module: str):
    """Import a v2/ module (they import each other as top-level names)."""
    import importlib

    with _in_v2_dir():
        return importlib.import_module(module)


def make_env(rank: int, cfg: Settings, session_path: Path) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        if cfg.game == "silver":
            from gameboy_rl.silver import SilverEnv

            env = Monitor(ButtonsWrapper(SilverEnv(cfg)))
            env.reset(seed=cfg.seed + rank)
            return env
        env = make_red_env(cfg, session_path)
        if cfg.stream:
            StreamWrapper = import_upstream("stream_agent_wrapper").StreamWrapper
            env = StreamWrapper(env, stream_metadata={"user": cfg.stream_user, "env_id": rank,
                                                      "color": "#447799", "extra": ""})
        env = Monitor(AgentStatsTail(ButtonsWrapper(env)))
        env.reset(seed=cfg.seed + rank)
        return env

    return _init


def make_vec_env(cfg: Settings, session_path: Path | None = None) -> VecEnv:
    """One PyBoy per worker process (SubprocVecEnv), like baseline_fast_v2."""
    session_path = session_path or cfg.run_dir / "session"
    set_random_seed(cfg.seed)
    if cfg.stream:
        logger.warning("GBRL_STREAM on: broadcasting coordinates to PWhiddy's public map")
    return SubprocVecEnv([make_env(i, cfg, session_path) for i in range(cfg.n_envs)])
