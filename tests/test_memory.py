"""The LSTM option: it builds, carries memory within a game and forgets between games. Tiny network, no emulator."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.common.vec_env import DummyVecEnv

from gameboy_rl.config import Settings
from gameboy_rl.silver_eval import Chooser
from gameboy_rl.train import algorithm, build_model


class E(gym.Env):
    observation_space = Dict({"screens": Box(0, 255, (48, 48, 3), np.uint8), "state": Box(-1, 1, (4,), np.float32)})
    action_space = Discrete(6)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 1.0, False, False, {}


def _cfg(**kw):
    return Settings(game="silver", memory="lstm", device="cpu", n_envs=2, ep_length=1024, **kw)  # n_steps 16


def test_memory_is_off_unless_asked():
    from stable_baselines3 import PPO

    assert Settings(game="silver").memory == "none" and algorithm(Settings(game="silver")) is PPO


def test_an_lstm_model_trains_a_few_steps_and_plays_with_memory():
    model = build_model(_cfg(), DummyVecEnv([E, E]))
    model.learn(total_timesteps=64)
    c = Chooser(model)
    assert c.recurrent
    obs = {k: np.asarray(v) for k, v in E.observation_space.sample().items()}
    c(obs)
    assert c.memory is not None and not c.first.any()
    mem = c.memory
    c(obs)
    assert len(mem) == len(c.memory) == 2  # (h, c) is passed along
    c.reset()
    assert c.memory is None and c.first.all()


def test_an_lstm_checkpoint_loads_and_continues(tmp_path):
    cfg = _cfg()
    env = DummyVecEnv([E, E])
    m = build_model(cfg, env)
    m.learn(64)
    m.save(tmp_path / "m")
    from gameboy_rl.train import load_model

    again = load_model(tmp_path / "m.zip", cfg, env)
    assert again.num_timesteps == m.num_timesteps
