"""ButtonsWrapper (info["buttons"] for the button-spam rule) and ProgressCallback (Pokemon progress per rollout)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gameboy_rl.env import ButtonsWrapper
from gameboy_rl.monitoring.progress import ProgressCallback


class _StubRedEnv(gym.Env):
    """Discrete(7) like RedGymEnv; returns an empty info dict like PWhiddy's step()."""

    action_space = spaces.Discrete(7)
    observation_space = spaces.Box(0, 255, (2,), np.uint8)

    def reset(self, seed=None, options=None):
        return np.zeros(2, np.uint8), {}

    def step(self, action):
        return np.zeros(2, np.uint8), 1.0, False, False, {}


def test_buttons_one_hot():
    env = ButtonsWrapper(_StubRedEnv())
    env.reset()
    _, reward, _, _, info = env.step(4)
    assert reward == 1.0
    assert info["buttons"].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert info["buttons"].dtype == np.int8


class _StubVecEnv:
    """Answers env_method/get_attr like a SubprocVecEnv of two RedGymEnvs."""

    def __init__(self):
        self.calls = []

    def env_method(self, name, *args, indices=None, **kwargs):
        self.calls.append(("env_method", name))
        return {"get_badges": [1, 3], "get_all_events_reward": [10, 20]}[name]

    def get_attr(self, name, indices=None):
        self.calls.append(("get_attr", name))
        maps = [np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)]
        maps[0][0, :2] = 255
        maps[1][1, :] = 255
        return {"explore_map": maps, "max_map_progress": [2, 5]}[name]


class _StubLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value, exclude=None):
        self.records[key] = value


class _StubModel:
    def __init__(self, env):
        self.env = env
        self.logger = _StubLogger()
        self.num_timesteps = 0

    def get_env(self):
        return self.env


def test_progress_logged_each_rollout():
    venv = _StubVecEnv()
    cb = ProgressCallback()
    model = _StubModel(venv)
    cb.init_callback(model)
    cb.on_rollout_end()
    r = model.logger.records
    assert r["pokemon/badges_max"] == 3 and r["pokemon/badges_mean"] == 2.0
    assert r["pokemon/events_max"] == 20 and r["pokemon/events_mean"] == 15.0
    assert r["pokemon/explore_tiles_max"] == 4 and r["pokemon/explore_tiles_mean"] == 3.0
    assert r["pokemon/map_progress_max"] == 5
    assert cb.last["badges_max"] == 3


def test_progress_never_breaks_training():
    class _Broken(_StubVecEnv):
        def env_method(self, name, *args, **kwargs):
            raise EOFError("worker gone")

    cb = ProgressCallback()
    model = _StubModel(_Broken())
    cb.init_callback(model)
    cb.on_rollout_end()  # must not raise
    assert model.logger.records == {}


class _StubStatsEnv(_StubRedEnv):
    def __init__(self):
        self.agent_stats = [{"step": i} for i in range(1000)]


def test_agent_stats_tail_only_exposes_last_entry():
    from stable_baselines3.common.monitor import Monitor

    from gameboy_rl.env import AgentStatsTail

    env = Monitor(AgentStatsTail(_StubStatsEnv()))
    # SubprocVecEnv.get_attr does getattr(outermost_env, name); TensorboardCallback then takes stats[-1].
    assert getattr(env, "agent_stats") == [{"step": 999}]
    assert len(env.unwrapped.agent_stats) == 1000
