"""Grafting a new input: identical behaviour before training, on a tiny network (no emulator)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gameboy_rl.graft import graft, needs_graft


def _space(story: bool):
    d = {"screens": Box(0, 255, (72, 80, 3), np.uint8), "visited": Box(0, 255, (48, 48, 1), np.uint8),
         "state": Box(-1, 1, (4,), np.float32), "recent_actions": MultiDiscrete([6, 6])}
    if story:
        d["story"] = Box(0, 1, (5,), np.float32)
    return Dict(d)


def _model(story: bool, seed: int):
    class E(gym.Env):
        observation_space, action_space = _space(story), Discrete(6)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, False, {}

    torch.manual_seed(seed)
    return PPO("MultiInputPolicy", DummyVecEnv([E]), n_steps=8, batch_size=8, device="cpu", seed=seed)


def test_a_grafted_network_decides_and_values_exactly_like_the_old_one():
    old, new = _model(False, 1), _model(True, 2)
    assert needs_graft(old, new)
    graft(old, new)
    rng = np.random.default_rng(0)
    for _ in range(4):
        obs = {k: np.stack([s.sample() for _ in range(3)]) for k, s in _space(True).spaces.items()}
        obs["story"] = rng.random((3, 5)).astype(np.float32)  # any story input: the new weights are zero
        chw = {k: np.transpose(v, (0, 3, 1, 2)) if v.ndim == 4 else v for k, v in obs.items()}
        t = lambda d: {k: torch.as_tensor(v).float() for k, v in d.items()}
        with torch.no_grad():
            a = new.policy.get_distribution(t(chw)).distribution.logits
            b = old.policy.get_distribution(t({k: v for k, v in chw.items() if k != "story"})).distribution.logits
            assert torch.allclose(a, b, atol=1e-5)
            va, vb = new.policy.predict_values(t(chw)), old.policy.predict_values(t({k: v for k, v in chw.items() if k != "story"}))
            assert torch.allclose(va, vb, atol=1e-5)


def test_removing_an_input_is_refused():
    with pytest.raises(ValueError, match="inputs removed"):
        graft(_model(True, 1), _model(False, 2))


def test_adam_counters_leave_the_gpu_after_a_load():
    from types import SimpleNamespace

    import torch

    from gameboy_rl.train import keep_counters_on_cpu

    class _OnGpu(torch.Tensor):
        moved = 0

        def cpu(self):
            _OnGpu.moved += 1
            return torch.tensor(7.0)

    state = {"a": {"step": torch.tensor(7.0).as_subclass(_OnGpu)}, "b": {}}  # "b" has not been updated yet
    keep_counters_on_cpu(SimpleNamespace(policy=SimpleNamespace(optimizer=SimpleNamespace(state=state))))
    assert _OnGpu.moved == 1 and state["a"]["step"].item() == 7.0 and state["b"] == {}
