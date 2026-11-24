"""PWhiddy's poke_26214400.zip was pickled under Python 3.11: its cloudpickled clip_range lambda segfaults on 3.12
when called. load_model must rebuild the schedules from floats (custom_objects). No ROM needed."""
from __future__ import annotations

from stable_baselines3 import PPO

from gameboy_rl.config import REPO_ROOT
from gameboy_rl.train import SCHEDULE_OVERRIDES

PRETRAINED = REPO_ROOT / "v2" / "runs" / "poke_26214400.zip"


def test_pretrained_schedules_are_rebuilt():
    model = PPO.load(PRETRAINED, device="cpu", custom_objects=SCHEDULE_OVERRIDES)
    assert model.num_timesteps == 26_214_400
    assert model.clip_range(1.0) == 0.2
    assert model.lr_schedule(1.0) == 3e-4
    assert model.observation_space["events"].n == 2488  # matches red_gym_env_v2's event flag range
