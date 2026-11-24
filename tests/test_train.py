"""End-to-end trainer runs: 2 PyBoy workers, a few hundred steps, from the Pokedex state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from stable_baselines3.common.vec_env import VecEnvWrapper

from gameboy_rl.config import REPO_ROOT, Settings
from gameboy_rl.env import make_vec_env
from gameboy_rl.train import run
from tests.conftest import requires_rom

pytestmark = requires_rom
PRETRAINED = REPO_ROOT / "v2" / "runs" / "poke_26214400.zip"
PRETRAINED_STEPS = 26_214_400


def _cfg(tmp_path, **kw):
    # ep_length 256 -> n_steps 4: one episode per env inside 512 steps, so PWhiddy's TensorboardCallback fires.
    base = dict(n_envs=2, ep_length=256, total_steps=512, ckpt_freq=256, runs_dir=tmp_path, run_id="t")
    base.update(kw)
    return Settings(**base)


def _ckpts(tmp_path):
    return sorted((tmp_path / "t" / "checkpoints").glob("ckpt_*.zip"))


def test_smoke_writes_artifacts(tmp_path):
    assert run(_cfg(tmp_path)) == "done"
    rd = tmp_path / "t"
    assert _ckpts(tmp_path)
    health = json.loads((rd / "health.json").read_text())
    assert health["status"] in {"HEALTHY", "WARN"}
    assert (rd / "config.json").exists()
    assert (rd / "DONE").read_text() == "done"
    assert json.loads((rd / "progress.json").read_text())["badges_max"] >= 0
    assert list((rd / "tb").rglob("events.out.tfevents.*"))


def test_resume_from_pretrained_checkpoint(tmp_path):
    cfg = _cfg(tmp_path, resume_from=PRETRAINED, total_steps=PRETRAINED_STEPS + 64)
    assert run(cfg) == "done"
    assert int(_ckpts(tmp_path)[-1].stem.split("_")[-1]) >= PRETRAINED_STEPS + 64


def test_run_checkpoint_wins_over_resume_from(tmp_path):
    run(_cfg(tmp_path, total_steps=256))
    # second leg: the run's own checkpoint is used; a stale resume_from is ignored
    assert run(_cfg(tmp_path, total_steps=512, resume_from=Path("does/not/exist.zip"))) == "done"
    assert int(_ckpts(tmp_path)[-1].stem.split("_")[-1]) >= 512


def test_missing_resume_from_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(_cfg(tmp_path, resume_from=tmp_path / "nope.zip"))


def test_budget_stop_writes_done(tmp_path):
    assert run(_cfg(tmp_path, max_hours=0.0)) == "budget"
    assert (tmp_path / "t" / "DONE").read_text() == "budget"


class _CrashAfter(VecEnvWrapper):
    def __init__(self, venv, n):
        super().__init__(venv)
        self.n = n

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        self.n -= 1
        if self.n < 0:
            raise EOFError("worker died")
        return self.venv.step_wait()


def test_env_crash_marks_critical(tmp_path):
    status = run(_cfg(tmp_path), env_factory=lambda c: _CrashAfter(make_vec_env(c), 50))
    assert status == "critical"
    health = json.loads((tmp_path / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and "env_crash" in health["reasons"]
    assert _ckpts(tmp_path)
