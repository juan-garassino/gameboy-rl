from __future__ import annotations

import numpy as np

from tests.conftest import requires_rom
from gameboy_rl.config import Settings


class _RandomModel:
    def __init__(self, n_actions):
        self.n = n_actions

    def predict(self, obs, deterministic=True):
        return np.array([np.random.randint(self.n)]), None


@requires_rom
def test_play_writes_video(tmp_path):
    from gameboy_rl.play import play

    video = tmp_path / "pokemon.mp4"
    stats = play(Settings(runs_dir=tmp_path, run_id="p"), model=_RandomModel(7), steps=15, video_path=video)
    assert video.exists() and video.stat().st_size > 0
    assert stats["steps"] == 15
    assert "badges" in stats
