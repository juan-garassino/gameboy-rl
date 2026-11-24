from __future__ import annotations

import numpy as np

from gameboy_rl.check_start import text_box_visible
from tests.conftest import requires_rom


def _overworld(seed=0):
    rng = np.random.default_rng(seed)
    shades = np.array([0, 85, 153, 255], np.uint8)
    return shades[rng.integers(0, 4, (144, 160))]


def _with_text_box(frame):
    f = frame.copy()
    f[96:144, :] = 255
    f[98:100, 2:158] = 0  # top border
    f[140:142, 2:158] = 0  # bottom border
    f[98:142, 2:4] = 0  # left border
    f[98:142, 156:158] = 0  # right border
    f[112:120, 8:100:3] = 0  # text line 1
    f[128:136, 8:60:3] = 0  # text line 2
    return f


def test_overworld_frame_has_no_text_box():
    assert not text_box_visible(_overworld())


def test_dialog_frame_detected():
    assert text_box_visible(_with_text_box(_overworld()))


def test_rgba_frame_accepted():
    f = _with_text_box(_overworld())
    rgba = np.stack([f, f, f, np.full_like(f, 255)], axis=-1)
    assert text_box_visible(rgba)


def test_blank_white_screen_is_not_a_text_box():
    # screen transitions fade to white; there is no border so it is not a dialog
    assert not text_box_visible(np.full((144, 160), 255, np.uint8))


@requires_rom
def test_check_start_default_state(tmp_path):
    from gameboy_rl.check_start import run
    from gameboy_rl.config import Settings

    result = run(Settings(runs_dir=tmp_path), n_actions=20)
    assert (tmp_path / "check_start.png").exists()
    assert result["state"].endswith("has_pokedex_nballs.state")
    assert result["first_frame_dialog"] is False
