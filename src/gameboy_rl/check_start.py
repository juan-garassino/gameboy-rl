"""`gameboy-rl check-start`: boot the configured state headless, take random actions, look for a text box.

Pokemon Red draws every dialog / battle text as a white box over the bottom 6 tile rows (y 96-143) with a
black frame. Text sits on tile rows 14 and 16, so tile rows 13 and 15 (y 104-111, 120-127) inside the frame
are pure white. Calibrated on a real "OAK: Come see me sometimes." frame: both bands 100% white.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from gameboy_rl.config import Settings
from gameboy_rl.env import make_red_env

logger = logging.getLogger(__name__)

WHITE, DARK = 200, 100
BLANK_BANDS = ((104, 112), (120, 128))  # tile rows 13 and 15, inside the box
INNER_X = slice(8, 152)


def _gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    return frame[:, :, 0] if frame.ndim == 3 else frame


def text_box_visible(frame: np.ndarray) -> bool:
    g = _gray(frame)
    bands_white = all((g[y0:y1, INNER_X] > WHITE).mean() >= 0.95 for y0, y1 in BLANK_BANDS)
    left_frame = (g[100:140, 0:8] < DARK).mean() >= 0.1
    right_frame = (g[100:140, 152:160] < DARK).mean() >= 0.1
    return bool(bands_white and left_frame and right_frame)


def _save(frame: np.ndarray, path: Path) -> None:
    rgb = np.asarray(frame)[:, :, :3] if np.asarray(frame).ndim == 3 else np.asarray(frame)
    Image.fromarray(rgb).resize((320, 288), Image.NEAREST).save(path)


def run(cfg: Settings, n_actions: int = 200, seed: int = 0) -> dict:
    """Saves <runs_dir>/check_start_first.png (after 1 action) and check_start.png (after n_actions)."""
    out_dir = Path(cfg.runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        env = make_red_env(cfg, Path(tmp) / "session")
        try:
            env.reset(seed=seed)
            env.step(int(rng.integers(env.action_space.n)))  # load_state leaves the screen buffer stale
            first = env.pyboy.screen.ndarray.copy()
            dialog_steps = int(text_box_visible(first))
            for _ in range(n_actions - 1):
                env.step(int(rng.integers(env.action_space.n)))
                dialog_steps += int(text_box_visible(env.pyboy.screen.ndarray))
            last = env.pyboy.screen.ndarray.copy()
            x, y, map_n = env.get_game_coords()
            party = env.read_m(0xD163)
        finally:
            env.pyboy.stop(save=False)
    _save(first, out_dir / "check_start_first.png")
    _save(last, out_dir / "check_start.png")
    final_dialog = text_box_visible(last)
    return {
        "state": str(cfg.init_state_path),
        "first_frame_dialog": text_box_visible(first),
        "final_frame_dialog": final_dialog,
        "dialog_frac": dialog_steps / n_actions,
        "verdict": "dialog" if final_dialog else "overworld",
        "map": int(map_n), "x": int(x), "y": int(y), "party_size": int(party),
        "png": str(out_dir / "check_start.png"),
    }
