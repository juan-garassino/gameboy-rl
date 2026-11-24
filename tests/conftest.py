"""Shared fixtures.

- `clean_env`: strips GBRL_* vars so Settings tests see defaults (motivated by src/gameboy_rl/config.py env_prefix).
- `requires_rom`: skip marker for tests that boot PyBoy (roms/ is gitignored; the ROM never ships with the repo).
"""
from __future__ import annotations

import os

import pytest

from gameboy_rl.config import REPO_ROOT

ROM = REPO_ROOT / "roms" / "pokemon_red_version.gb"
requires_rom = pytest.mark.skipif(not ROM.exists(), reason="roms/pokemon_red_version.gb not present")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("GBRL_") or k == "MLFLOW_TRACKING_URI":
            monkeypatch.delenv(k, raising=False)
