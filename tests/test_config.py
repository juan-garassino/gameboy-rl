from __future__ import annotations

from pathlib import Path

import pytest

from gameboy_rl.config import REPO_ROOT, Settings


def test_defaults():
    s = Settings()
    assert s.n_envs == 32
    assert s.init_state == Path("has_pokedex_nballs.state")
    assert s.rom_path == Path("roms/pokemon_red_version.gb")
    assert s.stream is False
    assert s.resume_from is None
    assert s.max_hours == 6.0 and s.max_eur == 8.0 and s.usd_per_eur == 1.05
    assert s.n_steps == 163_840 // 64  # baseline_fast_v2: ep_length // 64


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("GBRL_N_ENVS", "4")
    monkeypatch.setenv("GBRL_INIT_STATE", "init.state")
    monkeypatch.setenv("GBRL_STREAM", "true")
    s = Settings()
    assert s.n_envs == 4 and s.init_state == Path("init.state") and s.stream is True


def test_relative_paths_resolve_against_repo_root():
    s = Settings()
    assert s.init_state_path == REPO_ROOT / "has_pokedex_nballs.state"
    assert s.init_state_path.exists()
    assert s.rom_file == REPO_ROOT / "roms" / "pokemon_red_version.gb"
    assert s.resume_path is None


def test_absolute_paths_kept(tmp_path):
    s = Settings(rom_path=tmp_path / "r.gb", resume_from=tmp_path / "c.zip")
    assert s.rom_file == tmp_path / "r.gb"
    assert s.resume_path == tmp_path / "c.zip"


def test_budget_from_cost(monkeypatch):
    monkeypatch.setenv("GBRL_COST_PER_HR_USD", "0.5")
    assert Settings().hours_budget == 6.0  # 8 EUR buys 16.8 h > 6 h cap
    monkeypatch.setenv("GBRL_COST_PER_HR_USD", "3.0")
    assert Settings().hours_budget == pytest.approx(8 * 1.05 / 3.0)


def test_budget_without_cost_uses_max_hours():
    assert Settings(max_hours=2.0).hours_budget == 2.0


def test_run_dir(tmp_path):
    s = Settings(runs_dir=tmp_path, run_id="abc")
    assert s.run_dir == tmp_path / "abc"
