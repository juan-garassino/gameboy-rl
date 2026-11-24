from __future__ import annotations

from pathlib import Path

import pytest

from gameboy_rl.cli import build_parser, settings_from_args


def test_subcommands():
    p = build_parser()
    for cmd in ("train", "smoke", "check-start"):  # status needs a run name
        assert p.parse_args([cmd]).cmd == cmd
    with pytest.raises(SystemExit):
        p.parse_args([])


def test_overrides():
    args = build_parser().parse_args(["train", "--n-envs", "4", "--steps", "1000", "--state", "init.state",
                                      "--resume-from", "v2/runs/poke_26214400.zip"])
    s = settings_from_args(args)
    assert s.n_envs == 4 and s.total_steps == 1000
    assert s.init_state == Path("init.state")
    assert s.resume_from == Path("v2/runs/poke_26214400.zip")


def test_defaults_come_from_settings():
    s = settings_from_args(build_parser().parse_args(["check-start"]))
    assert s.init_state == Path("has_pokedex_nballs.state")
