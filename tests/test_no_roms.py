"""Guard: ROMs must never be committed (the repo is public). They live only in gitignored roms/ and, per run, in GCS.

Upstream PWhiddy files stay allowed: the .state save states (RAM snapshots, no ROM data) and the
pretrained policies (v2/runs/poke_26214400.zip and the v1 one under _upstream/).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROM_PATTERN = re.compile(r"(^|/)roms/|\.(gb|gbc|gba|sgb|sav|ram|zip|7z|rar)$", re.IGNORECASE)
ALLOWED = {"v2/runs/poke_26214400.zip", "_upstream/baselines/session_4da05e87_main_good/poke_439746560_steps.zip"}


def test_no_roms_tracked():
    repo = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True).stdout.split("\n")
    offenders = [f for f in files if f and ROM_PATTERN.search(f) and f not in ALLOWED]
    assert offenders == [], f"ROM-like files tracked by git: {offenders}"


def test_state_files_are_allowed_and_present():
    repo = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files", "*.state"], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.split()
    assert "has_pokedex_nballs.state" in files


def test_dockerignore_keeps_roms_out_of_the_image():
    ignore = (Path(__file__).resolve().parents[1] / ".dockerignore").read_text().split()
    for pattern in ("roms/", "*.gb", "*.gbc", "v2/runs/"):
        assert pattern in ignore
