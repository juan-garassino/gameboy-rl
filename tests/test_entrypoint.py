"""Entrypoint behaviour with stubbed gsutil / curl / gameboy-rl on PATH (no cloud, no pod)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAKE_ROM = b"fake rom"
FAKE_ROM_SHA1 = hashlib.sha1(FAKE_ROM).hexdigest()

# cp/rsync from gs:// write a file at the destination (last arg) so the entrypoint sees real files.
GSUTIL = r'''#!/usr/bin/env bash
echo "gsutil $*" >> "$CALLS"
dest="${@: -1}"
case "$*" in
  *stat*DONE*) [ "${STUB_DONE:-0}" = 1 ] && exit 0 || exit 1 ;;
  *" rm "*) exit 0 ;;
  *cp*roms/*)
    [ "${STUB_ROM_MISSING:-0}" = 1 ] && exit 1
    printf 'fake rom' > "$dest"; exit 0 ;;
  *cp*init/*)
    [ "${STUB_INIT_MISSING:-0}" = 1 ] && exit 1
    printf 'ckpt' > "$dest"; exit 0 ;;
  *rsync*gs://*/runs/*" "/*)
    [ "${STUB_RSYNC_FAIL:-0}" = 1 ] && exit 1
    if [ "${STUB_HAS_CKPT:-0}" = 1 ]; then mkdir -p "$dest/checkpoints"; printf 'c' > "$dest/checkpoints/ckpt_0000000100.zip"; fi
    exit 0 ;;
  *rsync*) [ "${STUB_RSYNC_FAIL:-0}" = 1 ] && exit 1 || exit 0 ;;
esac
exit 0
'''
CURL = r'''#!/usr/bin/env bash
echo "curl $*" >> "$CALLS"
case "$*" in
  *costPerHr*) echo '{"data":{"pod":{"costPerHr":0.5}}}' ;;
  *podTerminate*)
    n=$(( $(cat "$CALLS.term" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$CALLS.term"
    if [ "$n" -le "${STUB_TERMINATE_FAILS:-0}" ]; then echo '{"errors":[{"message":"boom"}]}'
    else echo '{"data":{"podTerminate":null}}'; fi ;;
esac
'''
TRAINER = r'''#!/usr/bin/env bash
echo "mlflow retries=${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-} timeout=${MLFLOW_HTTP_REQUEST_TIMEOUT:-}" >> "$CALLS"
echo "gameboy-rl $* cost=${GBRL_COST_PER_HR_USD:-} rom=${GBRL_ROM_PATH:-} resume=${GBRL_RESUME_FROM:-}" >> "$CALLS"
exit "${STUB_TRAIN_RC:-0}"
'''


def _trainer_line(calls: str) -> str | None:
    return next((line for line in calls.splitlines() if line.startswith("gameboy-rl ")), None)


def _run(tmp_path: Path, **env_overrides) -> tuple[int, str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in {"gsutil": GSUTIL, "curl": CURL, "gameboy-rl": TRAINER}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.log"
    calls.touch()
    ws = tmp_path / "ws"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls), "WORKSPACE_DIR": str(ws),
        "APP_DIR": str(REPO), "RUN_ID": "t", "GCS_BUCKET": "bucket", "MODE": "train",
        "RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "k", "SYNC_INTERVAL": "3600", "HOME": str(tmp_path),
        "ROM_SHA1": FAKE_ROM_SHA1, "POD_SELF_BACKOFF": "0",
    }
    env.update(env_overrides)
    proc = subprocess.run(["bash", str(REPO / "runpod" / "entrypoint.sh")], env=env, capture_output=True,
                          text=True, timeout=60)
    return proc.returncode, calls.read_text(), ws


def _health(ws: Path) -> dict:
    return json.loads((ws / "runs" / "t" / "health.json").read_text())


def test_done_marker_exits(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_DONE="1")
    assert rc == 0
    assert "podTerminate" in calls
    assert _trainer_line(calls) is None
    assert "roms/" not in calls  # a finished run never consumes the uploaded ROM


def test_rom_pulled_then_deleted_from_gcs_before_training(tmp_path):
    rc, calls, ws = _run(tmp_path)
    assert rc == 0
    lines = calls.splitlines()
    cp = next(i for i, ln in enumerate(lines) if " cp " in ln and "gs://bucket/gameboy-rl/roms/pokemon_red_version.gb" in ln)
    rm = next(i for i, ln in enumerate(lines) if " rm " in ln and "gs://bucket/gameboy-rl/roms/" in ln)
    train = next(i for i, ln in enumerate(lines) if ln.startswith("gameboy-rl "))
    assert cp < rm < train
    rom = ws / "roms" / "pokemon_red_version.gb"
    assert rom.read_bytes() == FAKE_ROM
    assert f"rom={rom}" in lines[train]


def test_missing_rom_is_critical_and_removes_pod(tmp_path):
    rc, calls, ws = _run(tmp_path, STUB_ROM_MISSING="1")
    assert rc != 0
    assert "podTerminate" in calls
    assert _trainer_line(calls) is None
    assert _health(ws)["status"] == "CRITICAL" and _health(ws)["reasons"] == ["rom"]


def test_rom_already_on_disk_skips_pull(tmp_path):
    rom = tmp_path / "ws" / "roms" / "pokemon_red_version.gb"
    rom.parent.mkdir(parents=True)
    rom.write_bytes(FAKE_ROM)
    rc, calls, _ = _run(tmp_path, STUB_ROM_MISSING="1")
    assert rc == 0
    assert "roms/" not in "\n".join(ln for ln in calls.splitlines() if ln.startswith("gsutil"))
    assert _trainer_line(calls) is not None


def test_rom_checksum_mismatch_is_critical(tmp_path):
    rc, calls, ws = _run(tmp_path, ROM_SHA1="0" * 40)
    assert rc != 0
    assert _trainer_line(calls) is None
    assert _health(ws)["reasons"] == ["rom"]
    assert not (ws / "roms" / "pokemon_red_version.gb").exists()


def test_init_checkpoint_pulled_for_fresh_run(tmp_path):
    rc, calls, ws = _run(tmp_path)
    assert rc == 0
    assert "gs://bucket/gameboy-rl/init/poke_26214400.zip" in calls
    assert f"resume={ws / 'init' / 'poke_26214400.zip'}" in _trainer_line(calls)


def test_existing_run_checkpoint_skips_init_pull(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_HAS_CKPT="1")
    assert rc == 0
    assert "init/poke_26214400.zip" not in calls
    assert _trainer_line(calls).endswith("resume=")


def test_missing_init_checkpoint_is_critical_and_keeps_rom_in_gcs(tmp_path):
    rc, calls, ws = _run(tmp_path, STUB_INIT_MISSING="1")
    assert rc != 0
    assert _trainer_line(calls) is None
    assert _health(ws)["reasons"] == ["init_ckpt"]
    assert "roms/" not in calls  # checked before the ROM is pulled (and deleted)
    assert "podTerminate" in calls


def test_empty_init_ckpt_starts_fresh(tmp_path):
    rc, calls, _ = _run(tmp_path, INIT_CKPT="")
    assert rc == 0
    assert "init/" not in calls
    assert _trainer_line(calls).endswith("resume=")


def test_sync_failure_does_not_stop_training(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_RSYNC_FAIL="1")
    assert rc == 0
    assert "gameboy-rl train" in calls
    assert "podTerminate" in calls


def test_cost_exported_to_trainer(tmp_path):
    _, calls, _ = _run(tmp_path)
    assert "gameboy-rl train cost=0.5" in calls


def test_trainer_failure_still_syncs_and_terminates(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_TRAIN_RC="3")
    assert rc == 3
    lines = calls.splitlines()
    train_idx = next(i for i, line in enumerate(lines) if line.startswith("gameboy-rl "))
    after = "\n".join(lines[train_idx:])
    assert "rsync" in after and "podTerminate" in after


def test_local_run_without_pod_id_skips_terminate(tmp_path):
    rc, calls, _ = _run(tmp_path, RUNPOD_POD_ID="")
    assert rc == 0
    assert "podTerminate" not in calls


def test_missing_required_env_still_removes_pod(tmp_path):
    rc, calls, _ = _run(tmp_path, RUN_ID="")
    assert rc != 0
    assert "podTerminate" in calls
    assert _trainer_line(calls) is None


def test_mlflow_http_limits_exported(tmp_path):
    _, calls, _ = _run(tmp_path)
    assert "mlflow retries=2 timeout=20" in calls


def test_terminate_retries_on_graphql_errors(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_TERMINATE_FAILS="2")
    assert rc == 0
    assert calls.count("podTerminate") == 3


def test_gsutil_wrapped_in_timeout():
    script = (REPO / "runpod" / "entrypoint.sh").read_text()
    assert 'timeout "$GCS_TIMEOUT" gsutil' in script
