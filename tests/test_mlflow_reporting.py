from __future__ import annotations

import json
from pathlib import Path

import pytest

from gameboy_rl.config import Settings
from gameboy_rl.monitoring import mlflow_tracking
from gameboy_rl.monitoring.health import HealthReport, report_stall
from gameboy_rl.status import format_status
from gameboy_rl.train import CheckpointKeeper, HealthReporter, TrackingUnavailable, run


class _Rec:
    def __init__(self):
        self.tags, self.artifacts = {}, []

    def set_tag(self, k, v):
        self.tags[k] = v

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((str(path), artifact_path))


def test_health_reporter_tags_on_change_and_throttles_uploads(tmp_path, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(mlflow_tracking, "set_tag", rec.set_tag)
    monkeypatch.setattr(mlflow_tracking, "log_artifact", rec.log_artifact)
    health_json = tmp_path / "health.json"
    health_json.write_text("{}")
    clock = iter([0.0, 10.0, 700.0])
    reporter = HealthReporter(health_json, upload_every_s=600, clock=lambda: next(clock))

    reporter(HealthReport("HEALTHY", []))
    reporter(HealthReport("WARN", ["approx_kl"]))
    reporter(HealthReport("WARN", ["approx_kl"]))

    assert rec.tags["health_status"] == "WARN"
    assert rec.tags["health_reasons"] == "approx_kl"
    assert len(rec.artifacts) == 2  # t=0 and t=700, not t=10


def test_format_status():
    text = format_status("pokered-x", {"health_status": 1.0, "pokemon_badges_max": 2.0, "pokemon_events_max": 37.0,
                                        "rollout_ep_rew_mean": 1.5, "time_fps": 900.0},
                         {"health_status": "WARN", "health_reasons": "approx_kl", "final_status": None}, step=500_000)
    assert "WARN" in text and "approx_kl" in text and "500,000" in text
    assert "badges" in text and "37" in text


def test_experiment_name():
    assert mlflow_tracking.EXPERIMENT == "gameboy-rl"


def test_mlflow_new_helpers_noop_when_disabled():
    assert mlflow_tracking.enabled() is False
    mlflow_tracking.set_tag("a", "b")
    mlflow_tracking.log_artifact(__file__, "x")


class _FakeMlflow:
    def __init__(self, fail_start, existing=None):
        self.fail_start = fail_start
        self.existing = existing
        self.created = None
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append("uri")

    def get_experiment_by_name(self, name):
        if self.fail_start:
            raise RuntimeError("401 not authenticated")
        return self.existing

    def create_experiment(self, name, artifact_location=None):
        self.created = (name, artifact_location)

    def set_experiment(self, name):
        if self.fail_start:
            raise RuntimeError("401 not authenticated")

    def start_run(self, run_name):
        self.calls.append("start")

    def log_metrics(self, metrics, step=0):
        self.calls.append("metrics")

    def set_tag(self, k, v):
        self.calls.append("tag")

    def log_params(self, p):
        pass

    def end_run(self):
        self.calls.append("end")


def test_failed_start_disables_all_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.set_tag("k", "v")
    assert "metrics" not in fake.calls and "tag" not in fake.calls  # no stray auto-created runs


def test_successful_start_enables_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=False)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.end("done")
    assert fake.calls.count("metrics") == 1 and "end" in fake.calls
    assert mlflow_tracking._active is False


def test_start_creates_experiment_with_served_artifacts(monkeypatch):
    fake = _FakeMlflow(fail_start=False)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    assert mlflow_tracking.start(Settings()) is True
    assert fake.created == ("gameboy-rl", "mlflow-artifacts:/gameboy-rl")
    mlflow_tracking.end("done")


def test_failed_start_returns_false(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "mlflow", _FakeMlflow(fail_start=True))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    assert mlflow_tracking.start(Settings()) is False


class _ProbeFailMlflow(_FakeMlflow):
    def log_artifact(self, path, artifact_path=None):
        raise RuntimeError("upload failed")


def test_failed_probe_upload_returns_false(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "mlflow", _ProbeFailMlflow(fail_start=False))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    probe = tmp_path / "config.json"
    probe.write_text("{}")
    assert mlflow_tracking.start(Settings(), probe=probe) is False
    assert mlflow_tracking._active is False


class _StubPPO:
    num_timesteps = 1_000

    def save(self, path):
        Path(path).write_bytes(b"zip")


def test_checkpoint_keeper_uploads_each_checkpoint(tmp_path, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(mlflow_tracking, "log_artifact", rec.log_artifact)
    beats = []
    keeper = CheckpointKeeper(Settings(keep_ckpts=1), tmp_path, heartbeat=lambda: beats.append(1))
    keeper.model = _StubPPO()
    path = keeper.save()
    assert path.exists()
    assert rec.artifacts == [(str(path), "checkpoints")]
    assert len(beats) == 2  # before and after the (possibly slow) upload


def test_stall_is_reported_to_mlflow(monkeypatch):
    rec = _Rec()
    ended = []
    monkeypatch.setattr(mlflow_tracking, "set_tag", rec.set_tag)
    monkeypatch.setattr(mlflow_tracking, "end", ended.append)
    report_stall(HealthReport("CRITICAL", ["stalled"]))
    assert rec.tags["health_status"] == "CRITICAL" and rec.tags["health_reasons"] == "stalled"
    assert ended == ["stalled"]


class _NeverCalled:
    def __call__(self, cfg):
        raise AssertionError("env must not be created when the tracker is unreachable")


def test_fail_fast_when_tracker_unavailable(tmp_path, monkeypatch):
    probes = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "start", lambda cfg, probe=None: probes.append(probe) or False)
    cfg = Settings(runs_dir=tmp_path, run_id="t", n_envs=2)
    with pytest.raises(TrackingUnavailable):
        run(cfg, env_factory=_NeverCalled())
    assert not (tmp_path / "t" / "DONE").exists()  # a relaunch with the same RUN_ID must still train
    health = json.loads((tmp_path / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and health["reasons"] == ["mlflow"]
    assert probes == [tmp_path / "t" / "config.json"]  # config.json is the upload probe


def test_any_crash_is_critical_and_still_writes_done(tmp_path):
    def boom(cfg):
        raise RuntimeError("pyboy exploded")

    assert run(Settings(runs_dir=tmp_path, run_id="t", n_envs=2), env_factory=boom) == "critical"
    health = json.loads((tmp_path / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and "crash" in health["reasons"]
    assert (tmp_path / "t" / "DONE").read_text() == "critical"
