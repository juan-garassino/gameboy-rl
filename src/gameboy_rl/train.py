"""PPO on PWhiddy's v2 env with baseline_fast_v2's hyperparameters, plus health / budget / checkpoint plumbing.

Mirrors sega-rl's train.run: resume, CheckpointKeeper, HealthCallback, BudgetCallback, env-crash -> CRITICAL,
DONE marker with done|critical|budget.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import Logger, make_output_format
from stable_baselines3.common.vec_env import VecEnv

from gameboy_rl.config import Settings
from gameboy_rl.env import import_upstream, make_vec_env
from gameboy_rl.monitoring import mlflow_tracking
from gameboy_rl.monitoring.health import HealthCallback, HealthMonitor
from gameboy_rl.monitoring.progress import ProgressCallback

logger = logging.getLogger(__name__)

# baseline_fast_v2.py: PPO("MultiInputPolicy", env, n_steps=ep_length // 64, batch_size=512, n_epochs=1,
#                          gamma=0.997, ent_coef=0.01); learning_rate / clip_range are SB3's defaults.
LEARNING_RATE, CLIP_RANGE = 3e-4, 0.2
PPO_PARAMS = dict(batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, learning_rate=LEARNING_RATE,
                  clip_range=CLIP_RANGE)
# Checkpoints carry cloudpickled schedule lambdas; PWhiddy's (Python 3.11 bytecode) segfault when called on 3.12.
# Floats here make SB3 rebuild them (_setup_model -> get_schedule_fn).
SCHEDULE_OVERRIDES = {"learning_rate": LEARNING_RATE, "lr_schedule": LEARNING_RATE, "clip_range": CLIP_RANGE}
ENV_CRASH_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)


def build_model(cfg: Settings, env: VecEnv) -> PPO:
    return PPO("MultiInputPolicy", env, n_steps=cfg.n_steps, seed=cfg.seed, verbose=0, **PPO_PARAMS)


def load_model(path: Path, cfg: Settings, env: VecEnv) -> PPO:
    """Load a checkpoint and re-size its rollout buffer to this run's n_steps/n_envs (as baseline_fast_v2 does)."""
    model = PPO.load(path, env=env, custom_objects=SCHEDULE_OVERRIDES)
    model.n_steps = cfg.n_steps
    model.n_envs = cfg.n_envs
    model.rollout_buffer.buffer_size = cfg.n_steps
    model.rollout_buffer.n_envs = cfg.n_envs
    model.rollout_buffer.reset()
    return model


class CheckpointKeeper(BaseCallback):
    """Saves ckpt_<steps>.zip every cfg.ckpt_freq steps, keeping the newest cfg.keep_ckpts."""

    def __init__(self, cfg: Settings, ckpt_dir: Path):
        super().__init__()
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        self.next_save = cfg.ckpt_freq

    def _on_training_start(self) -> None:
        self.next_save = (self.model.num_timesteps // self.cfg.ckpt_freq + 1) * self.cfg.ckpt_freq

    def save(self) -> Path:
        path = self.ckpt_dir / f"ckpt_{self.model.num_timesteps:010d}.zip"
        self.model.save(path)
        for old in sorted(self.ckpt_dir.glob("ckpt_*.zip"))[: -self.cfg.keep_ckpts]:
            old.unlink()
        return path

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_save:
            self.save()
            self.next_save += self.cfg.ckpt_freq
        return True


class BudgetCallback(BaseCallback):
    """Stops training at the first rollout boundary after the wall-clock deadline."""

    def __init__(self, deadline_ts: float):
        super().__init__()
        self.deadline_ts = deadline_ts
        self.hit = False

    def _on_rollout_end(self) -> None:
        if time.time() >= self.deadline_ts:
            self.hit = True
            logger.warning("time budget reached, stopping")

    def _on_step(self) -> bool:
        return not self.hit


class HealthReporter:
    """Mirrors health into MLflow: tags on every status change, health.json at most every upload_every_s.

    MLflow (--serve-artifacts) is the durable channel: the pod's GCS token expires after 1 h (sega-rl Task 10).
    """

    def __init__(self, health_json: Path, upload_every_s: float = 600.0, clock: Callable[[], float] = time.time):
        self.health_json = health_json
        self.upload_every_s = upload_every_s
        self.clock = clock
        self.last_key: tuple | None = None
        self.last_upload: float | None = None

    def __call__(self, report) -> None:
        key = (report.status, tuple(report.reasons))
        if key != self.last_key:
            mlflow_tracking.set_tag("health_status", report.status)
            mlflow_tracking.set_tag("health_reasons", ",".join(report.reasons))
            self.last_key = key
        now = self.clock()
        if self.last_upload is None or now - self.last_upload >= self.upload_every_s:
            mlflow_tracking.log_artifact(self.health_json)
            self.last_upload = now


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.zip"))
    return ckpts[-1] if ckpts else None


def _make_logger(run_dir: Path) -> Logger:
    tb = str(run_dir / "tb")
    formats = [make_output_format("stdout", tb), make_output_format("tensorboard", tb)]
    if mlflow_tracking.enabled():
        formats.append(mlflow_tracking.MLflowOutputFormat())
    return Logger(tb, formats)


def _resume_source(cfg: Settings, ckpt_dir: Path) -> Path | None:
    """The run's own newest checkpoint wins; otherwise cfg.resume_from (e.g. PWhiddy's poke_26214400.zip)."""
    own = _latest_checkpoint(ckpt_dir)
    if own:
        return own
    if cfg.resume_path is None:
        return None
    if not cfg.resume_path.exists():
        raise FileNotFoundError(f"GBRL_RESUME_FROM not found: {cfg.resume_path}")
    return cfg.resume_path


def run(cfg: Settings, env_factory: Callable[[Settings], VecEnv] = make_vec_env) -> str:
    """Train (or resume) until cfg.total_steps, CRITICAL health, or the time budget. Returns done|critical|budget."""
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_from = _resume_source(cfg, ckpt_dir)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    deadline = time.time() + cfg.hours_budget * 3600

    env = env_factory(cfg)
    if resume_from:
        logger.info("resuming from %s", resume_from)
        model = load_model(resume_from, cfg, env)
    else:
        model = build_model(cfg, env)
    model.set_logger(_make_logger(run_dir))
    mlflow_tracking.start(cfg)

    health = HealthCallback(
        HealthMonitor(cfg),
        run_dir / "health.json",
        on_critical=lambda r: logger.error("CRITICAL training health: %s", r.reasons),
        stall_secs=cfg.fps_zero_secs,
        on_report=HealthReporter(run_dir / "health.json"),
    )
    keeper = CheckpointKeeper(cfg, ckpt_dir)
    budget = BudgetCallback(deadline)
    progress = ProgressCallback()
    TensorboardCallback = import_upstream("tensorboard_callback").TensorboardCallback
    callbacks = CallbackList([keeper, progress, health, TensorboardCallback(str(run_dir / "tb")), budget])

    status = "done"
    remaining = cfg.total_steps - model.num_timesteps
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=resume_from is None,
                        log_interval=1)
        if health.last_report and health.last_report.status == "CRITICAL":
            status = "critical"
        elif budget.hit:
            status = "budget"
    except ENV_CRASH_ERRORS as e:
        logger.exception("environment worker crashed: %s", e)
        health.write(health.monitor.mark_critical("env_crash"))
        status = "critical"
    finally:
        keeper.model = model
        final_ckpt = keeper.save()
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")

    if progress.last:
        (run_dir / "progress.json").write_text(json.dumps(progress.last, indent=2))
    (run_dir / "DONE").write_text(status)
    for artifact in ("health.json", "progress.json", "DONE"):
        if (run_dir / artifact).exists():
            mlflow_tracking.log_artifact(run_dir / artifact)
    mlflow_tracking.log_artifact(final_ckpt, "checkpoints")
    mlflow_tracking.end(status)
    logger.info("run %s finished: %s at %d steps", cfg.run_id, status, model.num_timesteps)
    return status
