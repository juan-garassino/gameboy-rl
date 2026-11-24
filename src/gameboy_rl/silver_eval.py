"""Silver's evidence: baselines before training, then a color video + game-outcome stats every eval_freq steps."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from gameboy_rl.config import Settings
from gameboy_rl.monitoring import mlflow_tracking
from gameboy_rl.silver import SilverEnv, play_episode, run_baselines

logger = logging.getLogger(__name__)
VIDEO_FPS = 8  # one agent step = 0.4 s of game, so ~3x real speed


class SilverEvalCallback(BaseCallback):
    def __init__(self, cfg: Settings, run_dir: Path, heartbeat: Callable[[], None] | None = None,
                 env_factory: Callable[[Settings], object] = SilverEnv):
        super().__init__()
        self.cfg, self.run_dir, self.heartbeat, self.env_factory = cfg, Path(run_dir), heartbeat, env_factory
        self.next_eval = 0

    def _on_training_start(self) -> None:
        path = self.run_dir / "baselines.json"
        if not path.exists():
            env = self.env_factory(self.cfg)
            try:
                out = run_baselines(env, self.cfg.eval_steps, self.heartbeat)
            finally:
                env.close()
            path.write_text(json.dumps(out, indent=2))
            mlflow_tracking.log_metrics(out, step=0)
            mlflow_tracking.log_artifact(path)
            logger.info("baselines: %s", out)
        self.next_eval = self.model.num_timesteps + self.cfg.eval_freq

    def _choose(self, obs: dict) -> int:
        batch = {k: np.asarray(v)[None] for k, v in obs.items()}
        return int(self.model.predict(batch, deterministic=False)[0][0])

    def evaluate(self) -> dict[str, float]:
        steps, frames = self.model.num_timesteps, []
        env = self.env_factory(self.cfg)
        try:
            stats = play_episode(env, self.cfg.eval_steps, self._choose, self.heartbeat, frames)
        finally:
            env.close()
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(video), frames, fps=VIDEO_FPS, macro_block_size=16, format="FFMPEG")
        mlflow_tracking.log_artifact(video, "videos")
        mlflow_tracking.log_metrics({f"eval/{k}": v for k, v in stats.items()}, step=steps)
        logger.info("eval @%d: %s", steps, stats)
        return stats

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_eval:
            self.next_eval = self.model.num_timesteps + self.cfg.eval_freq
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        self.evaluate()
