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
from gameboy_rl.silver import SilverEnv, play_episodes, run_baselines

logger = logging.getLogger(__name__)
VIDEO_FPS = 8  # one agent step = 0.4 s of game, so ~3x real speed
VIDEO_SCALE = 4  # the screen is 160x144: players open that tiny


class SilverEvalCallback(BaseCallback):
    def __init__(self, cfg: Settings, run_dir: Path, heartbeat: Callable[[], None] | None = None,
                 env_factory: Callable[[Settings], object] = SilverEnv):
        super().__init__()
        # judged from the normal start only, whatever the training games do
        self.cfg = cfg.model_copy(update={"frontier_prob": 0.0})
        self.run_dir, self.heartbeat, self.env_factory = Path(run_dir), heartbeat, env_factory
        self.next_eval = 0

    def _on_training_start(self) -> None:
        path = self.run_dir / "baselines.json"
        if not path.exists():
            env = self.env_factory(self.cfg)
            try:
                out = run_baselines(env, self.cfg.eval_steps, self.heartbeat, self.cfg.eval_episodes)
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
            stats = play_episodes(env, self.cfg.eval_steps, self.cfg.eval_episodes, self._choose, self.heartbeat, frames)
            areas = env.stats.areas()  # of the last game, in the order reached
        finally:
            env.close()
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(video), fps=VIDEO_FPS, macro_block_size=16, format="FFMPEG") as writer:
            for frame in frames:
                writer.append_data(frame.repeat(VIDEO_SCALE, axis=0).repeat(VIDEO_SCALE, axis=1))
        mlflow_tracking.log_artifact(video, "videos")
        mlflow_tracking.log_metrics({f"eval/{k}": v for k, v in stats.items()}, step=steps)
        logger.info("eval @%d: %s areas=%s", steps, stats, areas)
        mlflow_tracking.set_tag("last_eval_areas", " > ".join(areas))
        return stats

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_eval:
            self.next_eval = self.model.num_timesteps + self.cfg.eval_freq
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        self.evaluate()
