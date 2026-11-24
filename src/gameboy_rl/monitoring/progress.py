"""Pokemon progress (badges, event flags, explore-map coverage) logged to the SB3 logger once per rollout.

Read-only: uses RedGymEnv's existing methods/attributes through env_method/get_attr, never touches rewards.
"""
from __future__ import annotations

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.last: dict[str, float] = {}

    def _on_step(self) -> bool:
        return True

    def _collect(self) -> dict[str, float]:
        env = self.training_env
        badges = np.asarray(env.env_method("get_badges"), dtype=np.float64)
        events = np.asarray(env.env_method("get_all_events_reward"), dtype=np.float64)
        tiles = np.asarray([np.count_nonzero(m) for m in env.get_attr("explore_map")], dtype=np.float64)
        map_progress = np.asarray(env.get_attr("max_map_progress"), dtype=np.float64)
        return {
            "badges_max": badges.max(), "badges_mean": badges.mean(),
            "events_max": events.max(), "events_mean": events.mean(),
            "explore_tiles_max": tiles.max(), "explore_tiles_mean": tiles.mean(),
            "map_progress_max": map_progress.max(),
        }

    def _on_rollout_end(self) -> None:
        try:
            stats = self._collect()
        except Exception as e:  # progress is diagnostics only; env-crash handling lives in train.run
            logger.warning("progress stats skipped: %s: %s", type(e).__name__, e)
            return
        for k, v in stats.items():
            self.logger.record(f"pokemon/{k}", float(v))
        self.last = {k: float(v) for k, v in stats.items()}
