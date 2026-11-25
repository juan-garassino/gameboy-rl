from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]  # editable install: /app in the pod, the clone locally
V2_DIR = REPO_ROOT / "v2"  # PWhiddy's env lives here, imported as-is


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def repo_path(p: Path) -> Path:
    """Relative paths are relative to the repo root, so the CLI works from any cwd."""
    return p if p.is_absolute() else REPO_ROOT / p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GBRL_", extra="ignore")

    game: Literal["red", "silver"] = "red"  # GBRL_GAME=silver (color, player CLAUDE; see silver.py)
    init_state: Path = Path("has_pokedex_nballs.state")  # GBRL_INIT_STATE=init.state (past Oak's intro + parcel)
    rom_path: Path = Path("roms/pokemon_red_version.gb")  # GBRL_ROM_PATH=/workspace/roms/pokemon_red_version.gb
    run_id: str = Field(default_factory=_utc_run_id)  # GBRL_RUN_ID=20261124-150000
    total_steps: int = 150_000_000  # GBRL_TOTAL_STEPS=150000000 (absolute; includes a resumed checkpoint's steps)
    n_envs: int = 32  # GBRL_N_ENVS=32 (~ pod vCPUs)
    ep_length: int = 2048 * 80  # GBRL_EP_LENGTH=163840 (agent steps per episode; PPO n_steps = ep_length // 64)
    seed: int = 0  # GBRL_SEED=0
    ckpt_freq: int = 5_000_000  # GBRL_CKPT_FREQ=5000000 (each checkpoint, ~15 MB, is also uploaded to MLflow)
    keep_ckpts: int = 3  # GBRL_KEEP_CKPTS=3
    runs_dir: Path = Path("runs")  # GBRL_RUNS_DIR=/workspace/runs
    resume_from: Path | None = None  # GBRL_RESUME_FROM=/workspace/init/poke_26214400.zip (when no run ckpt yet)
    eval_episodes: int = 3  # GBRL_EVAL_EPISODES=3 (silver: games per eval and per baseline; one game proves nothing)
    stall_steps: int = 1024  # GBRL_STALL_STEPS=1024 (silver: dry steps that end a game, × (1 + path steps reached))
    frontier_prob: float = 0.5  # GBRL_FRONTIER_PROB=0.5 (silver: share of games starting at the furthest point reached)
    ent_coef: float = 0.01  # GBRL_ENT_COEF=0.02 (silver: a stuck policy needs more exploration than Red's 0.01)
    eval_freq: int = 500_000  # GBRL_EVAL_FREQ=500000 (silver: color video + outcome stats)
    eval_steps: int = 4096  # GBRL_EVAL_STEPS=4096 (agent steps per eval/baseline game at most, ~27 min; a stalled game ends)
    stream: bool = False  # GBRL_STREAM=true (broadcasts coords to PWhiddy's PUBLIC map; off for privacy)
    stream_user: str = "gameboy-rl"  # GBRL_STREAM_USER=gameboy-rl (name shown on the public map)
    stop_on_critical: bool = True  # GBRL_STOP_ON_CRITICAL=false
    max_hours: float = 6.0  # GBRL_MAX_HOURS=6
    max_eur: float = 8.0  # GBRL_MAX_EUR=8
    usd_per_eur: float = 1.05  # GBRL_USD_PER_EUR=1.05
    cost_per_hr_usd: float | None = None  # GBRL_COST_PER_HR_USD=0.44 (set by the RunPod entrypoint)

    kl_warn: float = 0.05  # GBRL_KL_WARN=0.05
    kl_crit: float = 0.2  # GBRL_KL_CRIT=0.2
    clipfrac_warn: float = 0.3  # GBRL_CLIPFRAC_WARN=0.3
    entropy_collapse_frac: float = 0.2  # GBRL_ENTROPY_COLLAPSE_FRAC=0.2
    ev_warn_after: float = 0.1  # GBRL_EV_WARN_AFTER=0.1 (fraction of training)
    ev_crit: float = -0.5  # GBRL_EV_CRIT=-0.5
    plateau_frac: float = 0.2  # GBRL_PLATEAU_FRAC=0.2
    fps_warn_frac: float = 0.5  # GBRL_FPS_WARN_FRAC=0.5
    fps_zero_secs: float = 300.0  # GBRL_FPS_ZERO_SECS=300 (also the hang watchdog)
    button_spam_frac: float = 0.9  # GBRL_BUTTON_SPAM_FRAC=0.9

    @field_validator("n_envs")
    @classmethod
    def _even_n_envs(cls, v: int) -> int:
        # v2/tensorboard_callback.py rearranges explore maps into 2 rows at every episode end.
        if v < 2 or v % 2:
            raise ValueError(f"n_envs must be even and >= 2, got {v}")
        return v

    @property
    def steps(self) -> int:
        return self.total_steps

    @property
    def n_steps(self) -> int:
        return self.ep_length // 64

    @property
    def hours_budget(self) -> float:
        """Wall-clock cap: max_hours, tightened by the EUR cap when the pod's hourly cost is known."""
        if not self.cost_per_hr_usd:
            return self.max_hours
        return min(self.max_hours, self.max_eur * self.usd_per_eur / self.cost_per_hr_usd)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def init_state_path(self) -> Path:
        return repo_path(self.init_state)

    @property
    def rom_file(self) -> Path:
        return repo_path(self.rom_path)

    @property
    def resume_path(self) -> Path | None:
        return repo_path(self.resume_from) if self.resume_from else None


settings = Settings()
