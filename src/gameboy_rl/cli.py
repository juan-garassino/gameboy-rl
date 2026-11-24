from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

from gameboy_rl.config import Settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gameboy-rl", description="PPO on Pokemon Red (PWhiddy's v2 env, wrapped).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_overrides(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--state", type=Path, help="start .state (default: GBRL_INIT_STATE)")
        sp.add_argument("--rom", type=Path, help="ROM path (default: GBRL_ROM_PATH)")
        sp.add_argument("--steps", type=int, help="total timesteps, absolute (default: GBRL_TOTAL_STEPS)")
        sp.add_argument("--n-envs", type=int)
        sp.add_argument("--run-id")
        sp.add_argument("--resume-from", type=Path, help="checkpoint used when the run has none yet")
        return sp

    with_overrides(sub.add_parser("train", help="train or resume a run"))
    with_overrides(sub.add_parser("smoke", help="2-env run, ~2k steps, one full episode, in a temp dir"))
    cs = with_overrides(sub.add_parser("check-start", help="boot the start state, random actions, look for dialog"))
    cs.add_argument("--actions", type=int, default=200)
    pl = with_overrides(sub.add_parser("play", help="replay a checkpoint locally into an mp4"))
    pl.add_argument("--model", type=Path, required=True)
    pl.add_argument("--play-steps", type=int, default=1500, help="agent steps (1 step = 0.4 s of game)")
    pl.add_argument("--video", type=Path, required=True)
    pl.add_argument("--deterministic", action="store_true")
    st = sub.add_parser("status", help="live run state from MLflow (health, badges, fps)")
    st.add_argument("run_name", help="run id = MLflow run name, e.g. 20261124-180000")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {"init_state": args.state, "rom_path": args.rom, "total_steps": args.steps,
                 "n_envs": args.n_envs, "run_id": args.run_id, "resume_from": args.resume_from}
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _cmd_train(args: argparse.Namespace) -> int:
    from gameboy_rl import train

    try:
        status = train.run(settings_from_args(args))
    except train.TrackingUnavailable as e:
        logger.error("%s", e)
        return 4
    print(status)
    return 0 if status != "critical" else 2


def _cmd_smoke(args: argparse.Namespace) -> int:
    from gameboy_rl import train

    base = settings_from_args(args)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = base.model_copy(update=dict(n_envs=2, ep_length=1024, total_steps=base.total_steps
                                          if args.steps else 2048, ckpt_freq=1024, runs_dir=Path(tmp),
                                          run_id="smoke"))
        status = train.run(cfg)
        progress = cfg.run_dir / "progress.json"
        if progress.exists():
            print(progress.read_text())
    print(status)
    return 0 if status == "done" else 2


def _cmd_check_start(args: argparse.Namespace) -> int:
    from gameboy_rl import check_start

    result = check_start.run(settings_from_args(args), n_actions=args.actions)
    print(json.dumps(result, indent=2))
    print(f"start state looks like: {result['verdict'].upper()} (frame saved to {result['png']})")
    return 0 if result["verdict"] == "overworld" else 1


def _cmd_play(args: argparse.Namespace) -> int:
    from stable_baselines3 import PPO

    from gameboy_rl.play import play
    from gameboy_rl.train import SCHEDULE_OVERRIDES

    model = PPO.load(args.model, custom_objects=SCHEDULE_OVERRIDES, device="cpu")
    stats = play(settings_from_args(args), model, args.play_steps, args.video, deterministic=args.deterministic)
    print(json.dumps(stats))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from gameboy_rl.status import fetch

    print(fetch(args.run_name))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"train": _cmd_train, "smoke": _cmd_smoke, "check-start": _cmd_check_start,
                "status": _cmd_status, "play": _cmd_play}
    return handlers[args.cmd](args)
