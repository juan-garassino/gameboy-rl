"""Judge a Silver checkpoint on many full-length games at once, on idle CPU cores: python -m gameboy_rl.check MODEL [--games 16]

Each game starts at the normal start and is played as in training (sampled), in its own process.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from gameboy_rl.config import Settings
from gameboy_rl.silver import PATH


def play(args: tuple[str, int, int]) -> dict:
    import torch
    from stable_baselines3 import PPO

    from gameboy_rl.silver import MAP_GROUP, MAP_NUMBER, SilverEnv

    path, seed, steps = args
    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu", custom_objects={"learning_rate": 3e-4, "lr_schedule": 3e-4, "clip_range": 0.2})
    torch.manual_seed(seed)  # loading a model resets torch's seed: without this every game would sample the same way
    env = SilverEnv(Settings(game="silver", run_id="check", frontier_prob=0.0, ep_length=steps))
    obs, _ = env.reset(seed=seed)
    reached, at_home, t = {0: 0}, 0, 0
    for t in range(steps):
        action = int(model.predict({k: np.asarray(v)[None] for k, v in obs.items()}, deterministic=False)[0][0])
        obs, _, stalled, over, _ = env.step(action)
        reached.setdefault(env.stats.path, t)
        at_home += (env._read(MAP_GROUP), env._read(MAP_NUMBER)) in ((24, 6), (24, 7))
        if stalled or over:
            break
    out = {"path": env.stats.path, "steps": t + 1, "level": env.stats.best["level"], "badges": env.stats.best["badge"], "house_steps": at_home, "reached": reached}
    env.close()
    return out


def summary(games: list[dict]) -> dict:
    n = len(games)
    steps = {name: sum(g["path"] > k for g in games) for k, (_, _, name) in enumerate(PATH)}
    return {"games": n, "got to": steps, "badges": sum(g["badges"] > 0 for g in games), "mean level": float(np.mean([g["level"] for g in games])),
            "mean steps": float(np.mean([g["steps"] for g in games])), "share of steps in the player's house": float(sum(g["house_steps"] for g in games) / sum(g["steps"] for g in games))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    with mp.get_context("spawn").Pool(a.procs) as pool:
        games = pool.map(play, [(str(a.model), 1000 + i, a.steps) for i in range(a.games)], chunksize=1)
    print(json.dumps({"model": a.model.name, **summary(games)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
