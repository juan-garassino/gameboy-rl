"""Pokémon Silver (Game Boy Color, in color): env, reward and game-outcome stats.

RAM addresses come from pret/pokegold's pokesilver.sym; the ROM is byte-identical to that build (SHA-1 49b163f7…).
The reward pays only for game progress a player would recognise, so doing nothing must earn exactly nothing.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gameboy_rl.config import Settings

PLAYER_NAME, MAP_GROUP, MAP_NUMBER, Y, X = 0xD1A3, 0xDA00, 0xDA01, 0xDA02, 0xDA03
JOHTO_BADGES, KANTO_BADGES, BATTLE_MODE = 0xD57C, 0xD57D, 0xD116
EVENTS_START, EVENTS_END = 0xD7B7, 0xD8B7
CAUGHT_START, CAUGHT_END = 0xDBE4, 0xDC04
PARTY_COUNT, PARTY, MON_SIZE, EXP, LEVEL, HP, MAX_HP = 0xDA22, 0xDA2A, 0x30, 0x08, 0x1F, 0x22, 0x24

BUTTONS = ["down", "left", "right", "up", "a", "b"]  # no START: menus and saving are not progress
ACTION_FRAMES, PRESS_FRAMES = 24, 8
FRAME_STACK = 3
WEIGHTS = {"badge": 50.0, "event": 2.0, "map": 0.1, "caught": 1.0, "level": 0.5, "exp": 0.01, "tile": 0.005, "road_tile": 0.045}  # road_tile adds to tile

# The story's road to the first badge, from pret/pokegold's map_constants.asm. Each step pays more than the last.
PATH: list[tuple[int, int, str]] = [
    (24, 3, "Route 29"), (26, 3, "Cherrygrove City"), (26, 1, "Route 30"), (26, 10, "Mr. Pokemon's house"),
    (24, 5, "Elm's lab (egg)"), (26, 2, "Route 31"), (10, 5, "Violet City"), (10, 7, "Violet Gym"),
]
_ROADS = {(g, n) for g, n, name in PATH if name.startswith("Route") or name.endswith("City")}  # not interiors
PATH_REWARDS = [3.0 + i for i in range(len(PATH))]
AREAS: dict[tuple[int, int], str] = {
    (24, 4): "New Bark Town", (24, 5): "Elm's lab", (24, 6): "player's house 1F", (24, 7): "player's house 2F",
    (24, 8): "neighbor's house", (24, 9): "Elm's house", (24, 2): "Route 27",
    (24, 13): "Route 29-46 gate", (5, 9): "Route 46", (26, 4): "Cherrygrove Mart", (26, 5): "Cherrygrove Pokecenter",
    (26, 7): "guide gent's house", **{(g, n): name for g, n, name in PATH},
}

Read = Callable[[int], int]


def _bits(read: Read, start: int, end: int) -> int:
    return sum(bin(read(a)).count("1") for a in range(start, end))


class SilverStats:
    """Game progress, read from RAM. Every count is a high-water mark, so nothing can be farmed by toggling."""

    def __init__(self, read: Read, path: int = 0):
        self.read = read
        self.start_path = path
        self.tiles: set[tuple[int, int, int, int]] = set()
        self.maps: dict[tuple[int, int], None] = {}  # an ordered set: the order areas were reached in
        self.path = path
        self.road_tiles = 0  # new tiles on the PATH maps
        self.base = self._raw()
        self.best = dict(self.base)
        self.observe()
        self.start_tiles, self.start_maps = len(self.tiles), len(self.maps)
        self.start_road_tiles = self.road_tiles

    def _raw(self) -> dict[str, int]:
        r = self.read
        party = min(r(PARTY_COUNT), 6)
        return {"badge": _bits(r, JOHTO_BADGES, KANTO_BADGES + 1), "event": _bits(r, EVENTS_START, EVENTS_END),
                "caught": _bits(r, CAUGHT_START, CAUGHT_END),
                "level": sum(r(PARTY + i * MON_SIZE + LEVEL) for i in range(party)),
                "exp": sum(r(PARTY + i * MON_SIZE + EXP) << 16 | r(PARTY + i * MON_SIZE + EXP + 1) << 8
                           | r(PARTY + i * MON_SIZE + EXP + 2) for i in range(party))}

    def observe(self) -> None:
        for k, v in self._raw().items():
            self.best[k] = max(self.best[k], v)
        if self.read(BATTLE_MODE) == 0:
            m = (self.read(MAP_GROUP), self.read(MAP_NUMBER))
            self.maps[m] = None
            if self.path < len(PATH) and m == PATH[self.path][:2]:
                self.path += 1
            tile = (*m, self.read(X), self.read(Y))
            if tile not in self.tiles and m in _ROADS:
                self.road_tiles += 1
            self.tiles.add(tile)

    def progress(self) -> dict[str, int]:
        gained = {k: self.best[k] - self.base[k] for k in self.base}
        return {**gained, "map": len(self.maps) - self.start_maps, "tile": len(self.tiles) - self.start_tiles,
                "road_tile": self.road_tiles - self.start_road_tiles, "path": self.path}

    def value(self) -> float:
        p = self.progress()
        return sum(w * p[k] for k, w in WEIGHTS.items()) + sum(PATH_REWARDS[self.start_path:self.path])

    def areas(self) -> list[str]:
        return [AREAS.get(m, f"map {m[0]}.{m[1]}") for m in self.maps]

    def hp_fraction(self) -> float:
        hp = full = 0
        for i in range(min(self.read(PARTY_COUNT), 6)):
            a = PARTY + i * MON_SIZE
            hp += self.read(a + HP) * 256 + self.read(a + HP + 1)
            full += self.read(a + MAX_HP) * 256 + self.read(a + MAX_HP + 1)
        return hp / full if full else 0.0


class StallClock:
    """Ends an episode that stopped making progress: a stuck game teaches nothing and fills the batch."""

    def __init__(self, limit: int):
        self.limit, self.dry = limit, 0

    def update(self, progressed: bool, patience: int = 1) -> bool:
        self.dry = 0 if progressed else self.dry + 1
        return self.dry >= self.limit * patience


class Frontier:
    """The furthest point on the path any game has reached, so some games practise there instead of at home.

    With a directory, all processes share it: one lucky game moves everyone's starting point forward.
    """

    def __init__(self, directory: Path | None = None, share: bool = True):
        self.directory = directory if share else None
        self.state: bytes | None = None
        self.path = 0

    def _sync(self) -> None:
        if self.directory is None or not self.directory.is_dir():
            return
        found = sorted(self.directory.glob("path_*.state"))
        if found and (path := int(found[-1].stem.split("_")[1])) > self.path:
            self.state, self.path = found[-1].read_bytes(), path

    def reached(self, pyboy, path: int) -> None:
        if path > self.path:
            buf = io.BytesIO()
            pyboy.save_state(buf)
            self.state, self.path = buf.getvalue(), path
            if self.directory is not None:
                self.directory.mkdir(parents=True, exist_ok=True)
                tmp = self.directory / f".path_{path:02d}.{os.getpid()}.tmp"
                tmp.write_bytes(self.state)
                tmp.replace(self.directory / f"path_{path:02d}.state")  # atomic: readers never see half a state

    def pick(self, rng: np.random.Generator, prob: float) -> tuple[io.BytesIO, int] | None:
        if prob > 0:
            self._sync()
        if self.state is None or rng.random() >= prob:
            return None
        return io.BytesIO(self.state), self.path


class SilverEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: Settings):
        from pyboy import PyBoy

        for label, path in (("ROM", cfg.rom_file), ("start state", cfg.init_state_path)):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        self.cfg = cfg
        self.pyboy = PyBoy(str(cfg.rom_file), window="null", cgb=True, sound_emulated=False)
        self.pyboy.set_emulation_speed(0)
        self.action_space = spaces.Discrete(len(BUTTONS))
        self.observation_space = spaces.Dict({
            "screens": spaces.Box(0, 255, (72, 80, 3 * FRAME_STACK), np.uint8),
            "visited": spaces.Box(0, 255, (48, 48, 1), np.uint8),
            "state": spaces.Box(-1.0, 1.0, (10,), np.float32),
            "areas": spaces.Box(0.0, 1.0, (len(AREAS),), np.float32),  # which known areas this game has reached
            "recent_actions": spaces.MultiDiscrete([len(BUTTONS)] * FRAME_STACK),
        })
        self.stats: SilverStats | None = None
        self.frontier = Frontier(cfg.run_dir / "frontier", share=cfg.frontier_prob > 0)

    def _read(self, addr: int) -> int:
        return self.pyboy.memory[1, addr]  # WRAM bank 1, whatever bank the game has paged in

    def render(self) -> np.ndarray:
        return np.array(self.pyboy.screen.ndarray[:, :, :3])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        further = self.frontier.pick(self.np_random, self.cfg.frontier_prob)
        if further:
            self.pyboy.load_state(further[0])
        else:
            with open(self.cfg.init_state_path, "rb") as f:
                self.pyboy.load_state(f)
        self.pyboy.tick(1 + int(self.np_random.integers(0, 60)), True)  # vary the RNG so envs are not clones
        self.stats = SilverStats(self._read, path=further[1] if further else 0)
        self.value, self.steps = 0.0, 0
        self.stall = StallClock(self.cfg.stall_steps)
        self.screens = np.zeros((72, 80, 3 * FRAME_STACK), np.uint8)
        self.recent = np.zeros(FRAME_STACK, np.int64)
        return self._obs(), {}

    def step(self, action):
        button = BUTTONS[int(action)]
        self.pyboy.button_press(button)
        self.pyboy.tick(PRESS_FRAMES, False)
        self.pyboy.button_release(button)
        return self._advance(int(action), ACTION_FRAMES - PRESS_FRAMES)

    def step_nothing(self):
        """Same clock as step(), no button: the do-nothing baseline."""
        return self._advance(0, ACTION_FRAMES)

    def _advance(self, action: int, frames: int):
        self.pyboy.tick(frames - 1, False)
        self.pyboy.tick(1, True)
        self.stats.observe()
        self.frontier.reached(self.pyboy, self.stats.path)
        self.recent = np.roll(self.recent, 1)
        self.recent[0] = action
        self.steps += 1
        new = self.stats.value()
        reward, self.value = new - self.value, new
        return self._obs(), float(reward), self.stall.update(reward > 0, patience=1 + self.stats.path), self.steps >= self.cfg.ep_length, {"progress": self.stats.progress()}

    def _obs(self) -> dict:
        self.screens = np.roll(self.screens, 3, axis=2)
        self.screens[:, :, :3] = self.render()[::2, ::2]
        r, s = self._read, self.stats
        group, number, x, y = r(MAP_GROUP), r(MAP_NUMBER), r(X), r(Y)
        visited = np.zeros((48, 48, 1), np.uint8)
        for g, n, tx, ty in s.tiles:
            if (g, n) == (group, number) and abs(tx - x) < 12 and abs(ty - y) < 12:
                c, w = (tx - x + 12) * 2, (ty - y + 12) * 2
                visited[w:w + 2, c:c + 2] = 255
        vec = [s.hp_fraction(), min(1.0, s.best["level"] / 100), min(1.0, s.best["badge"] / 16),
               float(r(BATTLE_MODE) > 0), group / 32, number / 32, min(1.0, x / 64), min(1.0, y / 64),
               min(6, r(PARTY_COUNT)) / 6, s.path / len(PATH)]
        return {"screens": self.screens.copy(), "visited": visited,
                "areas": np.array([m in s.maps for m in AREAS], np.float32),
                "state": np.clip(np.array(vec, np.float32), -1, 1), "recent_actions": self.recent.copy()}

    # ProgressCallback's interface (same names as PWhiddy's Red env)
    def get_badges(self) -> int:
        return self.stats.best["badge"]

    def get_all_events_reward(self) -> int:
        return self.stats.progress()["event"]

    @property
    def explore_map(self) -> np.ndarray:
        return np.ones(len(self.stats.tiles), np.uint8)

    @property
    def max_map_progress(self) -> int:
        return len(self.stats.maps)

    def close(self):
        self.pyboy.stop(save=False)


class DoNothingEarnsReward(RuntimeError):
    """The reward is broken: refuse to spend money training on it."""


def play_episode(env, steps: int, choose: Callable[[dict], int | None], heartbeat: Callable[[], None] | None = None,
                 frames: list | None = None) -> dict[str, float]:
    """Play `steps` agent steps; choose() returns an action, or None to press nothing."""
    obs, _ = env.reset()
    total, info = 0.0, {"progress": env.stats.progress()}
    for _ in range(steps):
        if heartbeat is not None:
            heartbeat()
        action = choose(obs)
        obs, reward, stalled, truncated, info = env.step_nothing() if action is None else env.step(action)
        total += reward
        if frames is not None:
            frames.append(env.render())
        if stalled or truncated:
            break
    return {"return": total, **{k: float(v) for k, v in info["progress"].items()}}


def play_episodes(env, steps: int, episodes: int, choose: Callable[[dict], int | None],
                  heartbeat: Callable[[], None] | None = None, frames: list | None = None) -> dict[str, float]:
    """Mean stats over several games (best path too); frames, when given, record the first game."""
    runs = [play_episode(env, steps, choose, heartbeat, frames if i == 0 else None) for i in range(episodes)]
    mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    return {**mean, "path_best": max(r["path"] for r in runs), "episodes": float(episodes)}


def run_baselines(env, steps: int, heartbeat: Callable[[], None] | None = None, episodes: int = 1) -> dict[str, float]:
    nothing = play_episodes(env, steps, episodes, lambda obs: None, heartbeat)
    if nothing["return"] > 0:
        raise DoNothingEarnsReward(f"doing nothing earned {nothing['return']}: {nothing}")
    random = play_episodes(env, steps, episodes, lambda obs: int(env.action_space.sample()), heartbeat)
    return {**{f"baseline/nothing_{k}": v for k, v in nothing.items()},
            **{f"baseline/random_{k}": v for k, v in random.items()}}
