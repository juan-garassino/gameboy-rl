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

from gameboy_rl.achievements import BADGES, STORY, AchievementLog
from gameboy_rl.config import Settings

PLAYER_NAME, MAP_GROUP, MAP_NUMBER, Y, X = 0xD1A3, 0xDA00, 0xDA01, 0xDA02, 0xDA03
JOHTO_BADGES, KANTO_BADGES, BATTLE_MODE = 0xD57C, 0xD57D, 0xD116
EVENTS_START, EVENTS_END = 0xD7B7, 0xD8B7
CAUGHT_START, CAUGHT_END = 0xDBE4, 0xDC04
PARTY_COUNT, PARTY, MON_SIZE, EXP, LEVEL, HP, MAX_HP = 0xDA22, 0xDA2A, 0x30, 0x08, 0x1F, 0x22, 0x24

BUTTONS = ["down", "left", "right", "up", "a", "b"]  # no START: menus and saving are not progress
ACTION_FRAMES, PRESS_FRAMES = 24, 8
FRAME_STACK = 3
# Story flags that mean real progress (numbers: achievements.STORY), in points. Other flags pay nothing.
STORY_PAY = {30: 5.0, 31: 15.0, **{i: 30.0 for i in range(1213, 1221)}}
HEAL_PAY, HEAL_MAX = 0.5, 3.0  # per full party restored outside battle, at most this much per game
WEIGHTS = {"badge": 50.0, "story": 1.0, "healed": 1.0, "map": 0.1, "caught": 1.0, "level": 2.0, "exp": 0.03, "tile": 0.005, "road_tile": 0.045, "retraced": 0.0}  # road_tile adds to tile; retraced is only counted

# The story's road to the first badge, from pret/pokegold's map_constants.asm. Each step pays more than the last.
PATH: list[tuple[int, int, str]] = [
    (24, 3, "Route 29"), (26, 3, "Cherrygrove City"), (26, 1, "Route 30"), (26, 10, "Mr. Pokemon's house"),
    (24, 5, "Elm's lab (egg)"), (26, 2, "Route 31"), (10, 5, "Violet City"), (10, 7, "Violet Gym"),
]
# Steps that are done only when the game's own flag says so (being in the room is not enough): the egg got, the egg given.
NEEDS_FLAG = {3: 30, 4: 31}
# Where each map lies along the road, so a compass can say how far the next goal is (interiors not listed keep the last).
ROAD_ORDER = {(24, 4): 0, (24, 5): 0, (24, 3): 1, (26, 3): 2, (26, 1): 3, (26, 10): 4, (26, 2): 4, (10, 5): 5, (10, 7): 6}
COMPASS = 3.0  # paid per map closer to the next goal (and in part per tile), taken back when moving away: loops cancel
_LAB, _HOUSE = (24, 5), (26, 10)  # Elm's, Mr. Pokemon's
_DOORS = {_LAB: 0.45, _HOUSE: 3.89}  # where those two doors stand along the road


def road_position(m: tuple[int, int], x: int, y: int) -> float | None:
    """How far along the road home -> Violet City a tile is, in maps (3.5 = halfway up Route 30). None: not on the road.

    Map sizes are read from the game (New Bark 20x18 tiles, Route 29 60x18, Cherrygrove 40x18, Route 30 20x54).
    """
    if m in _DOORS:
        return _DOORS[m]
    if m == (24, 4):  # New Bark: the way out is west, around (0, 8)
        return 1 - min(1.0, (x + abs(y - 8)) / 20)
    if m == (24, 3):  # Route 29, walked east to west
        return 1 + min(1.0, max(0, 59 - x) / 60)
    if m == (26, 3):  # Cherrygrove: in at (39, 7), out north around (17, 0)
        return 2 + 1 - min(1.0, (abs(x - 17) + y) / 29)
    if m == (26, 1):  # Route 30, walked south to north
        return 3 + min(1.0, max(0, 53 - y) / 54)
    if m == (26, 2):  # Route 31, walked east to west
        return 4 + min(1.0, max(0, 39 - x) / 40)
    return {(10, 5): 5.0, (10, 7): 6.0}.get(m)


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
        self.start_story = self._story()
        self.story = 0.0
        self.healed = 0.0
        self._hp: tuple[float, int, int] | None = None  # (health, levels, party) when last seen outside battle
        self.maps: dict[tuple[int, int], None] = {}  # an ordered set: the order areas were reached in
        self.path = path
        self.road_tiles = 0  # new tiles on the PATH maps
        self.shaping = 0.0  # the compass: summed since the start, changes with the map, never counts as progress
        self._near: float | None = None
        self._goal = self._here = 0
        self._goal_step = -1
        self._away: int | None = None
        self._step_away: tuple[int, int] = (path, 99)  # (path step, fewest maps left to the goal during it)
        self.retraced = 0  # old PATH tiles walked again after a later step of the path (each once per step)
        self.road_seen: set[tuple[int, int, int, int, int]] = set()
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

    def _story(self) -> frozenset[int]:
        return frozenset(i for i in STORY_PAY if self.read(EVENTS_START + i // 8) >> (i % 8) & 1)

    def _watch_story_and_health(self) -> None:
        new = self._story() - self.start_story
        self.story = max(self.story, sum(STORY_PAY[i] for i in new))
        if self.read(BATTLE_MODE) != 0:
            return
        now = (self.hp_fraction(), self.best["level"], min(self.read(PARTY_COUNT), 6))
        if self._hp is not None and self._hp[0] > 0.001 and now[1:] == self._hp[1:] and now[0] - self._hp[0] > 0.2:
            self.healed = min(HEAL_MAX, self.healed + HEAL_PAY * (now[0] - self._hp[0]))  # blackouts (from 0) and level-ups do not count
        self._hp = now

    def observe(self) -> None:
        for k, v in self._raw().items():
            self.best[k] = max(self.best[k], v)
        self._watch_story_and_health()
        if self.read(BATTLE_MODE) == 0:
            m = (self.read(MAP_GROUP), self.read(MAP_NUMBER))
            self.maps[m] = None
            if self.path < len(PATH) and m == PATH[self.path][:2] and self._flag_done(NEEDS_FLAG.get(self.path)):
                self.path += 1
            tile = (*m, self.read(X), self.read(Y))
            self._compass()
            if self._closest_yet() and m in _ROADS and (self.path, *tile) not in self.road_seen:
                self.road_seen.add((self.path, *tile))
                self.road_tiles += 1  # once per path step, new or old ground, but never on maps away from the goal
                self.retraced += tile in self.tiles
            self.tiles.add(tile)

    def _flag_done(self, i: int | None) -> bool:
        return i is None or bool(self.read(EVENTS_START + i // 8) >> (i % 8) & 1)

    def _closest_yet(self) -> bool:
        """Is this map as near the goal as this path step has got? The wrong way pays nothing."""
        if self.path >= len(PATH) or self.away is None:
            return True
        step, best = self._step_away
        best = min(best, self.away) if step == self.path else self.away
        self._step_away = (self.path, best)
        return self.away <= best

    @property
    def away(self) -> int | None:
        """Maps to the next goal, plus one until the goal's own map is entered (None: not known yet)."""
        return self._away

    def _compass(self) -> None:
        m = (self.read(MAP_GROUP), self.read(MAP_NUMBER))
        here = ROAD_ORDER.get(m)
        position = road_position(m, self.read(X), self.read(Y))
        if here is None or position is None or self.path >= len(PATH):
            return
        goal_map = PATH[self.path][:2]
        in_goal, goal = m == goal_map, ROAD_ORDER[goal_map]
        self._here, self._away = here, abs(goal - here) + (not in_goal)
        target = _DOORS.get(goal_map, float(goal))
        rest = 0.0 if in_goal else 0.5
        if in_goal and m == _LAB:  # up to Elm: the egg was handed over from (4, 2)
            rest = 0.5 * min(1.0, (abs(self.read(X) - 4) + abs(self.read(Y) - 2)) / 12)
        near = -(abs(target - position) + rest)
        if self._near is not None and self._goal_step == self.path:
            self.shaping += COMPASS * (near - self._near)
        self._near, self._goal, self._goal_step = near, goal, self.path

    def progress(self) -> dict[str, int]:
        gained = {k: self.best[k] - self.base[k] for k in self.base}
        return {**gained, "map": len(self.maps) - self.start_maps, "tile": len(self.tiles) - self.start_tiles,
                "story": self.story, "healed": self.healed, "road_tile": self.road_tiles - self.start_road_tiles, "retraced": self.retraced, "path": self.path}

    def value(self) -> float:
        return self.base_value() + self.shaping

    def base_value(self) -> float:
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
    """Points along the story any game has reached, so some games practise there instead of at home.

    One state per (path step, maps left to the goal): the first game to be there. Half the practice
    starts are the furthest point, half any of them, so the way between two steps is practised too. With a directory, all
    processes share the points.
    """

    def __init__(self, directory: Path | None = None, share: bool = True):
        self.directory = directory if share else None
        self.states: dict[tuple[int, int], bytes] = {}  # (path, away) -> game state

    def _sync(self) -> None:
        if self.directory is None or not self.directory.is_dir():
            return
        for f in self.directory.glob("path_*_away_*.state"):
            _, path, _, away = f.stem.split("_")
            if (key := (int(path), int(away))) not in self.states:
                self.states[key] = f.read_bytes()

    def reached(self, pyboy, path: int, away: int | None) -> None:
        if path == 0 or away is None or (path, away) in self.states:
            return
        buf = io.BytesIO()
        pyboy.save_state(buf)
        self.states[(path, away)] = buf.getvalue()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / f"path_{path:02d}_away_{away:02d}.state"
            if not target.exists():
                tmp = self.directory / f".{target.stem}.{os.getpid()}.tmp"
                tmp.write_bytes(buf.getvalue())
                tmp.replace(target)  # atomic: readers never see half a state

    def pick(self, rng: np.random.Generator, prob: float) -> tuple[io.BytesIO, int] | None:
        if prob > 0:
            self._sync()
        if not self.states or rng.random() >= prob:
            return None
        keys = sorted(self.states, key=lambda k: (k[0], -k[1]))
        key = keys[-1] if rng.random() < 0.5 else keys[int(rng.integers(len(keys)))]
        return io.BytesIO(self.states[key]), key[0]


N_STORY = len(STORY) + len(BADGES) + len(PATH) + 1 + 1  # flags, badges, goal one-hot (with "done"), closeness


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
            "story": spaces.Box(0.0, 1.0, (N_STORY,), np.float32),  # what the game says has happened, and the next goal
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
        self.value, self.base, self.steps = 0.0, 0.0, 0
        self.stall = StallClock(self.cfg.stall_steps)
        self.log = AchievementLog(self.cfg.run_dir / "achievements", self._read)
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
        self.frontier.reached(self.pyboy, self.stats.path, self.stats.away)
        if self.cfg.frontier_prob > 0:  # training games only: evals are scored separately
            self.log.update(self.stats.path, self.steps)
        self.recent = np.roll(self.recent, 1)
        self.recent[0] = action
        self.steps += 1
        new = self.stats.value()
        reward, self.value = new - self.value, new
        base = self.stats.base_value()
        advanced, self.base = base > self.base, base  # the compass must not keep a stuck game alive
        return self._obs(), float(reward), self.stall.update(advanced, patience=1 + self.stats.path), self.steps >= self.cfg.ep_length, {"progress": self.stats.progress()}

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
                "story": self._story_obs(),
                "state": np.clip(np.array(vec, np.float32), -1, 1), "recent_actions": self.recent.copy()}

    def _story_obs(self) -> np.ndarray:
        s, r = self.stats, self._read
        flags = [r(EVENTS_START + i // 8) >> (i % 8) & 1 for i in STORY]
        badges = [r(JOHTO_BADGES) >> k & 1 for k in range(len(BADGES))]
        goal = [float(s.path == k) for k in range(len(PATH) + 1)]
        near = 1.0 - min(6, abs(s._goal - s._here)) / 6 if s._near is not None else 0.0
        return np.array([*flags, *badges, *goal, near], np.float32)

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
