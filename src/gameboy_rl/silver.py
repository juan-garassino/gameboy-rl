"""Pokémon Silver (Game Boy Color, in color): env, reward and game-outcome stats.

RAM addresses come from pret/pokegold's pokesilver.sym; the ROM is byte-identical to that build (SHA-1 49b163f7…).
The reward pays only for game progress a player would recognise, so doing nothing must earn exactly nothing.
"""
from __future__ import annotations

import functools
import io
import json
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
# Fighting pays up to what the first gym needs, then nothing: unbounded, it was worth more than the story (6 of 6 games
# with the egg went back to grind Route 30); with nothing, games ran from every battle.
LEVEL_TARGET = 14  # Falkner's Pidgeotto is level 9; a lead at 14 should win
WINS_PAID = 15  # battles won (the enemy fainted) that pay, per game
WEIGHTS = {"badge": 50.0, "story": 1.0, "healed": 1.0, "map": 0.1, "caught": 1.0, "level": 1.5, "exp": 0.002, "won": 0.75, "tile": 0.005, "road_tile": 0.045, "retraced": 0.0}  # road_tile adds to tile; retraced is only counted

# The story's road to the first badge, from pret/pokegold's map_constants.asm. Each step pays more than the last.
PATH: list[tuple[int, int, str]] = [
    (24, 3, "Route 29"), (26, 3, "Cherrygrove City"), (26, 1, "Route 30"), (26, 10, "Mr. Pokemon's house"),
    (24, 5, "Elm's lab (egg)"), (26, 2, "Route 31"), (10, 5, "Violet City"), (10, 7, "Violet Gym"),
]
# Steps that are done only when the game's own flag says so (being in the room is not enough): the egg got, the egg given.
NEEDS_FLAG = {3: 30, 4: 31}
# Where each map lies along the road, so a compass can say how far the next goal is (interiors not listed keep the last).
ROAD_ORDER = {(24, 6): 0, (24, 7): 0, (24, 4): 0, (24, 5): 0, (24, 3): 1, (26, 3): 2, (26, 1): 3, (26, 10): 4, (26, 2): 4, (10, 5): 5, (10, 7): 6}
COMPASS = 3.0  # paid per map closer to the next goal (and in part per tile), taken back when moving away: loops cancel
_LAB, _HOUSE = (24, 5), (26, 10)  # Elm's, Mr. Pokemon's
_DOORS = {_LAB: 0.45, _HOUSE: 3.89}  # where those two doors stand along the road
_EXITS = {_LAB: (4, 11), _HOUSE: (2, 7)}  # the tile inside each room by its door
# Where you can stand in each room, found by walking it at random in the game. Straight-line distance to the door left a
# pocket behind the Poké Ball table that looked farther from the door the way out; walking distance has none.
_WALKABLE = {
    _LAB: {(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (1, 8), (2, 2), (2, 3), (2, 4), (2, 5), (2, 8), (3, 2), (3, 3), (3, 4), (3, 5), (3, 8), (3, 9), (3, 10), (3, 11), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (5, 1), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (6, 4), (6, 5), (6, 8), (6, 9), (6, 10), (7, 4), (7, 5), (7, 8), (7, 9), (7, 10), (7, 11), (8, 4), (8, 5), (8, 8), (8, 9), (8, 10), (8, 11), (9, 4), (9, 5), (9, 8), (9, 9), (9, 10), (9, 11)},
    _HOUSE: {(0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (4, 2), (4, 3), (4, 6), (4, 7), (5, 2), (5, 3), (5, 6), (5, 7), (6, 2), (6, 3), (6, 5), (6, 6), (6, 7), (7, 2), (7, 5), (7, 6), (7, 7)},
}


def _walking_distances(room: tuple[int, int]) -> dict[tuple[int, int], int]:
    tiles, door = _WALKABLE[room], _EXITS[room]
    dist, todo = {door: 0}, [door]
    for x, y in todo:  # breadth first: the list grows while we walk it
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n in tiles and n not in dist:
                dist[n] = dist[(x, y)] + 1
                todo.append(n)
    return dist


_TO_DOOR = {room: _walking_distances(room) for room in _WALKABLE}

# The road maps, flooded in the emulator after the egg was given (silver_maps.json: walkable tiles, one-tile moves, exits).
# Distances on them follow the real path: on Route 30 the way to Route 31 is a U-turn past a trainer, and measured by height
# alone the top right (by Mr. Pokemon's house) looked closest; every game stopped there.
_MAPS = {tuple(map(int, k.split("."))): v for k, v in json.loads((Path(__file__).parent / "silver_maps.json").read_text()).items()}
_ORDER = {**ROAD_ORDER, _LAB: -1}  # the lab lies off New Bark, one step past it on the way home
_ROAD_MAPS = {(24, 4), (24, 3), (26, 3), (26, 1), (26, 2), (10, 5)}
_GYM, _FALKNER = (10, 7), (5, 2)  # Violet Gym; the tile below Falkner (pret/pokegold: he stands at 5, 1)
_OUTSIDE = {_LAB: ((24, 4), (6, 4)), _HOUSE: ((26, 1), (17, 6)), (24, 6): ((24, 4), (13, 6))}  # a room's door, seen from outside


@functools.lru_cache(maxsize=None)
def _to_exit(m: tuple[int, int], target: tuple[int, int]) -> tuple[dict[tuple[int, int], int], int] | None:
    """Walking distance on map m to the tiles that lead into map `target`, and the largest such distance."""
    data = _MAPS.get(m)
    exits = [tuple(e[0]) for e in data["exits"] if tuple(e[2]) == target] if data else []
    if not exits:
        return None
    moves = {(tuple(a), tuple(b)) for a, b in data["moves"]}
    leaves = {a for a, _ in moves}
    moves |= {(b, a) for a, b in moves if b not in leaves}  # tiles the flood could not leave (a trainer in sight): walk back
    back: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for a, b in moves:
        back.setdefault(b, []).append(a)
    dist, todo = {e: 0 for e in exits}, list(exits)
    for p in todo:
        for a in back.get(p, []):
            if a not in dist:
                dist[a] = dist[p] + 1
                todo.append(a)
    return dist, max(dist.values()) or 1


def walking_far(m: tuple[int, int], x: int, y: int, goal_map: tuple[int, int]) -> float | None:
    """Maps still to cross to reach goal_map, counting this one by walking distance; None where the map data does not help."""
    if m in _OUTSIDE and m != goal_map:  # in a room that is not the goal: from its door outside, plus the way to that door
        om, (ox, oy) = _OUTSIDE[m]
        outside = walking_far(om, ox, oy, goal_map)
        if outside is None:
            return None
        inside = _TO_DOOR[m].get((x, y), 12) if m in _TO_DOOR else abs(x - 6) + abs(y - 7)
        return outside + 0.15 * (inside + 1) / 12
    if m not in _MAPS or goal_map not in _ORDER or m not in _ORDER:
        return None
    exits = {tuple(e[2]) for e in _MAPS[m]["exits"]}
    if goal_map in exits:
        step = goal_map
    else:
        toward = [c for c in exits if c in _ROAD_MAPS and c in _ORDER and abs(_ORDER[goal_map] - _ORDER[c]) < abs(_ORDER[goal_map] - _ORDER[m])]
        if not toward:
            return None
        step = min(toward, key=lambda c: abs(_ORDER[goal_map] - _ORDER[c]))
    found = _to_exit(m, step)
    if found is None or (x, y) not in found[0]:
        return None
    dist, longest = found
    return dist[(x, y)] / longest + max(0, abs(_ORDER[goal_map] - _ORDER[m]) - 1)
_DOOR_TILES = {_LAB: ((24, 4), 6, 3, 20), _HOUSE: ((26, 1), 17, 5, 66)}  # outside: (map, x, y, tiles that count as one map)


def road_position(m: tuple[int, int], x: int, y: int, home: bool = False) -> float | None:
    """How far along the road home -> Violet City a tile is, in maps (3.5 = halfway up Route 30). None: not on the road.

    Within a map it counts the tiles to the exit we want: the one toward Violet, or with `home` the one back. (Measured to
    one exit only, a corner of Cherrygrove looked as good as the way home, and 8 of 8 games stayed in it.)
    Map sizes and doors are read from the game (New Bark 20x18 tiles, Route 29 60x18, Cherrygrove 40x18, Route 30 20x54).
    """
    if m in _DOORS:
        return _DOORS[m]
    if m == (24, 7):  # player's house 2F, where a knockout leaves you: to the stairs
        return -0.05 - 0.15 * (abs(x - 7) + y) / 12
    if m == (24, 6):  # 1F: to the door
        return 0.15 - 0.15 * (abs(x - 6) + abs(y - 7)) / 14
    if m == (24, 4):  # New Bark: the way out is west, around (0, 8)
        return 1 - (x + abs(y - 8)) / 20
    if m == (24, 3):  # Route 29: east end (59, 8) to west end (0, 7)
        return 1 + (59 - x) / 60
    if m == (26, 3):  # Cherrygrove: east end (39, 7), north end (17, 0)
        return 2 + (abs(x - 39) + abs(y - 7)) / 29 if home else 3 - (abs(x - 17) + y) / 29
    if m == (26, 1):  # Route 30, south end to north end
        return 3 + (53 - y) / 54
    if m == (26, 2):  # Route 31, walked east to west
        return 4 + (39 - x) / 40
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
TEXT_PRESSES = 40  # presses at most per step to click through a text box outside battle
TEXT_READ = 12  # of those, A first (reads the text); then B, which also backs out of a menu that A opened (a PC: after
# a knockout, games woke up beside the bedroom PC and spent thousands of steps in its menus)


def text_box(frame: np.ndarray) -> bool:
    """A message box is open: the bottom of the screen white under a dark border line. Calibrated on 1950 frames: never
    while walking, only when the player could not move (outside battle)."""
    white = (frame[104:136, 8:152, :3].min(axis=2) > 200).mean()
    line = (frame[96:101, 8:152, :3].max(axis=2) < 80).mean(axis=1).max()
    return bool(white > 0.7 and line > 0.8)


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
        self.won, self._battle_exp = 0, None  # battles won this game; experience when the current battle began
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

    def _watch_battles(self) -> None:
        """A battle is won when it ends and the party gained experience during it (only a fainted enemy gives any)."""
        fighting = self.read(BATTLE_MODE) != 0
        exp = self._raw()["exp"]
        if fighting and self._battle_exp is None:
            self._battle_exp = exp
        elif not fighting and self._battle_exp is not None:
            if exp > self._battle_exp:
                self.won = min(WINS_PAID, self.won + 1)
            self._battle_exp = None

    def observe(self) -> None:
        self._watch_battles()
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
        if self.path >= len(PATH):  # in the gym: on to Falkner, past his trainers (the badge pays the rest)
            if m == _GYM and self.read(BATTLE_MODE) == 0:
                x, y = self.read(X), self.read(Y)
                near = -0.5 * min(1.0, (abs(x - _FALKNER[0]) + abs(y - _FALKNER[1])) / 14)
                if self._near is not None and self._goal_step == self.path:
                    self.shaping += COMPASS * (near - self._near)
                self._near, self._goal_step = near, self.path
            return
        here = ROAD_ORDER.get(m)
        goal_map = PATH[self.path][:2] if self.path < len(PATH) else None
        position = road_position(m, self.read(X), self.read(Y), home=goal_map == _LAB)
        if here is None or position is None or self.path >= len(PATH):
            return
        in_goal, goal = m == goal_map, ROAD_ORDER[goal_map]
        self._here, self._away = here, abs(goal - here) + (not in_goal)
        target = _DOORS.get(goal_map, float(goal))
        rest = 0.0 if in_goal else 0.5
        if m in _EXITS and not in_goal:  # in a room the goal is not in: out through its door (after the egg, games stayed by Elm)
            ex, ey = _EXITS[m]
            x, y = self.read(X), self.read(Y)
            rest += 0.15 * _TO_DOOR[m].get((x, y), abs(x - ex) + abs(y - ey) + 6) / 12
        if in_goal and m == _LAB:  # up to Elm: the egg was handed over from (4, 2)
            rest = 0.5 * min(1.0, (abs(self.read(X) - 4) + abs(self.read(Y) - 2)) / 12)
        walked = None if in_goal else walking_far(m, self.read(X), self.read(Y), goal_map)
        far = abs(target - position) if walked is None else walked
        if goal_map in _DOOR_TILES and m == _DOOR_TILES[goal_map][0]:  # on the door's own map: walk to the door itself
            _, dx, dy, scale = _DOOR_TILES[goal_map]
            far = (abs(self.read(X) - dx) + abs(self.read(Y) - dy)) / scale
        near = -(far + rest)
        if self._near is not None and self._goal_step == self.path:
            self.shaping += COMPASS * (near - self._near)
        self._near, self._goal, self._goal_step = near, goal, self.path

    def progress(self) -> dict[str, int]:
        gained = {k: self.best[k] - self.base[k] for k in self.base}
        gained["level"] = max(0, min(self.best["level"], LEVEL_TARGET) - min(self.base["level"], LEVEL_TARGET))
        gained["won"] = self.won
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

    One state per (path step, maps left to the goal): the first game to be there. Every path step gets
    the same share of practice, so the early road is not forgotten. With a directory, all
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
        steps = sorted({p for p, _ in self.states})
        step = steps[int(rng.integers(len(steps)))]
        keys = sorted(k for k in self.states if k[0] == step)
        key = keys[int(rng.integers(len(keys)))]
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
        before = self._where()
        self.pyboy.button_press(button)
        self.pyboy.tick(PRESS_FRAMES, False)
        self.pyboy.button_release(button)
        self._bumped = button in ("down", "left", "right", "up") and self._read(BATTLE_MODE) == 0 and self._where() == before
        return self._advance(int(action), ACTION_FRAMES - PRESS_FRAMES)

    def _where(self) -> tuple[int, int, int, int]:
        return self._read(MAP_GROUP), self._read(MAP_NUMBER), self._read(X), self._read(Y)

    def _click_through_text(self) -> None:
        """Outside battle a text box freezes the player until it is clicked through (the aide's scene after the egg
        took about 30 presses and stopped every game): press A for it, so the network is not asked while it cannot act."""
        if self.cfg.click_text and getattr(self, "_bumped", False) and self._where() == self._where_before_tick:
            # Walked into someone or something and did not move: talk to it. After the egg the officer stands in the lab's
            # aisle and only leaves once spoken to; no text box shows until then, and no game got past him.
            self.pyboy.button_press("a")
            self.pyboy.tick(PRESS_FRAMES, False)
            self.pyboy.button_release("a")
            self.pyboy.tick(ACTION_FRAMES - PRESS_FRAMES - 1, False)
            self.pyboy.tick(1, True)
        self._bumped = False
        for i in range(TEXT_PRESSES if self.cfg.click_text else 0):
            if self._read(BATTLE_MODE) != 0 or not text_box(self.pyboy.screen.ndarray):
                return
            button = "a" if i < TEXT_READ else "b"
            self.pyboy.button_press(button)
            self.pyboy.tick(PRESS_FRAMES, False)
            self.pyboy.button_release(button)
            self.pyboy.tick(ACTION_FRAMES - PRESS_FRAMES - 1, False)
            self.pyboy.tick(1, True)

    def step_nothing(self):
        """Same clock as step(), no button: the do-nothing baseline."""
        return self._advance(0, ACTION_FRAMES)

    def _advance(self, action: int, frames: int):
        self._where_before_tick = self._where()
        self.pyboy.tick(frames - 1, False)
        self.pyboy.tick(1, True)
        self._click_through_text()
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
        return self._obs(), float(reward), self.stall.update(advanced, patience=1 + self.stats.path - self.stats.start_path), self.steps >= self.cfg.ep_length, {"progress": self.stats.progress()}

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
