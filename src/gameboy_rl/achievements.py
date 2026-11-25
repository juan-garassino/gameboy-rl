"""What Pokémon Silver itself says has happened: named story flags and badges, read from RAM.

Flag numbers are the positions in pret/pokegold's constants/event_flags.asm (the ROM is byte-identical to that build);
checked on our start state: EVENT_GOT_A_POKEMON_FROM_ELM (26) is set there.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

EVENTS_START = 0xD7B7
JOHTO_BADGES = 0xD57C
STORY: dict[int, str] = {
    26: "got the starter from Elm", 30: "got the Mystery Egg from Mr. Pokemon", 31: "gave the egg to Elm",
    1213: "beat Falkner (gym 1)", 1214: "beat Bugsy (gym 2)", 1215: "beat Whitney (gym 3)", 1216: "beat Morty (gym 4)",
    1217: "beat Jasmine (gym 5)", 1218: "beat Chuck (gym 6)", 1219: "beat Pryce (gym 7)", 1220: "beat Clair (gym 8)",
}
BADGES = ["Zephyr", "Hive", "Plain", "Fog", "Mineral", "Storm", "Glacier", "Rising"]  # Johto, in order


def achieved(read: Callable[[int], int]) -> dict[str, str]:
    """{key: text} for everything the game says is done; badges too, since a badge is what a gym win gives."""
    out = {f"event_{i}": text for i, text in STORY.items() if read(EVENTS_START + i // 8) >> (i % 8) & 1}
    return {**out, **{f"badge_{b}": f"got the {b} Badge" for k, b in enumerate(BADGES) if read(JOHTO_BADGES) >> k & 1}}


class AchievementLog:
    """Writes one file the first time *any* game does something; all processes share the directory."""

    def __init__(self, directory: Path, read: Callable[[int], int]):
        self.directory, self.read = Path(directory), read
        self.known = set(achieved(read))  # what the start state already has is not an achievement

    def update(self, path: int, steps: int) -> list[str]:
        new = []
        for key, text in achieved(self.read).items():
            if key in self.known:
                continue
            self.known.add(key)
            self.directory.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(self.directory / f"{key}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                json.dump({"what": text, "path_step": path, "game_steps": steps, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
            new.append(text)
        return new
