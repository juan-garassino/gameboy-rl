"""Silver reward and baselines on a fake RAM (no emulator, no ROM)."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from gameboy_rl import silver
from gameboy_rl.silver import DoNothingEarnsReward, SilverStats, StallClock, play_episode, run_baselines


class _Ram(dict):
    def __missing__(self, key):
        return 0


def _ram():
    r = _Ram({silver.MAP_GROUP: 24, silver.MAP_NUMBER: 4, silver.X: 6, silver.Y: 5, silver.PARTY_COUNT: 1})
    r[silver.PARTY + silver.LEVEL] = 5
    r[silver.EVENTS_START] = 0b101
    return r


def test_doing_nothing_is_worth_nothing():
    s = SilverStats(_ram().__getitem__)
    for _ in range(100):
        s.observe()
    assert s.value() == 0.0


@pytest.mark.parametrize("change,expected", [
    ({silver.JOHTO_BADGES: 1}, 50.0), ({silver.EVENTS_START: 0b111}, 2.0), ({silver.CAUGHT_START: 1}, 1.0),
    ({silver.PARTY + silver.LEVEL: 6}, 0.5), ({silver.X: 7}, 0.005), ({silver.MAP_NUMBER: 5}, 0.105)])
def test_progress_is_rewarded(change, expected):
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    ram.update(change)
    s.observe()
    assert s.value() == pytest.approx(expected)


def _visit(ram, stats, *maps):
    for group, number in maps:
        ram[silver.MAP_GROUP], ram[silver.MAP_NUMBER] = group, number
        stats.observe()


def test_the_story_path_pays_more_the_further_it_gets():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    gains, last = [], 0.0
    for group, number, _ in silver.PATH:
        _visit(ram, s, (group, number))
        gains.append(s.value() - last)
        last = s.value()
    assert s.progress()["path"] == len(silver.PATH) and gains == sorted(gains) and gains[0] >= 2.0
    _visit(ram, s, *[(g, n) for g, n, _ in silver.PATH], (24, 4))
    assert s.value() == pytest.approx(last)  # walking back pays nothing


def test_all_the_houses_in_town_pay_less_than_leaving_it():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    _visit(ram, s, (24, 5), (24, 6), (24, 7), (24, 8), (24, 9))
    houses = s.value()
    _visit(ram, s, (24, 3))
    assert s.value() - houses > houses
    assert s.areas()[-1] == "Route 29"


def test_walking_the_story_roads_pays_ten_times_walking_around_town():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    ram[silver.X] = 7
    s.observe()
    town = s.value()
    _visit(ram, s, (24, 3))
    on_road = s.value()
    ram[silver.X] = 8
    s.observe()
    assert s.value() - on_road == pytest.approx(10 * town) and s.progress()["road_tile"] == 2


def test_path_steps_must_be_reached_in_order():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    _visit(ram, s, (26, 10))
    assert s.progress()["path"] == 0


def test_toggling_and_backtracking_cannot_be_farmed():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    for _ in range(20):
        for flags, x in ((0b111, 7), (0b101, 6)):
            ram[silver.EVENTS_START], ram[silver.X] = flags, x
            s.observe()
    assert s.value() == pytest.approx(2.0 + 0.005)


def test_winning_battles_pays_through_experience():
    ram = _ram()
    ram[silver.PARTY + silver.EXP + 2] = 135
    s = SilverStats(ram.__getitem__)
    assert s.value() == 0.0
    ram[silver.PARTY + silver.EXP + 1], ram[silver.PARTY + silver.EXP + 2] = 1, 0  # 256: big-endian, 3 bytes
    s.observe()
    assert s.progress()["exp"] == 121 and s.value() == pytest.approx(121 * silver.WEIGHTS["exp"])


def test_battles_do_not_count_as_exploring():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    ram[silver.BATTLE_MODE], ram[silver.X] = 1, 40
    s.observe()
    assert s.value() == 0.0


def test_stall_clock_ends_an_episode_only_after_a_dry_spell():
    clock = StallClock(limit=3)
    assert [clock.update(False), clock.update(False), clock.update(True)] == [False, False, False]
    assert [clock.update(False), clock.update(False), clock.update(False)] == [False, False, True]


def test_a_game_that_got_further_is_given_more_time():
    clock = StallClock(limit=3)
    assert [clock.update(False, patience=2) for _ in range(6)] == [False, False, False, False, False, True]


def test_the_path_goes_back_to_elm_with_the_egg():
    names = [name for _, _, name in silver.PATH]
    assert names.index("Mr. Pokemon's house") + 1 == names.index("Elm's lab (egg)") < names.index("Route 31")


def test_a_game_can_start_further_along_the_path():
    ram = _ram()
    s = SilverStats(ram.__getitem__, path=2)
    assert s.value() == 0.0 and s.path == 2
    _visit(ram, s, silver.PATH[2][:2])
    assert s.path == 3 and s.value() == pytest.approx(silver.PATH_REWARDS[2] + 0.15)


class _Boy:
    def save_state(self, f):
        f.write(b"cherrygrove")


def test_frontier_remembers_the_furthest_point_and_replays_it():
    f = silver.Frontier()
    assert f.pick(np.random.default_rng(0), prob=1.0) is None  # nothing reached yet: normal start
    f.reached(_Boy(), 2)
    f.reached(None, 1)  # not further: ignored, the emulator is not even asked
    state, path = f.pick(np.random.default_rng(0), prob=1.0)
    assert state.read() == b"cherrygrove" and path == 2
    assert f.pick(np.random.default_rng(0), prob=0.0) is None


def test_stall_limit_and_entropy_bonus_are_settings():
    from gameboy_rl.config import Settings
    from gameboy_rl.train import ppo_params

    cfg = Settings(game="silver", stall_steps=7, ent_coef=0.02)
    assert cfg.stall_steps == 7 and ppo_params(cfg)["ent_coef"] == 0.02
    assert ppo_params(Settings())["ent_coef"] == 0.01 and ppo_params(Settings())["n_epochs"] == 1  # Red keeps PWhiddy's
    assert ppo_params(Settings(game="silver", n_epochs=3))["n_epochs"] == 3


def test_hp_fraction_is_big_endian():
    ram = _ram()
    ram.update({silver.PARTY + silver.HP: 1, silver.PARTY + silver.HP + 1: 0, silver.PARTY + silver.MAX_HP: 2})
    assert SilverStats(ram.__getitem__).hp_fraction() == 0.5


class _StubEnv:
    action_space = Discrete(6)

    def __init__(self, idle_reward=0.0):
        self.idle_reward, self.calls, self.ends_at = idle_reward, [], None
        self.stats = SilverStats(_ram().__getitem__)

    def reset(self):
        return {}, {}

    def step(self, action):
        self.calls.append(action)
        return {}, 1.0, self.ends_at == len(self.calls), False, {"progress": self.stats.progress()}

    def step_nothing(self):
        self.calls.append(None)
        return {}, self.idle_reward, False, False, {"progress": self.stats.progress()}

    def render(self):
        return np.zeros((144, 160, 3), np.uint8)


def test_baselines_compare_nothing_and_random():
    out = run_baselines(_StubEnv(), steps=5)
    assert out["baseline/nothing_return"] == 0.0 and out["baseline/random_return"] == 5.0


def test_refuses_to_train_when_doing_nothing_pays():
    with pytest.raises(DoNothingEarnsReward):
        run_baselines(_StubEnv(idle_reward=0.01), steps=5)


def test_play_episode_stops_when_the_game_stalls():
    env = _StubEnv()
    env.ends_at = 3
    play_episode(env, 50, lambda obs: 0)
    assert len(env.calls) == 3


def test_baselines_average_several_games():
    class _Env(_StubEnv):
        games = 0

        def reset(self):
            self.games += 1
            return {}, {}

    env = _Env()
    out = run_baselines(env, steps=5, episodes=4)
    assert env.games == 8 and out["baseline/random_episodes"] == 4


def test_play_episode_records_color_frames():
    frames = []
    play_episode(_StubEnv(), 4, lambda obs: 0, frames=frames)
    assert len(frames) == 4 and frames[0].shape == (144, 160, 3)


def test_eval_callback_checks_baselines_then_writes_a_video(tmp_path, monkeypatch):
    from gameboy_rl import silver_eval
    from gameboy_rl.config import Settings

    uploads = []
    monkeypatch.setattr(silver_eval.mlflow_tracking, "log_artifact", lambda p, a=None: uploads.append((p.name, a)))
    monkeypatch.setattr(silver_eval.mlflow_tracking, "log_metrics", lambda m, step: None)

    class _Env(_StubEnv):
        def close(self):
            pass

    class _Model:
        num_timesteps = 0
        logger = None

        def get_env(self):
            return None

        def predict(self, obs, deterministic):
            return np.array([2]), None

    cfg = Settings(game="silver", eval_freq=100, eval_steps=4)
    cb = silver_eval.SilverEvalCallback(cfg, tmp_path, env_factory=lambda c: _Env())
    model = _Model()
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert ("baselines.json", None) in uploads
    model.num_timesteps = 100
    cb.on_step()
    assert ("eval_0000000100.mp4", "videos") in uploads
    assert (tmp_path / "videos" / "eval_0000000100.mp4").stat().st_size > 0


def test_silver_settings_and_dispatch():
    from gameboy_rl.config import Settings

    assert Settings().game == "red"
    assert Settings(game="silver").n_steps == Settings(game="silver").ep_length // 64
