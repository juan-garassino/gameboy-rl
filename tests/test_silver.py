"""Silver reward and baselines on a fake RAM (no emulator, no ROM)."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import Discrete

from gameboy_rl import achievements, silver
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
    ({silver.JOHTO_BADGES: 1}, 50.0), ({silver.CAUGHT_START: 1}, 1.0),
    ({silver.PARTY + silver.LEVEL: 6}, 2.0), ({silver.X: 7}, 0.005), ({silver.MAP_NUMBER: 5}, 0.105)])
def test_progress_is_rewarded(change, expected):
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    ram.update(change)
    s.observe()
    assert s.base_value() == pytest.approx(expected)


def _visit(ram, stats, *maps):
    for group, number in maps:
        ram[silver.MAP_GROUP], ram[silver.MAP_NUMBER] = group, number
        stats.observe()


def test_the_story_path_pays_more_the_further_it_gets():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    gains, last = [], 0.0
    for k, (group, number, _) in enumerate(silver.PATH):
        if k in silver.NEEDS_FLAG:
            _story_flag(ram, silver.NEEDS_FLAG[k])
        _visit(ram, s, (group, number))
        now = s.base_value() - s.story  # the egg flags pay on top of the steps
        gains.append(now - last)
        last = now
    assert s.progress()["path"] == len(silver.PATH) and gains == sorted(gains) and gains[0] >= 2.0
    _visit(ram, s, *[(g, n) for g, n, _ in silver.PATH], (24, 4))
    walked_back = s.base_value()
    _visit(ram, s, *[(g, n) for g, n, _ in silver.PATH], (24, 4))
    assert s.base_value() == pytest.approx(walked_back)  # the same ground again pays nothing


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
    town = s.base_value()
    _visit(ram, s, (24, 3))
    on_road = s.base_value()
    ram[silver.X] = 8
    s.observe()
    assert s.base_value() - on_road == pytest.approx(10 * town) and s.progress()["road_tile"] == 2


def test_road_pays_once_per_step_and_only_toward_the_goal():
    ram = _ram()
    s = SilverStats(ram.__getitem__, path=4)  # practice: the egg in hand, in New Bark, memory of the road empty
    ram[silver.X] = 5
    s.observe()
    before = s.base_value()
    for x in range(59, 30, -1):  # the wrong way, back over all of Route 29
        ram[silver.X] = x
        _visit(ram, s, (24, 3))
    tile = silver.WEIGHTS["tile"]
    assert s.progress()["road_tile"] == 0 and s.base_value() - before < 0.1 + 30 * tile + 1e-9  # exploring pay only
    assert s.shaping < -silver.COMPASS / 2  # and the compass takes more than that back

    s = SilverStats(ram.__getitem__, path=4)  # practice again, now from Route 29: the right way pays every new tile once
    for _ in range(2):
        for x in range(31, 60):
            ram[silver.X] = x
            _visit(ram, s, (24, 3))
        paid = s.progress()["road_tile"]
        assert 27 <= paid <= 29
    assert s.shaping > 0


def test_inside_the_lab_the_compass_leads_to_elm():
    ram = _ram()
    ram[silver.MAP_NUMBER], ram[silver.X], ram[silver.Y] = 5, 4, 11
    s = SilverStats(ram.__getitem__, path=4)
    gains = []
    for y in (9, 6, 2):
        before, ram[silver.Y] = s.shaping, y
        s.observe()
        gains.append(s.shaping - before)
    assert all(g > 0 for g in gains) and sum(gains) == pytest.approx(silver.COMPASS * 0.5 * 9 / 12)


def test_the_compass_works_tile_by_tile_along_the_road():
    along = [silver.road_position((24, 4), 13, 5), silver.road_position((24, 4), 0, 8), silver.road_position((24, 3), 59, 9),
             silver.road_position((24, 3), 20, 9), silver.road_position((26, 3), 39, 7), silver.road_position((26, 3), 17, 0),
             silver.road_position((26, 1), 6, 53), silver.road_position((26, 1), 6, 20), silver.road_position((26, 10), 3, 6),
             silver.road_position((26, 2), 39, 9), silver.road_position((26, 2), 5, 9), silver.road_position((10, 5), 1, 1)]
    assert along == sorted(along) and silver.road_position((24, 6), 1, 1) is None
    ram = _ram()
    ram[silver.MAP_NUMBER], ram[silver.X] = 3, 40
    s = SilverStats(ram.__getitem__, path=1)  # goal Cherrygrove, west along Route 29
    ram[silver.X] = 34
    s.observe()
    assert s.shaping == pytest.approx(silver.COMPASS * 6 / 60)


def test_the_compass_pays_for_getting_closer_and_takes_it_back():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    start = s.shaping
    _visit(ram, s, (24, 3))  # Route 29: step 1 reached, so the goal becomes Cherrygrove
    at_29 = s.shaping
    _visit(ram, s, (24, 4))  # walks back home
    home = s.shaping
    _visit(ram, s, (24, 3), (24, 4), (24, 3))  # and wanders to and fro
    assert home < at_29 and s.shaping == pytest.approx(at_29)  # loops cancel: nothing to farm
    assert s.progress()["path"] == 1 and start == 0


def test_the_compass_points_home_with_the_egg_and_never_counts_as_progress():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    _visit(ram, s, *[(g, n) for g, n, _ in silver.PATH[:3]])
    _story_flag(ram, 30)
    _visit(ram, s, silver.PATH[3][:2])  # up to Mr. Pokemon's house: now Elm is the goal
    base = s.value() - s.shaping
    _visit(ram, s, (26, 1))  # Route 30, one map back toward home
    assert s.value() - s.shaping == pytest.approx(base, abs=0.06)  # no path pay for that, only a road tile
    assert s.path == 4 and s.shaping > 0 and s.base_value() == pytest.approx(s.value() - s.shaping)


def _story_flag(ram, i):
    ram[silver.EVENTS_START + i // 8] = ram.get(silver.EVENTS_START + i // 8, 0) | 1 << (i % 8)


def test_named_story_flags_pay_and_the_start_does_not():
    ram = _ram()
    _story_flag(ram, 30)  # already done at the start
    s = SilverStats(ram.__getitem__)
    s.observe()
    assert s.base_value() == 0.0
    _story_flag(ram, 31)
    s.observe()
    assert s.base_value() == pytest.approx(silver.STORY_PAY[31])
    ram[silver.EVENTS_START + 31 // 8] = 0  # a flag that is cleared again does not pay twice or go negative
    s.observe()
    assert s.base_value() == pytest.approx(silver.STORY_PAY[31])


def _hp(ram, hp, full=20):
    ram[silver.PARTY + silver.HP + 1], ram[silver.PARTY + silver.MAX_HP + 1] = hp, full


def test_healing_pays_a_little_up_to_a_cap_and_never_for_a_blackout_or_a_level_up():
    ram = _ram()
    _hp(ram, 20)
    s = SilverStats(ram.__getitem__)
    _hp(ram, 5)
    s.observe()
    _hp(ram, 20)
    s.observe()
    assert 0 < s.healed <= silver.HEAL_PAY
    _hp(ram, 0)
    s.observe()
    _hp(ram, 20)  # woke up at home: from 0
    s.observe()
    before = s.healed
    assert before == pytest.approx(silver.HEAL_PAY * 0.75)
    for _ in range(40):
        for hp in (4, 20):
            _hp(ram, hp)
            s.observe()
    assert s.healed == pytest.approx(silver.HEAL_MAX)
    _hp(ram, 4)
    ram[silver.BATTLE_MODE] = 1
    _hp(ram, 20)
    s.observe()
    assert s.healed == pytest.approx(silver.HEAL_MAX)


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
    assert s.base_value() == pytest.approx(0.005)  # an unnamed flag pays nothing, however often it toggles


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


def test_the_egg_steps_count_only_when_the_game_says_so():
    ram = _ram()
    s = SilverStats(ram.__getitem__)
    _visit(ram, s, *[(g, n) for g, n, _ in silver.PATH[:4]])  # inside Mr. Pokemon's house, no egg yet
    assert s.path == 3 and s.away == 0
    _story_flag(ram, 30)
    s.observe()
    assert s.path == 4  # got the egg: now Elm
    with_egg = s.shaping
    _visit(ram, s, (26, 1), (26, 3), (24, 3), (24, 4))
    assert s.away == 1 and s.shaping > with_egg
    town = s.shaping
    _visit(ram, s, (24, 5))  # inside the lab, egg not given: still the goal, and a step closer
    assert s.path == 4 and s.away == 0 and s.shaping > town + 1.0
    _visit(ram, s, (24, 4))
    assert s.path == 4 and s.shaping == pytest.approx(town)  # walking out again gives it back
    _visit(ram, s, (24, 5))
    before = s.base_value()
    _story_flag(ram, 31)
    s.observe()
    assert s.path == 5 and s.base_value() - before == pytest.approx(silver.PATH_REWARDS[4] + silver.STORY_PAY[31])


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


def test_processes_share_the_points_any_of_them_reached(tmp_path):
    a, b = silver.Frontier(tmp_path), silver.Frontier(tmp_path)
    a.reached(_Boy(), 3, 2)
    state, path = b.pick(np.random.default_rng(0), prob=1.0)
    assert state.read() == b"cherrygrove" and path == 3
    b.reached(None, 3, 2)  # known already: the emulator is not even asked
    b.reached(_Boy(), 3, 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["path_03_away_01.state", "path_03_away_02.state"]


def test_the_shared_points_are_not_written_by_games_that_may_not_start_there(tmp_path):
    f = silver.Frontier(tmp_path, share=False)
    f.reached(_Boy(), 2, 1)
    assert not list(tmp_path.iterdir())


def test_the_goal_already_reached_does_not_hide_the_way_to_it():
    f = silver.Frontier()
    f.reached(_Boy(), 4, 0)  # a state in the lab with the egg
    f.reached(_Boy(), 4, 5)  # the egg just got, far from home: worth practising too
    assert set(f.states) == {(4, 0), (4, 5)}


def test_every_path_step_gets_the_same_share_of_practice():
    f = silver.Frontier()
    assert f.pick(np.random.default_rng(0), prob=1.0) is None  # nothing reached yet: normal start
    f.reached(_Boy(), 0, 3)  # home is the normal start
    f.reached(_Boy(), 2, None)  # no idea how far: skip
    assert not f.states
    for path, away in ((2, 2), (4, 5), (4, 4), (4, 3)):
        f.reached(_Boy(), path, away)
    f.reached(None, 4, 5)  # known already
    rng = np.random.default_rng(0)
    paths = [f.pick(rng, prob=1.0)[1] for _ in range(400)]
    assert 0.42 < paths.count(2) / 400 < 0.58  # one point against three, but one step against one
    assert f.pick(rng, prob=0.0) is None


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


def _flag(ram, i):
    ram[achievements.EVENTS_START + i // 8] = ram.get(achievements.EVENTS_START + i // 8, 0) | 1 << (i % 8)


def test_the_game_names_what_has_been_achieved():
    ram = _Ram()
    assert achievements.achieved(ram.__getitem__) == {}
    _flag(ram, 30)
    ram[achievements.JOHTO_BADGES] = 0b1
    assert achievements.achieved(ram.__getitem__) == {"event_30": "got the Mystery Egg from Mr. Pokemon",
                                                     "badge_Zephyr": "got the Zephyr Badge"}


def test_each_achievement_is_logged_once_for_all_games_and_the_start_does_not_count(tmp_path):
    ram = _Ram()
    _flag(ram, 26)  # the start state already has it
    a, b = (achievements.AchievementLog(tmp_path, ram.__getitem__) for _ in range(2))
    assert a.update(0, 10) == []
    _flag(ram, 30)
    assert a.update(4, 900) == ["got the Mystery Egg from Mr. Pokemon"]
    assert b.update(4, 950) == []  # another process did it second
    import json
    saved = json.loads((tmp_path / "event_30.json").read_text())
    assert saved["path_step"] == 4 and saved["game_steps"] == 900
