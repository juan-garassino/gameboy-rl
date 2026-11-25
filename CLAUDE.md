# CLAUDE.md — RL-gameboy-rl

PPO (SB3) on Pokémon Red with **PWhiddy's v2 env, imported unmodified** from `v2/`. This folder is a clone of
PWhiddy/PokemonRedExperiments: `origin` is PWhiddy's GitHub — **never push to origin**. Wrap, don't rewrite: do not
change reward / env logic in `v2/`. Trained on RunPod (archetype B), monitored via MLflow (experiment `gameboy-rl`).
Template for all monitoring + RunPod plumbing: `../RL-sega-rl` (keep the two copy-compatible).

## Module map (`src/gameboy_rl/`)

| Module | Purpose |
|---|---|
| `config.py` | `Settings` (`GBRL_*`), `REPO_ROOT`/`V2_DIR`, relative paths resolve against the repo root, `hours_budget` (EUR cap ÷ pod USD/h), `n_steps = ep_length // 64`, even `n_envs` |
| `env.py` | `make_red_env` (RedGymEnv built inside `v2/`), `ButtonsWrapper` (`info["buttons"]`), `AgentStatsTail`, opt-in `StreamWrapper`, `make_vec_env` (SubprocVecEnv), `import_upstream` |
| `monitoring/health.py` | sega-rl's `HealthMonitor` rules + `HealthCallback` (health.json, watchdog → MLflow "stalled" → exit 3, `heartbeat()`) |
| `monitoring/mlflow_tracking.py` | no-op unless `MLFLOW_TRACKING_URI`; `start(cfg, probe)` is fail-fast; artifacts at `mlflow-artifacts:/gameboy-rl`; SB3 KVWriter |
| `monitoring/progress.py` | `ProgressCallback`: badges, event flags, explore-map tiles, map progress → `pokemon/*` each rollout |
| `train.py` | `run()`: baseline_fast_v2 PPO params, resume (run ckpt > `GBRL_RESUME_FROM`), checkpoints → MLflow, health, budget, PWhiddy's `TensorboardCallback`, any crash → CRITICAL, DONE in `finally` |
| `check_start.py` | `text_box_visible(frame)` heuristic, `run()` → `runs/check_start{,_first}.png` |
| `status.py` | `gameboy-rl status <run>`: live state from MLflow |
| `silver.py` | Pokémon Silver in color: `SilverEnv`, `SilverStats` (RAM from pret symbols, high-water marks so doing nothing earns 0), `PATH` (story road to the first badge, each step pays more; its outdoor tiles pay 10x; `NEEDS_FLAG`: the two egg steps count only when the game's flag is set), `areas` observation, a compass (`road_position`, tile by tile: +3 per map closer to the next goal, taken back when moving away; cancels over loops, never counts as progress; road tiles pay only on maps as near the goal as this step has got), `StallClock` (more patience further along), `Frontier` (half the games start at a point already reached: each path step equally often, one state per step and distance; a practice game's patience counts only the steps it made itself; evals never do), baselines |
| `silver_eval.py` | `SilverEvalCallback`: baselines first (refuses to train if doing nothing pays), then color video + outcome stats |
| `play.py` | `gameboy-rl play --model X --video out.mp4`: local headless replay of a checkpoint (the trainer records no eval videos) |
| `cli.py` | `gameboy-rl train|smoke|check-start|status` (argparse) |

`runpod/`: `Dockerfile` (CUDA 12.8 devel, uv, cu126 torch 2.14.1 swapped in, no ROM/ckpt), `entrypoint.sh` (see
`runpod/README.md`), `scripts/pod_self.sh` (cost / self-terminate with retries), `scripts/token_writer.sh`.
Upstream kept as-is: `v2/` (training path), `*.state`, `_upstream/` (v1 baselines, clip experiment, visualization).

## Gotchas

- **Start state**: default `has_pokedex_nballs.state` (Oak's lab, starter + Pokédex + balls). `init.state` = Red's
  bedroom before the scripted opening — the "too much dialog" start. Random actions next to Oak still open optional
  text boxes (~1/3 of frames in `check-start`); that's interaction, not a forced cutscene.
- **StreamWrapper off by default** (`GBRL_STREAM`): it sends coordinates to `wss://transdimensional.xyz/broadcast`.
- `red_gym_env_v2` opens `events.json` from the cwd → `env.py` constructs it inside `v2/` (`_in_v2_dir`).
- **PWhiddy's `poke_26214400.zip` was pickled on Python 3.11**: its `clip_range` lambda segfaults on 3.12.
  `load_model` passes `SCHEDULE_OVERRIDES` (lr 3e-4, clip 0.2 = SB3 defaults baseline_fast_v2 used). Obs space matches.
- `n_envs` must be even (upstream `TensorboardCallback` lays explore maps out in 2 rows at each episode end).
- `TensorboardCallback` fires one step before each episode end (`ep_length` = 163,840 steps/env); `AgentStatsTail`
  keeps it from pickling every per-step stats dict. Workers still hold their own full `agent_stats` list (upstream).
- Torch: pyproject pins CPU `torch==2.14.1`; the image installs the same version from cu126. Keep them equal.
- GCS token on the pod lasts 1 h; anything durable goes to MLflow (checkpoints every `ckpt_freq`, health, DONE).
- ROMs: never committed (`tests/test_no_roms.py`), never in the image (`.dockerignore`), deleted from GCS by the pod.
- Local heavy jobs one at a time (laptop froze 2026-11-09): `nice`, ≤ 2 envs, never `docker build`.

## Env vars

All `Settings` fields map to `GBRL_<FIELD>` (inline comments in `config.py`). Pod-only: `MODE`, `RUN_ID`,
`GCS_BUCKET`, `GCS_ACCESS_TOKEN`, `GCS_TIMEOUT`, `INIT_CKPT`, `ROM_SHA1`, `RUNPOD_API_KEY`/`GBRL_RUNPOD_API_KEY`,
`RUNPOD_POD_ID` (injected), `SYNC_INTERVAL`, `MLFLOW_*` (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`, `…_TIMEOUT=20` set by
the entrypoint).

## Silver

`make launch-silver` uploads the ROM and `roms/claude_totodile.state` (player CLAUDE, Totodile, outside Elm's lab; never in git)
and deploys `mode=train-silver`. Judge it by badges / events / maps against `baseline/*`, never by reward or loss.
`eval/path` is how far along `PATH` a game gets; evals and baselines average `GBRL_EVAL_EPISODES` games. silver-claude-3 (no path) toured New Bark's houses.
Episodes end after `GBRL_STALL_STEPS` steps without progress: silver-claude-2 (none) learned to talk to Elm forever.
