# gameboy-rl — PPO on Pokémon Red

A thin wrapper around **[PWhiddy/PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments)** (MIT,
© Peter Whidden — see `LICENSE`). PWhiddy's v2 environment, reward and PPO settings are used unmodified from `v2/`;
this repo adds a packaged trainer (`src/gameboy_rl/`), training-health monitoring, MLflow tracking and a
self-terminating RunPod job. The original README follows [below](#upstream-readme-pwhiddy); the v1 code lives in
`_upstream/` for reference only.

## What changed vs. upstream

- **Start state**: `has_pokedex_nballs.state` instead of `init.state`. The upstream run starts in Red's bedroom and has
  to play the whole scripted opening (Oak stops you, starter, rival battle, parcel errand, Pokédex) before free
  exploration; this state is in Oak's lab with a starter, the Pokédex and Poké Balls. Override: `GBRL_INIT_STATE`.
- **StreamWrapper is off** (it broadcasts coordinates to PWhiddy's public map). Opt in with `GBRL_STREAM=true`.
- ROM path: `roms/pokemon_red_version.gb` (gitignored, never committed, never in the image).

## Setup

```bash
uv sync --extra dev                       # Python 3.12, CPU torch
cp /path/to/your/pokemon_red_version.gb roms/   # legally obtained; SHA-1 ea9bcae617fdf159b045185467ae58b2e4a48b9a
make test                                 # PyBoy tests skip without the ROM
```

## Check the start state

```bash
make check-start      # boots GBRL_INIT_STATE headless, 200 random actions -> runs/check_start.png
```

Prints `OVERWORLD` or `DIALOG` (Red's text box = white box over the bottom of the screen) for the final frame, plus
the fraction of frames that showed a text box. Look at the PNG.

## Smoke run

```bash
make smoke            # 2 envs, 2048 steps, one full (short) episode, temp dir; prints progress + status
```

## Train

```bash
uv run gameboy-rl train --n-envs 2 --steps 100000                          # fresh policy
uv run gameboy-rl train --resume-from v2/runs/poke_26214400.zip --steps 30000000  # from PWhiddy's 26M-step policy
```

Runs land in `runs/<run_id>/`: `checkpoints/`, `health.json`, `progress.json`, `tb/` (TensorBoard), `DONE`
(`done|critical|budget`). Steps are absolute: resuming the 26.2M-step checkpoint counts from 26,214,400.

## RunPod

See [`runpod/README.md`](runpod/README.md). In short: upload the ROM (per run) and the init checkpoint (once) to the
private bucket, launch a pod with `RUN_ID`/`GCS_BUCKET`/MLflow creds; the pod resumes, pulls and deletes the ROM,
trains under `GBRL_MAX_HOURS` / `GBRL_MAX_EUR`, and removes itself on any exit.

## Health

`health.json` (and MLflow tags `health_status` / `health_reasons`) is `HEALTHY`, `WARN` or `CRITICAL`, same rules as
sega-rl: NaN, approx_kl, clip_fraction, entropy collapse, explained variance, reward plateau, fps drop / zero,
button spam (one button > 90%), stall watchdog (exit 3), env crash. CRITICAL stops the run.
Pokémon progress per rollout: `pokemon/badges_*`, `pokemon/events_*`, `pokemon/explore_tiles_*`,
`pokemon/map_progress_max`. Live: `make status RUN=<run_id>` (MLflow experiment `gameboy-rl`).

## Run ledger

| Date | RUN_ID | Start state | From | Steps | Hours / EUR | Final status | Badges / events | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | first paid run pending |

---

## Upstream README (PWhiddy)

### Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

### Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
### Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10+ is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

#### Windows Setup
Refer to this [Windows Setup Guide](_upstream/windows-setup-guide.md)

#### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

#### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

### Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


#### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

### Tracking Training Progress 📈

#### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/v2/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

#### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

### Static Visualization 🐜
Map visualization code can be found in `_upstream/visualization/`.

### Follow up work  
 
Check out our follow up projects & papers!  
  
#### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
#### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
#### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

### Supporting Libraries
Check out these awesome projects!
#### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

#### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>
