# RunPod runbook — gameboy-rl

Build the image in CI (none in this repo yet — copy sega-rl's `build-push.yml` / `deploy-runpod.yml` when wiring it);
never `docker build` on the laptop. The image contains no ROM and no checkpoint.

## Before a launch (per run)

```bash
# private bucket only — never upload a ROM to a public bucket
gcloud storage cp roms/pokemon_red_version.gb gs://garassino-ml-artifacts/gameboy-rl/roms/
# once (kept): the pretrained PWhiddy policy the run resumes from
gcloud storage cp v2/runs/poke_26214400.zip gs://garassino-ml-artifacts/gameboy-rl/init/
```

Pod env: `RUN_ID`, `GCS_BUCKET=garassino-ml-artifacts`, `GCS_ACCESS_TOKEN` (1 h, WIF-minted), `RUNPOD_API_KEY` (or
`GBRL_RUNPOD_API_KEY`), `MLFLOW_TRACKING_URI` + `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, and any
`GBRL_*` override (`GBRL_N_ENVS` ≈ vCPUs and **even**, `GBRL_MAX_HOURS`, `GBRL_MAX_EUR`, `GBRL_TOTAL_STEPS`).
Container disk only (no network volume). Emulation is CPU-bound: vCPU count matters more than the GPU.

## What the pod does (`entrypoint.sh`)

1. Exits (and removes itself) if `gs://…/gameboy-rl/runs/<RUN_ID>/DONE` exists.
2. Pulls the run dir (resume from the run's newest checkpoint).
3. No run checkpoint yet → pulls `init/$INIT_CKPT` (default `poke_26214400.zip`) and exports `GBRL_RESUME_FROM`.
   Missing → `health.json` CRITICAL `init_ckpt`, pod removed (the ROM in GCS is untouched). `INIT_CKPT=` = fresh policy.
4. Pulls `roms/pokemon_red_version.gb`, then **deletes every object under `gs://…/gameboy-rl/roms/`**, then checks
   the SHA-1 (`ea9bcae6…`). No ROM in GCS and none on disk, or a wrong hash → CRITICAL `rom`, pod removed.
   A pod restart after the delete therefore fails with `rom` — re-upload and relaunch.
5. Reads its own USD/h (`pod_self.sh cost`) → `GBRL_COST_PER_HR_USD`; training stops at
   min(`GBRL_MAX_HOURS`, `GBRL_MAX_EUR` worth of hours).
6. Syncs the run dir to GCS every `SYNC_INTERVAL` (600 s) while the token lasts; failures never stop training.
7. Runs `gameboy-rl $MODE`; on any exit: final sync, then `pod_self.sh remove` (billing stops).

## Watch a run

```bash
make status RUN=<RUN_ID>   # from MLflow: health + reasons, badges, events, map tiles, fps
```

MLflow experiment `gameboy-rl`, run name = RUN_ID: every SB3 metric (`train/*`, `rollout/*`, `pokemon/*`,
`env_stats/*`, `health/status`), `health.json` (≤ every 10 min), and at the end `progress.json`, `DONE` and the
final checkpoint. The GCS token dies after 1 h, so MLflow is the durable channel.
