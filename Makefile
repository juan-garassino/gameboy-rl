.PHONY: help install test smoke check-start status upload-inputs launch clean

GCS_ROOT := gs://garassino-ml-artifacts/gameboy-rl
ROM := roms/pokemon_red_version.gb

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

install:  ## uv sync with dev deps (CPU torch)
	uv sync --extra dev

test:  ## Run tests (PyBoy tests skip without the ROM), niced
	nice -n 19 env OMP_NUM_THREADS=2 uv run pytest -q

smoke:  ## 2-env run, 2048 steps, one short episode, temp dir, niced
	nice -n 19 env OMP_NUM_THREADS=2 uv run gameboy-rl smoke

check-start:  ## Boot GBRL_INIT_STATE, 200 random actions -> runs/check_start.png + OVERWORLD/DIALOG
	nice -n 19 uv run gameboy-rl check-start

status:  ## Live run state from MLflow: make status RUN=20261124-180000
	@if [ -z "$(RUN)" ]; then echo "Usage: make status RUN=<run id>"; exit 1; fi
	@uv run gameboy-rl status $(RUN) 2>/dev/null

upload-inputs:  ## Per launch: ROM (+ init checkpoint) to the PRIVATE bucket; the pod deletes the ROM after pulling
	@policy=$$(gcloud storage buckets get-iam-policy gs://garassino-ml-artifacts --format=json) || { \
		echo "cannot read bucket policy — refusing to upload ROMs"; exit 1; }; \
	if echo "$$policy" | grep -qE 'allUsers|allAuthenticatedUsers'; then \
		echo "bucket is public — refusing to upload ROMs"; exit 1; fi
	gcloud storage cp $(ROM) $(GCS_ROOT)/roms/
	gcloud storage cp -n v2/runs/poke_26214400.zip $(GCS_ROOT)/init/

launch: upload-inputs  ## Upload ROM + init checkpoint, then start a pod [ARGS="-f n_envs=32"]
	gh workflow run deploy-runpod.yml $(ARGS)

clean:  ## Remove caches and local runs
	rm -rf .pytest_cache runs **/__pycache__
