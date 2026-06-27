# Edge SLM sandbox — convenience targets.
# Usage: make help

SHELL := /bin/bash
DC := docker compose
BASE := -f docker/compose.yaml
CPU  := $(BASE) -f docker/compose.cpu.yaml
CUDA := $(BASE) -f docker/compose.cuda.yaml
ROCM := $(BASE) -f docker/compose.rocm.yaml

# Load docker/.env if present (falls back to .env.example defaults in compose).
ifneq (,$(wildcard docker/.env))
  include docker/.env
  export
endif
export COMPOSE_FILE :=
MODE_FILES ?= $(CPU)          # default mode is CPU; `make up-cuda` flips it

MODEL ?= qwen2.5:3b
LABEL ?= clean
CAP_GB ?= 8

.PHONY: help
help:                ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*## .*$$' $(firstword $(MAKEFILE_LIST)) \
	  | awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-18s\033[0m %s\n",$$1,$$2}'

.PHONY: env
env:                 ## Create docker/.env from the example if missing
	@test -f docker/.env || (cp docker/.env.example docker/.env && echo "created docker/.env")

.PHONY: up-cpu
up-cpu: env          ## Start Ollama in CPU mode (16 GB cap, no GPU)
	$(DC) $(CPU) up -d --build ollama
	@echo "Ollama up (CPU). Next: make pull MODEL=$(MODEL) && make bench LABEL=$(LABEL)"

.PHONY: up-cuda
up-cuda: env         ## Start Ollama in CUDA mode (real VRAM; needs NVIDIA host)
	$(DC) $(CUDA) up -d --build ollama
	@echo "Ollama up (CUDA). Next: make pull MODEL=$(MODEL) && make bench-cuda LABEL=$(LABEL)"

.PHONY: up-rocm
up-rocm: env         ## Start Ollama in ROCm mode (real VRAM; needs AMD Linux host)
	$(DC) $(ROCM) up -d ollama
	@echo "Ollama up (ROCm). Next: make pull MODEL=$(MODEL) && make bench-rocm LABEL=$(LABEL)"

.PHONY: pull
pull:                ## Pull a model into the running Ollama (MODEL=qwen2.5:3b)
	docker exec edge-slm-ollama ollama pull $(MODEL)

.PHONY: bench
bench:               ## Run the harness, CPU mode (LABEL=clean|multitask)
	$(DC) $(CPU) run --rm bench --label $(LABEL)

.PHONY: bench-cuda
bench-cuda:          ## Run the harness, CUDA mode (real VRAM)
	$(DC) $(CUDA) run --rm bench --label $(LABEL)

.PHONY: bench-rocm
bench-rocm:          ## Run the harness, ROCm mode (real AMD VRAM)
	$(DC) $(ROCM) run --rm bench

.PHONY: bench-metal
bench-metal:         ## Run the harness NATIVELY on Apple GPU (Metal, no Docker)
	bash native/run_metal_bench.sh $(LABEL)

.PHONY: bench-metal-capped
bench-metal-capped:  ## Metal bench under a simulated VRAM cap (CAP_GB=8; needs sudo)
	bash native/bench_metal_capped.sh $(CAP_GB)

.PHONY: ps
ps:                  ## Show resident models + container stats
	-docker exec edge-slm-ollama ollama ps
	-docker stats --no-stream edge-slm-ollama

.PHONY: logs
logs:                ## Tail Ollama logs
	$(DC) $(CPU) logs -f ollama

.PHONY: down
down:                ## Stop containers (keeps the model volume)
	$(DC) $(CPU) down

.PHONY: clean
clean:               ## Stop and DELETE the pulled-model volume
	$(DC) $(CPU) down -v
