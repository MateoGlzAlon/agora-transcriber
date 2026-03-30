PROJECT = agora-transcriber
COMPOSE  = docker compose

.PHONY: help build extract segment transcribe enhance all run status \
        logs up down clean clean-status clean-raw reset purge everything

help:
	@echo ""
	@echo "  AEGEE Agora Transcriber — Pipeline Stages"
	@echo "  =========================================="
	@echo ""
	@echo "  Run each stage on a different day to avoid long sessions:"
	@echo ""
	@echo "    make extract      Convert video files to audio (fast, no GPU/LLM)"
	@echo "    make segment      Split audio by segment definitions (fast, no GPU/LLM)"
	@echo "    make transcribe   Transcribe audio with Whisper (slow — heavy CPU)"
	@echo "    make enhance      Enhance transcripts with LLM via Ollama (slow — LLM)"
	@echo "    make all          Run all stages in order (skips already-done files)"
	@echo ""
	@echo "  Status & monitoring:"
	@echo ""
	@echo "    make status       Show which files have been processed in each stage"
	@echo "    make logs         Follow all service logs"
	@echo ""
	@echo "  Lifecycle:"
	@echo ""
	@echo "    make build        Build Docker containers"
	@echo "    make up           Start Ollama in the background"
	@echo "    make down         Stop all services"
	@echo "    make run          Run the full pipeline in one shot"
	@echo ""
	@echo "  Cleanup:"
	@echo ""
	@echo "    make clean        Remove final output files (data/output/)"
	@echo "    make clean-status Remove stage status markers (re-run stages from scratch)"
	@echo "    make clean-raw    Remove raw transcripts (data/raw/)"
	@echo "    make reset        Full cleanup: stop containers + remove all generated files"
	@echo "    make purge        Reset + remove Docker volumes and images"
	@echo "    make everything   Full one-shot: reset → build → run all stages"
	@echo ""

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d ollama

# ── Pipeline stages ──────────────────────────────────────────────────────────
# extract, segment, transcribe: no Ollama needed (--no-deps skips starting it)
# enhance: starts Ollama automatically via depends_on in docker-compose.yml

extract:
	$(COMPOSE) run --rm --no-deps app extract

segment:
	$(COMPOSE) run --rm --no-deps app segment

transcribe:
	$(COMPOSE) run --rm --no-deps app transcribe

enhance:
	$(COMPOSE) run --rm app enhance

# Run all pipeline stages in order (skips already-completed files)
all: extract segment transcribe enhance

# Full pipeline in one shot (starts Ollama, runs all stages)
run:
	$(COMPOSE) run --rm app

# ── Status ───────────────────────────────────────────────────────────────────

status:
	@echo ""
	@echo "  Pipeline Status"
	@echo "  ==============="
	@if [ -d data/status ] && [ "$$(ls data/status/ 2>/dev/null | wc -l)" -gt 0 ]; then \
		echo ""; \
		for f in data/status/*; do \
			echo "  ✓ $$(basename $$f)"; \
		done; \
		echo ""; \
		echo "  Marker files are in data/status/ — 'cat data/status/<file>' for details."; \
	else \
		echo "  (no stages completed yet)"; \
	fi
	@echo ""

# ── Utilities ────────────────────────────────────────────────────────────────

logs:
	$(COMPOSE) logs -f

down:
	$(COMPOSE) down

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	rm -f data/output/*.txt

clean-status:
	rm -rf data/status

clean-raw:
	rm -rf data/raw

reset: down clean clean-status clean-raw
	$(COMPOSE) down -v

purge: down
	$(COMPOSE) down -v --rmi all

# Full one-shot workflow: clean slate → build → run all stages
everything: reset build run
