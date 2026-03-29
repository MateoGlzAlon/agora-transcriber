PROJECT = agora-transcriber
COMPOSE  = docker compose

.PHONY: help build up down logs pull-model run clean reset purge everything

help:
	@echo "Available commands:"
	@echo "  make build        Build Docker containers"
	@echo "  make up           Start the Ollama service in the background"
	@echo "  make pull-model   Manually pull the LLM model into Ollama"
	@echo "  make run          Run the transcription pipeline"
	@echo "  make logs         Follow all service logs"
	@echo "  make down         Stop all services"
	@echo "  make clean        Remove generated output files"
	@echo "  make reset        Full cleanup (containers + volumes + outputs)"
	@echo "  make purge        Remove volumes and images for this project"
	@echo "  make everything   Full workflow: reset → build → run"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d ollama

pull-model:
	docker exec agora-ollama ollama pull $${OLLAMA_MODEL:-llama3}

run:
	$(COMPOSE) up app

logs:
	$(COMPOSE) logs -f

down:
	$(COMPOSE) down

clean:
	rm -f data/output/*.txt

reset: down clean
	$(COMPOSE) down -v

purge: down
	$(COMPOSE) down -v --rmi all

# Full one-shot workflow: clean slate → build → start Ollama → run app
# The app entrypoint waits for Ollama and pulls the model automatically.
everything: reset build run
