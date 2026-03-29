PROJECT = agora-transcriber
COMPOSE = docker compose

# Detect OS
ifeq ($(OS),Windows_NT)
    RM = del /Q
    MKDIR = mkdir
    SEP = \\
else
    RM = rm -f
    MKDIR = mkdir -p
    SEP = /
endif

OUTPUT_DIR = data$(SEP)output

.PHONY: help build up down logs pull-model run clean reset everything

help:
	@echo Available commands:
	@echo   make build        Build Docker containers
	@echo   make up           Start the Ollama service in the background
	@echo   make pull-model   Manually pull the LLM model into Ollama
	@echo   make run          Run the transcription pipeline
	@echo   make logs         Follow all service logs
	@echo   make down         Stop all services
	@echo   make clean        Remove generated output files
	@echo   make reset        Full cleanup (containers + volumes + outputs)
	@echo   make everything   Full workflow: reset -> build -> run

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d ollama

pull-model:
	docker exec agora-ollama ollama pull llama3

run:
	$(COMPOSE) up app

logs:
	$(COMPOSE) logs -f

down:
	$(COMPOSE) down

clean:
	-$(RM) $(OUTPUT_DIR)$(SEP)*.txt

reset: down clean
	$(COMPOSE) down -v

everything: reset build run