#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-llama3}"

echo "Waiting for Ollama to be ready..."
until curl -sf "http://ollama:11434/api/version" > /dev/null 2>&1; do
    echo "  Ollama not ready yet — retrying in 3s..."
    sleep 3
done
echo "Ollama is up."

echo "Pulling model: $MODEL (skipped if already cached)..."
curl -s -X POST "http://ollama:11434/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$MODEL\"}" \
    | grep -E '"status"' | tail -3 || true

echo "Model ready."
echo ""

exec python main.py
