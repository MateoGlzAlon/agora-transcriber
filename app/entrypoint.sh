#!/bin/bash
set -e

STAGE="${1:-}"
MODEL="${OLLAMA_MODEL:-llama3}"

# Only wait for Ollama when running the enhance stage or the full pipeline.
# Extract, segment, and transcribe don't need Ollama.
if [ -z "$STAGE" ] || [ "$STAGE" = "enhance" ]; then
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
fi

exec python main.py "$@"
