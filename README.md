# AEGEE Agora Transcriber

A fully local, API-free audio transcription pipeline designed for AEGEE Agora meeting recordings.

It uses historical Agora minutes as context to improve accuracy on AEGEE-specific terminology — antenna names, delegate names, acronyms, and procedural language.

## How it works

```
Audio file(s)
    │
    ▼
Whisper (local STT) ──────────────────────── timestamps + raw text
    │
    ▼
ChromaDB (RAG) ◄── Historical minutes PDFs ─ retrieves relevant context
    │
    ▼
Ollama LLM (local) ─────────────────────── corrects names, terms, structure
    │
    ▼
data/output/<filename>.txt
```

1. **Whisper** transcribes the audio locally. Each output line includes a timestamp: `[HH:MM:SS --> HH:MM:SS] text`.
2. **RAG** indexes the historical minutes (PDFs in `data/context/`) into a local vector database and retrieves the most relevant passages.
3. **Ollama** runs a local LLM that uses the retrieved context to fix proper nouns, AEGEE acronyms, and transcription errors — without inventing anything.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)

No other dependencies — everything runs inside containers.

## Setup

### 1. Add your audio files

Place your Agora recording(s) in:

```
data/audio/
```

Supported formats: `.m4a`, `.mp3`, `.wav`, `.ogg`, `.flac`, `.mp4`, `.webm`, `.opus`

### 2. Add historical minutes (context)

Place previous Agora minutes (PDF or TXT) in:

```
data/context/
```

The more minutes you add, the better the correction quality. The repository already includes three example documents.

### 3. Configure (optional)

Edit `.env` to adjust models:

```env
OLLAMA_MODEL=llama3    # LLM for correction (llama3, mistral, phi3, etc.)
WHISPER_MODEL=medium   # Whisper model (tiny/base/small/medium/large-v3)
```

`medium` is the recommended Whisper model. Use `large-v3` for maximum accuracy (requires more memory and time).

## Usage

```bash
# Full pipeline (recommended for first run)
make everything
```

This will:
1. Stop and clean any previous state
2. Build the Docker images
3. Start Ollama (waits for it to be healthy)
4. Pull the LLM model (only on first run — cached afterwards)
5. Run the transcription pipeline

### Individual commands

```bash
make build        # Build containers
make up           # Start Ollama in the background
make run          # Run the transcription app
make logs         # Follow logs
make down         # Stop all services
make clean        # Delete output files
make reset        # Full cleanup including Docker volumes
```

### Output

Results are written to `data/output/<filename>.txt`.

## Models

### Whisper (transcription)

| Model     | Size   | Speed (CPU) | Accuracy |
|-----------|--------|-------------|----------|
| tiny      | 75 MB  | Very fast   | Low      |
| base      | 145 MB | Fast        | Low      |
| small     | 465 MB | Moderate    | Good     |
| medium    | 1.5 GB | Slow        | **Recommended** |
| large-v3  | 3 GB   | Very slow   | Best     |

### LLM (correction via Ollama)

Any model available through Ollama works. Default is `llama3`. To use a different model, change `OLLAMA_MODEL` in `.env` — the entrypoint will pull it automatically on startup.

## Speaker diarization

Automatic speaker identification (who said what) is **not enabled** by default because it requires `pyannote.audio` which needs a HuggingFace access token.

The output does include **per-segment timestamps** from Whisper, which allows manual speaker labelling.

To enable diarization in the future, see the comments in `app/diarization.py`.

## Project structure

```
.
├── app/
│   ├── main.py           # Entry point and pipeline orchestration
│   ├── transcription.py  # Whisper-based audio transcription
│   ├── rag.py            # PDF ingestion, chunking, and vector search
│   ├── llm.py            # Ollama LLM correction with chunked processing
│   ├── diarization.py    # Speaker diarization stub (disabled)
│   ├── entrypoint.sh     # Waits for Ollama, pulls model, runs pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── audio/            # Input audio files (add yours here)
│   ├── context/          # Historical Agora minutes (PDF/TXT)
│   └── output/           # Generated transcriptions (created automatically)
├── docker-compose.yml
├── Makefile
└── .env
```
