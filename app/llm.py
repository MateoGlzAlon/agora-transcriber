import os
import time
import requests
from tqdm import tqdm

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Maximum characters per LLM call — prevents overflowing the model context window
MAX_CHUNK_CHARS = 4000


def _build_prompt(transcript_chunk: str, context: str) -> str:
    return f"""You are an expert transcription editor for AEGEE (Association des États Généraux des Étudiants de l'Europe) Agora meetings.

Your task is to correct and clean a raw Whisper audio transcription using historical context from previous Agora minutes.

HISTORICAL CONTEXT (excerpts from previous Agora minutes):
{context}

RAW TRANSCRIPTION SEGMENT:
{transcript_chunk}

INSTRUCTIONS:
- Correct proper nouns: AEGEE antenna names, European city names, delegate names, and AEGEE-specific terms
- Fix obvious Whisper errors using the context above as reference (e.g. mis-heard antenna names, acronyms like CD, SUCT, JC, EBM, CIA, etc.)
- Do NOT invent, add, or remove any information — only correct what is clearly wrong
- Keep timestamps exactly as they appear: [HH:MM:SS --> HH:MM:SS]
- Output clean, readable meeting-minutes style text
- Preserve the full content — do NOT summarise

CORRECTED OUTPUT:
"""


def _call_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split transcript into chunks at line boundaries near max_chars."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line) + 1  # +1 for newline

    if current:
        chunks.append("\n".join(current))

    return chunks


def enhance(transcript: str, context: str) -> str:
    """
    Use the local LLM (via Ollama) to correct the raw transcript.

    Long transcripts are processed in chunks to avoid hitting the model's
    context-window limit.
    """
    chunks = _split_into_chunks(transcript)
    results: list[str] = []

    bar = tqdm(chunks, desc="  LLM chunks", unit="chunk", ncols=70)
    for i, chunk in enumerate(bar, 1):
        bar.set_postfix_str(f"{len(chunk):,} chars")
        prompt = _build_prompt(chunk, context)
        t0 = time.time()
        try:
            corrected = _call_ollama(prompt)
            elapsed = time.time() - t0
            bar.write(f"  Chunk {i}/{len(chunks)} done in {elapsed:.1f}s — {len(corrected):,} chars out")
            results.append(corrected)
        except Exception as exc:
            bar.write(f"  Warning: chunk {i} failed ({time.time() - t0:.1f}s): {exc}")
            results.append(chunk)  # fall back to raw chunk

    return "\n\n".join(results)
