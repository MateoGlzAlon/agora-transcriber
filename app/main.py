import os
import time
from transcription import transcribe
from rag import load_context, query_context
from llm import enhance

AUDIO_DIR = "/data/audio"
OUTPUT_DIR = "/data/output"

SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".opus"}

# Use the first N characters of the transcript to query for relevant context
CONTEXT_QUERY_CHARS = 2000


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def main():
    pipeline_start = time.time()

    print("=" * 50)
    print("  AEGEE Agora Transcriber")
    print("=" * 50)
    print()

    load_context()

    audio_files = [
        f for f in sorted(os.listdir(AUDIO_DIR))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"Found {len(audio_files)} audio file(s) to process.\n")

    for idx, filename in enumerate(audio_files, 1):
        file_start = time.time()
        audio_path = os.path.join(AUDIO_DIR, filename)
        stem = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, stem + ".txt")

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"[{idx}/{len(audio_files)}] {filename}  ({file_size_mb:.1f} MB)")
        print("-" * 40)

        t0 = time.time()
        print("Step 1/3 — Transcribing audio with Whisper...")
        raw_transcript = transcribe(audio_path)
        lines = raw_transcript.count("\n") + 1
        print(f"  Done in {fmt_duration(time.time() - t0)} — {lines} segments, {len(raw_transcript):,} chars")

        t0 = time.time()
        print("Step 2/3 — Retrieving relevant context from historical minutes...")
        context = query_context(raw_transcript[:CONTEXT_QUERY_CHARS])
        print(f"  Done in {fmt_duration(time.time() - t0)}")

        t0 = time.time()
        print("Step 3/3 — Enhancing transcription with local LLM...")
        final_transcript = enhance(raw_transcript, context)
        print(f"  Done in {fmt_duration(time.time() - t0)} — {len(final_transcript):,} chars")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcription: {filename}\n\n")
            f.write(final_transcript)

        elapsed = fmt_duration(time.time() - file_start)
        print(f"Saved → {out_path}  (file total: {elapsed})\n")

    total = fmt_duration(time.time() - pipeline_start)
    print("=" * 50)
    print(f"  All files processed in {total}.")
    print("=" * 50)


if __name__ == "__main__":
    main()
