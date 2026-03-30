import os
import sys
import time
import argparse
from datetime import datetime

from video import extract_audio, VIDEO_EXTENSIONS
from segmentation import parse_segments_file, split_audio

VIDEO_DIR = "/data/video"
AUDIO_DIR = "/data/audio"
SEGMENTS_DIR = "/data/segments"
RAW_DIR = "/data/raw"
OUTPUT_DIR = "/data/output"
STATUS_DIR = "/data/status"

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".webm", ".opus"}

# Use the first N characters of the transcript to query for relevant context
CONTEXT_QUERY_CHARS = 2000


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def is_done(stem: str, stage: str) -> bool:
    return os.path.exists(os.path.join(STATUS_DIR, f"{stem}.{stage}"))


def mark_done(stem: str, stage: str, info: str = "") -> None:
    os.makedirs(STATUS_DIR, exist_ok=True)
    path = os.path.join(STATUS_DIR, f"{stem}.{stage}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Completed: {datetime.now().isoformat()}\n")
        if info:
            f.write(info + "\n")


def _list_audio_files() -> list[str]:
    """Return audio files directly in AUDIO_DIR (not segment chunk subdirs)."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    return [
        os.path.join(AUDIO_DIR, af)
        for af in sorted(os.listdir(AUDIO_DIR))
        if os.path.splitext(af)[1].lower() in AUDIO_EXTENSIONS
    ]


# ─── Stage: extract ──────────────────────────────────────────────────────────

def run_extract() -> None:
    print("=" * 50)
    print("  Stage: Extract (video → audio)")
    print("=" * 50)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if not video_files:
        print(f"No video files found in {VIDEO_DIR}")
        return

    for vf in video_files:
        stem = os.path.splitext(vf)[0]
        if is_done(stem, "extracted"):
            print(f"  SKIP  {vf}  (already extracted)")
            continue
        video_path = os.path.join(VIDEO_DIR, vf)
        print(f"  Extracting {vf}…", end=" ", flush=True)
        t0 = time.time()
        wav_path = extract_audio(video_path, AUDIO_DIR)
        elapsed = fmt_duration(time.time() - t0)
        print(f"done in {elapsed}  →  {wav_path}")
        mark_done(stem, "extracted", f"source: {vf}\noutput: {wav_path}\nduration: {elapsed}")

    print("\nExtraction complete.\n")


# ─── Stage: segment ───────────────────────────────────────────────────────────

def run_segment() -> None:
    print("=" * 50)
    print("  Stage: Segment (split audio by segment definitions)")
    print("=" * 50)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    audio_files = _list_audio_files()
    found_any = False

    for audio_path in audio_files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        segments_path = os.path.join(SEGMENTS_DIR, stem + ".txt")
        if not os.path.isfile(segments_path):
            continue
        found_any = True
        if is_done(stem, "segmented"):
            print(f"  SKIP  {stem}  (already segmented)")
            continue

        segments = parse_segments_file(segments_path)
        if not segments:
            print(f"  Warning: {segments_path} has no valid entries — skipping.")
            continue

        chunks_dir = os.path.join(AUDIO_DIR, stem + "_segments")
        print(f"  Splitting {stem} into {len(segments)} chunk(s)…", end=" ", flush=True)
        t0 = time.time()
        chunk_files = split_audio(audio_path, segments, chunks_dir)
        elapsed = fmt_duration(time.time() - t0)
        print(f"done in {elapsed}")
        chunk_info = "\n".join(f"  {c['label']}: {c['path']}" for c in chunk_files)
        mark_done(stem, "segmented",
                  f"segments: {len(segments)}\nduration: {elapsed}\nchunks:\n{chunk_info}")

    if not found_any:
        print("No audio files with matching segment definitions found in data/segments/.")

    print("\nSegmentation complete.\n")


# ─── Stage: transcribe ───────────────────────────────────────────────────────

def run_transcribe() -> None:
    # Lazy import — loads Whisper model here (slow, not needed for other stages)
    from transcription import transcribe

    print("=" * 50)
    print("  Stage: Transcribe (audio → raw transcript)")
    print("=" * 50)
    os.makedirs(RAW_DIR, exist_ok=True)

    audio_files = _list_audio_files()
    if not audio_files:
        print(f"No audio files found in {AUDIO_DIR}. Run 'make extract' first if you have videos.")
        return

    for audio_path in audio_files:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        if is_done(stem, "transcribed"):
            print(f"  SKIP  {stem}  (already transcribed)")
            continue

        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"\n  {stem}  ({file_size_mb:.1f} MB)")
        print("  " + "-" * 38)

        chunks_dir = os.path.join(AUDIO_DIR, stem + "_segments")
        if os.path.isdir(chunks_dir):
            raw_text = _transcribe_segmented(stem, chunks_dir, transcribe)
        else:
            print("  Transcribing whole file…")
            t0 = time.time()
            raw_text = transcribe(audio_path)
            elapsed = fmt_duration(time.time() - t0)
            print(f"  Done in {elapsed} — {len(raw_text):,} chars")

        raw_path = os.path.join(RAW_DIR, stem + ".txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        mark_done(stem, "transcribed", f"raw: {raw_path}\nchars: {len(raw_text)}")

    print("\nTranscription complete.\n")


def _transcribe_segmented(stem: str, chunks_dir: str, transcribe_fn) -> str:
    chunk_files = sorted(
        os.path.join(chunks_dir, f)
        for f in os.listdir(chunks_dir)
        if f.endswith(".wav")
    )
    print(f"  Found {len(chunk_files)} segment chunk(s) in {chunks_dir}")
    parts = []
    for i, chunk_path in enumerate(chunk_files, 1):
        label = os.path.splitext(os.path.basename(chunk_path))[0]
        print(f"  [{i}/{len(chunk_files)}] {label}")
        t0 = time.time()
        raw = transcribe_fn(chunk_path)
        elapsed = fmt_duration(time.time() - t0)
        print(f"    Done in {elapsed} — {len(raw):,} chars")
        parts.append(f"## {label}\n\n{raw}")
    return "\n\n---\n\n".join(parts)


# ─── Stage: enhance ──────────────────────────────────────────────────────────

def run_enhance() -> None:
    # Lazy imports — not needed for extract/segment/transcribe stages
    from rag import load_context, query_context
    from llm import enhance

    print("=" * 50)
    print("  Stage: Enhance (raw transcript → final output)")
    print("=" * 50)

    load_context()

    if not os.path.isdir(RAW_DIR):
        print(f"No raw transcripts found. Run 'make transcribe' first.")
        return

    raw_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    if not raw_files:
        print(f"No raw transcripts found in {RAW_DIR}. Run 'make transcribe' first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for rf in raw_files:
        stem = os.path.splitext(rf)[0]
        if is_done(stem, "enhanced"):
            print(f"  SKIP  {stem}  (already enhanced)")
            continue

        raw_path = os.path.join(RAW_DIR, rf)
        out_path = os.path.join(OUTPUT_DIR, stem + ".txt")

        print(f"\n  {stem}")
        print("  " + "-" * 38)

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        t0 = time.time()
        print("  Retrieving context…")
        context = query_context(raw_text[:CONTEXT_QUERY_CHARS])
        print(f"  Done in {fmt_duration(time.time() - t0)}")

        t0 = time.time()
        print("  Enhancing with LLM…")
        final_text = enhance(raw_text, context)
        elapsed = fmt_duration(time.time() - t0)
        print(f"  Done in {elapsed} — {len(final_text):,} chars")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcription: {stem}\n\n")
            f.write(final_text)

        mark_done(stem, "enhanced", f"output: {out_path}\nchars: {len(final_text)}\nduration: {elapsed}")

    print("\nEnhancement complete.\n")


# ─── Full pipeline ────────────────────────────────────────────────────────────

def run_all() -> None:
    pipeline_start = time.time()
    print("=" * 50)
    print("  AEGEE Agora Transcriber — Full Pipeline")
    print("=" * 50)
    print()
    run_extract()
    run_segment()
    run_transcribe()
    run_enhance()
    total = fmt_duration(time.time() - pipeline_start)
    print("=" * 50)
    print(f"  All stages complete in {total}.")
    print("=" * 50)


# ─── Entry point ─────────────────────────────────────────────────────────────

STAGES = {
    "extract": run_extract,
    "segment": run_segment,
    "transcribe": run_transcribe,
    "enhance": run_enhance,
}


def main():
    parser = argparse.ArgumentParser(description="AEGEE Agora Transcriber")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=list(STAGES.keys()),
        default=None,
        help="Pipeline stage to run (extract / segment / transcribe / enhance). "
             "Omit to run all stages.",
    )
    args = parser.parse_args()

    if args.stage:
        STAGES[args.stage]()
    else:
        run_all()


if __name__ == "__main__":
    main()
