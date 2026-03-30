import os
import time
from transcription import transcribe
from rag import load_context, query_context
from llm import enhance
from video import extract_audio, VIDEO_EXTENSIONS
from segmentation import parse_segments_file, split_audio

VIDEO_DIR = "/data/video"
AUDIO_DIR = "/data/audio"
SEGMENTS_DIR = "/data/segments"
OUTPUT_DIR = "/data/output"

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".webm", ".opus"}

# Use the first N characters of the transcript to query for relevant context
CONTEXT_QUERY_CHARS = 2000


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _collect_audio_files() -> list[str]:
    """
    Return absolute paths of all audio files to process:
    1. Videos from VIDEO_DIR are converted to WAV and added.
    2. Existing audio files in AUDIO_DIR are included.
    Duplicates (video already converted) are not re-processed.
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    converted_stems: set[str] = set()
    audio_paths: list[str] = []

    # --- convert videos ---
    video_files = sorted(
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if video_files:
        print(f"Found {len(video_files)} video file(s) — extracting audio…")
    for vf in video_files:
        stem = os.path.splitext(vf)[0]
        video_path = os.path.join(VIDEO_DIR, vf)
        print(f"  Extracting audio from {vf}…", end=" ", flush=True)
        t0 = time.time()
        wav_path = extract_audio(video_path, AUDIO_DIR)
        print(f"done in {fmt_duration(time.time() - t0)}  →  {wav_path}")
        converted_stems.add(stem)
        audio_paths.append(wav_path)

    # --- existing audio files (skip ones we just created from video) ---
    for af in sorted(os.listdir(AUDIO_DIR)):
        ext = os.path.splitext(af)[1].lower()
        stem = os.path.splitext(af)[0]
        if ext in AUDIO_EXTENSIONS and stem not in converted_stems:
            audio_paths.append(os.path.join(AUDIO_DIR, af))

    return audio_paths


def _process_file(audio_path: str, idx: int, total: int) -> None:
    file_start = time.time()
    filename = os.path.basename(audio_path)
    stem = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_DIR, stem + ".txt")

    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"[{idx}/{total}] {filename}  ({file_size_mb:.1f} MB)")
    print("-" * 40)

    # Check for a segments file
    segments_path = os.path.join(SEGMENTS_DIR, stem + ".txt")
    if os.path.isfile(segments_path):
        _process_segmented(audio_path, stem, segments_path, out_path)
    else:
        _process_whole(audio_path, filename, out_path)

    elapsed = fmt_duration(time.time() - file_start)
    print(f"Saved → {out_path}  (file total: {elapsed})\n")


def _process_whole(audio_path: str, filename: str, out_path: str) -> None:
    """Original behaviour: transcribe the whole file, then enhance."""
    t0 = time.time()
    print("Step 1/3 — Transcribing audio with Whisper…")
    raw_transcript = transcribe(audio_path)
    lines = raw_transcript.count("\n") + 1
    print(f"  Done in {fmt_duration(time.time() - t0)} — {lines} segments, {len(raw_transcript):,} chars")

    t0 = time.time()
    print("Step 2/3 — Retrieving relevant context…")
    context = query_context(raw_transcript[:CONTEXT_QUERY_CHARS])
    print(f"  Done in {fmt_duration(time.time() - t0)}")

    t0 = time.time()
    print("Step 3/3 — Enhancing transcription with local LLM…")
    final_transcript = enhance(raw_transcript, context)
    print(f"  Done in {fmt_duration(time.time() - t0)} — {len(final_transcript):,} chars")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcription: {filename}\n\n")
        f.write(final_transcript)


def _process_segmented(
    audio_path: str, stem: str, segments_path: str, out_path: str
) -> None:
    """Split audio by segment plan, transcribe each chunk, then enhance."""
    segments = parse_segments_file(segments_path)
    if not segments:
        print(f"  Warning: segments file {segments_path} has no valid entries — falling back to whole-file processing.")
        _process_whole(audio_path, os.path.basename(audio_path), out_path)
        return

    print(f"  Found {len(segments)} segment(s) in {os.path.basename(segments_path)}")

    chunks_dir = os.path.join(AUDIO_DIR, stem + "_segments")
    print(f"  Splitting audio into {len(segments)} chunk(s)…", end=" ", flush=True)
    t0 = time.time()
    chunk_files = split_audio(audio_path, segments, chunks_dir)
    print(f"done in {fmt_duration(time.time() - t0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    section_outputs: list[str] = []

    for i, chunk in enumerate(chunk_files, 1):
        label = chunk["label"]
        chunk_path = chunk["path"]
        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)

        print(f"\n  [{i}/{len(chunk_files)}] {label}  ({chunk_size_mb:.1f} MB)")

        t0 = time.time()
        print("    Transcribing…")
        raw = transcribe(chunk_path)
        print(f"    Done in {fmt_duration(time.time() - t0)} — {len(raw):,} chars")

        t0 = time.time()
        print("    Retrieving context…")
        context = query_context(raw[:CONTEXT_QUERY_CHARS])
        print(f"    Done in {fmt_duration(time.time() - t0)}")

        t0 = time.time()
        print("    Enhancing with LLM…")
        enhanced = enhance(raw, context)
        print(f"    Done in {fmt_duration(time.time() - t0)} — {len(enhanced):,} chars")

        section_outputs.append(f"## {label}\n\n{enhanced}")

    filename = os.path.basename(audio_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcription: {filename}\n\n")
        f.write("\n\n---\n\n".join(section_outputs))


def main():
    pipeline_start = time.time()

    print("=" * 50)
    print("  AEGEE Agora Transcriber")
    print("=" * 50)
    print()

    load_context()

    audio_files = _collect_audio_files()

    if not audio_files:
        print("No audio or video files found.")
        print(f"  • Place video files in {VIDEO_DIR}  (supported: {', '.join(sorted(VIDEO_EXTENSIONS))})")
        print(f"  • Place audio files in {AUDIO_DIR}  (supported: {', '.join(sorted(AUDIO_EXTENSIONS))})")
        return

    print(f"\nFound {len(audio_files)} file(s) to process.\n")

    for idx, audio_path in enumerate(audio_files, 1):
        _process_file(audio_path, idx, len(audio_files))

    total = fmt_duration(time.time() - pipeline_start)
    print("=" * 50)
    print(f"  All files processed in {total}.")
    print("=" * 50)


if __name__ == "__main__":
    main()
