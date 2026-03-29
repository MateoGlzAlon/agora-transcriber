import os
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")

print(f"Loading Whisper model: {WHISPER_MODEL}")
model = whisper.load_model(WHISPER_MODEL)


def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe(audio_path: str) -> str:
    """
    Transcribe an audio file using Whisper.

    Returns a timestamped transcript, one line per segment:
        [HH:MM:SS --> HH:MM:SS] text
    """
    print(f"  Transcribing: {audio_path}")
    print("  (each segment will print as it is decoded — this is slow on CPU)")
    result = model.transcribe(audio_path, language=None, verbose=True)  # auto-detect language

    detected_lang = result.get("language", "unknown")
    print(f"  Detected language: {detected_lang}")

    segments = result.get("segments", [])
    if segments:
        lines = []
        for seg in segments:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            text = seg["text"].strip()
            lines.append(f"[{start} --> {end}] {text}")
        return "\n".join(lines)

    # Fallback: return plain text if no segments
    return result.get("text", "")
