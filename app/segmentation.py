import os
import re
import ffmpeg


def _parse_timestamp(ts: str) -> float:
    """
    Parse a timestamp string to seconds.
    Accepts MM:SS or H:MM:SS / HH:MM:SS formats.
    """
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unrecognised timestamp format: '{ts}'")


def parse_segments_file(path: str) -> list[dict]:
    """
    Parse a segments definition file.

    Expected line format (one segment per line):
        Label -> MM:SS - MM:SS
        Label -> H:MM:SS - H:MM:SS

    Lines that are blank or do not match are ignored.
    Returns a list of dicts: {label, start, end} where start/end are seconds.
    """
    segments: list[dict] = []
    pattern = re.compile(r"^(.+?)\s*->\s*([\d:]+)\s*-\s*([\d:]+)\s*$")

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                print(f"  [segments] Line {lineno} skipped (unrecognised format): {line!r}")
                continue
            label = m.group(1).strip()
            try:
                start = _parse_timestamp(m.group(2))
                end = _parse_timestamp(m.group(3))
            except ValueError as exc:
                print(f"  [segments] Line {lineno} skipped ({exc})")
                continue
            segments.append({"label": label, "start": start, "end": end})

    return segments


def split_audio(audio_path: str, segments: list[dict], output_dir: str) -> list[dict]:
    """
    Cut *audio_path* into one WAV file per segment using ffmpeg.

    Returns a list of dicts: {label, path} in the same order as *segments*.
    Output files are written to *output_dir* and named
    ``00_Label.wav``, ``01_Label.wav``, …
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[dict] = []

    for i, seg in enumerate(segments):
        safe_label = re.sub(r"[^\w\s-]", "", seg["label"]).strip().replace(" ", "_")
        out_path = os.path.join(output_dir, f"{i:02d}_{safe_label}.wav")
        duration = seg["end"] - seg["start"]

        (
            ffmpeg
            .input(audio_path, ss=seg["start"], t=duration)
            .output(out_path, acodec="pcm_s16le", ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        results.append({"label": seg["label"], "path": out_path})

    return results
