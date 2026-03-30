import os
import ffmpeg

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".wmv"}


def extract_audio(video_path: str, audio_dir: str) -> str:
    """
    Extract audio from a video file and save as a 16 kHz mono WAV.
    Returns the path to the extracted audio file.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(audio_dir, f"{stem}.wav")

    (
        ffmpeg
        .input(video_path)
        .output(out_path, acodec="pcm_s16le", ar=16000, ac=1)
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path
