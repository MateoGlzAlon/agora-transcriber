# Speaker diarization module (optional / not active by default)
#
# Full automatic speaker diarization (identifying WHO speaks WHEN) requires
# pyannote.audio, which in turn needs a HuggingFace access token.
# Because this project runs 100% locally without external APIs or tokens,
# diarization is intentionally left as a stub.
#
# Whisper already outputs per-segment timestamps, which are included in the
# transcription output as [HH:MM:SS --> HH:MM:SS] markers.
#
# To enable diarization in the future:
#   1. Obtain a HuggingFace token and accept pyannote model licences
#   2. pip install pyannote.audio
#   3. Replace the stub below with the real implementation


def diarize(audio_path: str):
    """
    Placeholder — diarization is disabled in the default (no-API) configuration.
    Returns None.
    """
    return None
