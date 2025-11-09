import os
from pathlib import Path
from typing import Iterator, Optional

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


DEFAULT_VOICE_ID = "XrExE9yKIg1WjnnlVkGX"  # Matilda
PCM_OUTPUT_FORMAT = "pcm_16000"
PCM_SAMPLE_RATE = 16_000


def _load_env_from_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"\'"')
        os.environ.setdefault(key, value)


_load_env_from_file()

_elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
if not _elevenlabs_api_key:
    raise RuntimeError("ELEVENLABS_API_KEY is not set")

elevenlabs_client = ElevenLabs(api_key=_elevenlabs_api_key)


def _voice_settings() -> VoiceSettings:
    return VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
    )


def text_to_speech(
    text: str,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    output_format: str = PCM_OUTPUT_FORMAT,
    model_id: str = "eleven_turbo_v2_5",
    output_path: Optional[Path] = None,
) -> bytes:
    if not text:
        raise ValueError("Text cannot be empty")

    audio_generator = elevenlabs_client.text_to_speech.convert(
        voice_id=voice_id,
        optimize_streaming_latency="0",
        output_format=output_format,
        text=text,
        model_id=model_id,
        voice_settings=_voice_settings(),
    )

    audio_data = b"".join(audio_generator)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(audio_data)

    return audio_data


def text_to_speech_stream(
    text: str,
    *,
    voice_id: str = DEFAULT_VOICE_ID,
    output_format: str = PCM_OUTPUT_FORMAT,
    model_id: str = "eleven_turbo_v2_5",
) -> Iterator[bytes]:
    if not text:
        raise ValueError("Text cannot be empty")

    audio_generator = elevenlabs_client.text_to_speech.convert(
        voice_id=voice_id,
        optimize_streaming_latency="4",
        output_format=output_format,
        text=text,
        model_id=model_id,
        voice_settings=_voice_settings(),
    )

    for chunk in audio_generator:
        yield chunk


def get_available_voices():
    return elevenlabs_client.voices.get_all().voices
