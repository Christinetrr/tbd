import base64
import os
from pathlib import Path

from typing import Iterable

import cv2
import numpy as np
from openai import OpenAI
import logging


def _load_env_from_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_env_from_file()

_xai_api_key = os.getenv("GROK_API_KEY")
if not _xai_api_key:
    raise RuntimeError("GROK_API_KEY is not set")

_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

xai_client = OpenAI(api_key=_xai_api_key, base_url="https://api.x.ai/v1")
openai_client = OpenAI(api_key=_openai_api_key)

# send batch of video frames to Grok for summarization
def summarize_frames(frames_bgr: Iterable[np.ndarray]) -> str:
    """Send one or more frames (BGR NumPy arrays) to Grok and return a summary."""
    if isinstance(frames_bgr, np.ndarray):
        frames = [frames_bgr]
    else:
        frames = list(frames_bgr)

    if not frames:
        raise ValueError("Not enough frames: must contain at least one!")

    encoded_images = []
    for frame in frames:
        if frame is None:
            raise ValueError("frames_bgr must contain valid image arrays")
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Unable to encode frame as JPEG")
        encoded_images.append(
            {
                "type": "input_image",
                "image_base64": base64.b64encode(buffer).decode("utf-8"),
            }
        )

    response = xai_client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": (
                    """You are an expert video analysis system. 
                    Given a sequence of frames or visual descriptions extracted from a 
                    video, your task is to summarize the essential events, actions, and 
                    scene transitions that occur. Focus on clarity, temporal flow, and 
                    relevance—omit redundant details. 
                    Output a concise natural-language summary that captures what happens 
                    across the frames, preserving order and context."""
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Here is a collection of frames captured over the past hour. "
                            "Provide a concise summary of the key events."
                        ),
                    },
                    *encoded_images,
                ],
            },
        ],
    )

    return response.choices[0].message.content.strip()

#send audio file to whisper openai api for transcription
#send transcription to grok
def summarize_audio(audio_file):
    if audio_file is None:
        raise ValueError("audio file missing")

    # 1) Transcribe with Whisper
    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    text = transcription.text

    # 2) Summarize with Grok (sequential: uses the text above)
    completion = xai_client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given the following transcription text, create a concise summary "
                    "of the conversation. Use at most 3 sentences. "
                    "Be very concise and ONLY output plain text."
                ),
            },
            {
                "role": "user",
                "content": text,  # <-- plain string; no input_text wrapper
            },
        ],
    )

    return completion.choices[0].message.content.strip()
