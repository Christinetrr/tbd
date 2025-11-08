import base64
import os

from typing import Iterable

import cv2
import numpy as np
from openai import OpenAI

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

    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )

    response = xai_client.chat.completions.create(
        model="grok-3",
        messages=[
            {
                "role": "system",
                "content": "You are an expert audio analysis system. Given a transcription of an audio file, your task is to summarize the essential events, actions, and scene transitions that occur. Focus on clarity, temporal flow, and relevance—omit redundant details. Output a concise natural-language summary that captures what happens across the audio, preserving order and context.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": transcription.text,
                    },
                ],
            },
        ],
    )

    return response.choices[0].message.content.strip()