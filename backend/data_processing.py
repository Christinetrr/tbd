'''
process livefeed data from webcam 

1) preprocess data to eliminate noise 
2) face recognitiion 
    i) if face detected within radius -> run similarity check against existing profiles, 
        run live audio processing, send to LLM for summarization, store
        summarization and time in DB (nurse, parents, etc.)
3) Live video feed processing
    i) capture video feed in frames only when significant scene change
    -> send to LLM for summarization
        -> store summarization and time in DB
    ii) have temporary current relevant conversation recording (for the individual to 
    query their current conversation, indicate redundancy )
4) Data handling and storage
    i) thorughout the day scenes and events summarized
    ii) temporary current relevant conversation recording 
    iii) summarized audio data associated with relevant profile
    iv) facial profiles 
'''
import collections
import logging
import queue
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import sounddevice as sd
import webrtcvad
import threading
import wave

try:
    from .api import summarize_frames
except ImportError:  # pragma: no cover - allows running as script
    from api import summarize_frames

#initialize face cascade
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_FAMILIAR_URL = "https://64aa23c1f114.ngrok-free.app/recognize"
AUDIO_OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"

START_TALKING_FLAG = False


class AudioRecorder:
    def __init__(self, sample_rate: int = 16_000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Optional[sd.InputStream] = None
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def _callback(self, indata, _frames, _time, status):
        if status:
            logging.debug("Audio stream status: %s", status)
        with self._lock:
            self._buffer.extend(bytes(indata))

    def start(self):
        if sd is None:
            raise RuntimeError("sounddevice library not available; cannot capture audio.")
        if self._stream is not None:
            return
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        logging.info("Audio capture started.")

    def stop(self) -> bytes:
        if self._stream is None:
            return b""
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
        logging.info("Audio capture stopped.")
        return data


def _persist_audio(data: bytes, sample_rate: int, channels: int) -> Optional[Path]:
    if not data:
        return None
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = AUDIO_OUTPUT_DIR / f"audio_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.wav"
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(data)
    return filename


_audio_recorder: Optional[AudioRecorder] = None

def start_audio_capture() -> bool:
    global _audio_recorder
    if sd is None:
        logging.warning("Audio capture requires the sounddevice library.")
        return False
    if _audio_recorder is None:
        _audio_recorder = AudioRecorder()
    try:
        _audio_recorder.start()
        return True
    except Exception as exc:  # pragma: no cover - hardware dependent
        logging.warning("Unable to start audio capture: %s", exc)
        return False


def stop_audio_capture() -> Optional[Path]:
    global _audio_recorder
    if _audio_recorder is None:
        return None
    try:
        data = _audio_recorder.stop()
    except Exception as exc:  # pragma: no cover - hardware dependent
        logging.warning("Unable to stop audio capture: %s", exc)
        return None
    file_path = _persist_audio(data, _audio_recorder.sample_rate, _audio_recorder.channels)
    if file_path:
        logging.info("Saved audio capture to %s", file_path)
    return file_path


def _is_familiar_match(result) -> bool:
    if isinstance(result, dict):
        for key in ("match", "recognized", "familiar", "success", "found"):
            value = result.get(key)
            if isinstance(value, bool):
                return value
        return bool(result)
    return bool(result)


def preprocess_frame(frame):
    """preprocess video frame to reduce noise and normalize to grayscale"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(denoised)
    processed_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return processed_bgr, equalized

#class to keep previous frame states and used to detect significant scene change
class SceneChangeDetector:
    """Detects meaningful scene changes by tracking large pixel-area shifts."""

    # tune intensity_threshold, change_ratio, and smoothing for sensitivity
    def __init__(
        self,
        intensity_threshold: float = 35.0,
        change_ratio: float = 0.67,
        smoothing: int = 5,
    ):
        self.intensity_threshold = intensity_threshold
        self.change_ratio = change_ratio
        self.smoothing = max(smoothing, 1)
        self.previous_frame: Optional[np.ndarray] = None
        self._recent_ratios = collections.deque(maxlen=self.smoothing)

    def detect(self, frame: np.ndarray, gray: np.ndarray) -> bool:
        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        # dynamically compare the difference between previous frame, current frame, and 
        #update metrics  
        difference = cv2.absdiff(self.previous_frame, gray)
        _, mask = cv2.threshold(
            difference,
            self.intensity_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        changed_pixels = float(np.count_nonzero(cleaned_mask))
        total_pixels = float(cleaned_mask.size)

        #if the average ratio of the current frame is greater than the
        #threshold than return True, indicating a significant scene change
        ratio = changed_pixels / total_pixels if total_pixels else 0.0

        self.previous_frame = gray
        self._recent_ratios.append(ratio)
        averaged_ratio = float(np.mean(self._recent_ratios))
        return averaged_ratio > self.change_ratio


# constantly running and processing live webcam feed
def webcam_processing(camera_source=0, display_window: bool = False):
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source {camera_source!r}")

    scene_detector = SceneChangeDetector()
    frames_buffer: list[np.ndarray] = []
    period_start = time.time()
    was_face_detected = False
    capturing_audio = False
    # reads frames from webcam and projects
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # preprocess live video feed
        frame, gray_scaled_frame = preprocess_frame(frame)

        # only record frames when significant scene change is detected
        if scene_detector.detect(frame, gray_scaled_frame):
            print("detected scene change")
            frames_buffer.append(frame.copy())

        if frames_buffer and (time.time() - period_start) >= 3600:
            process_frame_batch(frames_buffer)
            frames_buffer.clear()
            period_start = time.time()


        face_present = face_detected(gray_scaled_frame)
        if face_present and not was_face_detected:
            snapshot = frame.copy()
            familiar_result = None
            try:
                familiar_result = find_familiar(snapshot)
                logging.info("Familiar face check: %s", familiar_result)
            except Exception as exc:
                logging.warning("find_familiar failed: %s", exc)
            if familiar_result and not capturing_audio and _is_familiar_match(familiar_result):
                START_TALKING_FLAG = True
                if start_audio_capture():
                    capturing_audio = True
                    logging.info("Audio capture engaged for familiar face.")
                # else:
                #     stop_audio_capture()
                #     capturing_audio = False
                #     START_TALKING_FLAG = False
        if capturing_audio and not audio_detected():
            stop_audio_capture()
            capturing_audio = False
            START_TALKING_FLAG = False
        was_face_detected = face_present

        # while face_detected(gray_scaled_frame) and audio_detected():
        #     print("SOMEONE IS SPEAKING: familiar face and audio detected")
        #     detection_flag = true
        #     recording_audio()
        # if detection_flag == true:
        #     recording_audio(frame)
        # else:
        #     continue

    if frames_buffer:
        process_frame_batch(frames_buffer)

    cap.release()
    if display_window:
        cv2.destroyAllWindows()

#find the camera device
def _parse_v4l2_devices(raw_output: str) -> dict[str, list[str]]:
    devices = {}
    current_device = None
    for line in raw_output.splitlines():
        if not line.strip():
            current_device = None
            continue
        if not line.startswith("\t"):
            current_device = line.rstrip(":").strip()
            devices[current_device] = []
        elif current_device:
            devices[current_device].append(line.strip())
    return devices


def resolve_brio_camera(
    target_name: str = "Brio 101",
    fallback_scan_limit: int = 10,
):
    """Locate the Logitech Brio video node, falling back to index scanning."""
    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        devices = _parse_v4l2_devices(output)
        target_pattern = re.compile(rf"{re.escape(target_name)}", re.IGNORECASE)
        for device_name, nodes in devices.items():
            if target_pattern.search(device_name):
                for node in nodes:
                    if node.startswith("/dev/video"):
                        logging.info("Using %s at %s", device_name, node)
                        return node
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logging.warning("Failed to query v4l2 devices: %s", exc)

    logging.info(
        "Falling back to scanning numeric indices (0-%s)", fallback_scan_limit - 1
    )
    return scan_camera_indices(limit=fallback_scan_limit)


def scan_camera_indices(limit: int = 10):
    """Return the first usable camera index within the provided range."""
    first_index = None
    for idx in range(limit):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera found at index {idx}")
            first_index = idx if first_index is None else first_index
            cap.release()
        else:
            cap.release()
    return first_index

#summarizes batch of frames and store summary in database with timestamp
def process_frame_batch(frames):
    summary = summarize_frames(frames)
    record_frames(frames, summary)

def record_frames(frames, summary):
    pass

#facial similarity check and recognition
def face_detected(gray_frame):
    if FACE_CASCADE.empty():
        logging.warning("OpenCV Haar cascade failed to load; skipping face detection.")
        return False

    detections = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    has_face = len(detections) > 0
    if has_face:
        logging.debug("Detected %d face(s) in current frame.", len(detections))
    return has_face

    #compare with similarity from database recorded facial profiles


def find_familiar(snapshot: np.ndarray):
    ok, encoded = cv2.imencode(".jpg", snapshot)
    if not ok:
        raise RuntimeError("Failed to encode snapshot for familiar-face lookup.")

    files = {
        "image": ("snapshot.jpg", encoded.tobytes(), "image/jpeg"),
    }
    headers = {"ngrok-skip-browser-warning": "true"}
    response = requests.post(
        _FAMILIAR_URL,
        headers=headers,
        files=files,
        timeout=20,
    )
    response.raise_for_status()
    try:
        return response.json()
    except ValueError:
        text = response.text.strip()
        return {"raw": text} if text else {}


#speech detection class using WebRTC VAD (ONLY DETECTS PRESENCE OF SPEECH)
class _VadSpeechDetector:
    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 30,
        #voice detection sensitivity level
        aggressiveness: int = 3,
        #length of silence to consider the user as not speaking
        max_silence_ms: int = 3000,
    ):
        if webrtcvad is None:
            raise RuntimeError("webrtcvad library not available.")
        #audio format parameters and settings
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # int16
        self.max_silence_frames = max(int(max_silence_ms / frame_duration_ms), 1)
        self._vad = webrtcvad.Vad(min(max(aggressiveness, 0), 3))
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._silence_counter = 0
        self._is_speaking = False
        self._ensure_stream()
    #MAIN FUNCTION retrieving audio frames ACTIVELY from the audio stream
    def _audio_callback(self, indata, frames, _time, status):
        if status:
            logging.debug("Audio stream status: %s", status)
        self._queue.put(bytes(indata))

    #ensure that the audio stream is active and running, if not start it
    def _ensure_stream(self):
        if sd is None:
            logging.warning(
                "sounddevice library not available; skipping audio monitoring startup."
            )
            return
        if self._stream is not None and self._stream.active:
            return
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                dtype="int16",
                channels=1,
                callback=self._audio_callback,
            )
            self._stream.start()
            logging.info("Audio monitoring stream started.")
        except Exception as exc:
            logging.warning("Unable to start audio stream: %s", exc)
            self._stream = None
    #pull frames out of the callback queue (queue actively receiving audio frames)
    #and run VAD to determine if the user is speaking
    def _process_queue(self):
        updated_speaking_state = self._is_speaking
        while not self._queue.empty():
            chunk = self._queue.get()
            if len(chunk) < self.frame_bytes:
                # pad short frames (e.g., last partial frame) with zeros
                chunk = chunk + b"\x00" * (self.frame_bytes - len(chunk))
            is_speech = self._vad.is_speech(chunk, self.sample_rate)

            if is_speech:
                self._silence_counter = 0
                updated_speaking_state = True
            elif self._is_speaking:
                self._silence_counter += 1
                if self._silence_counter >= self.max_silence_frames:
                    updated_speaking_state = False
                    self._silence_counter = 0
                else:
                    updated_speaking_state = True

        self._is_speaking = updated_speaking_state
    #check if the user is speaking, handles micro silencing
    def is_speaking(self) -> bool:
        if sd is None or webrtcvad is None:
            logging.warning(
                "Audio detection disabled; install sounddevice and webrtcvad."
            )
            return False
        if self._stream is None:
            self._ensure_stream()
        if self._stream is None:
            return False
        self._process_queue()
        return self._is_speaking


_speech_detector: Optional[_VadSpeechDetector] = None


def audio_detected() -> bool:
    global _speech_detector
    if _speech_detector is None:
        if sd is None or webrtcvad is None:
            logging.warning(
                "audio_detected requires sounddevice and webrtcvad; returning False."
            )
            return False
        _speech_detector = _VadSpeechDetector()
    return _speech_detector.is_speaking()

#run live audio processing and transcription appending to current conversation
def recording_audio(frame):
    #process face in frame
    pass

#send to LLM for summarization, process text, store
def process_text(text):
    pass

def record_frame(frame):
    #record frame to database
    pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    camera_source = resolve_brio_camera()
    if camera_source is None:
        raise RuntimeError("No usable camera source detected.")

    logging.info("Opening camera source %r", camera_source)
    #start the webcam processing
    webcam_processing(camera_source=camera_source, display_window=False)

if __name__ == "__main__":
    main()