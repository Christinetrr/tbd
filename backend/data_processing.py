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
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

try:
    from backend.api import summarize_frames
    from backend.db import append_timeline_event
except ImportError:
    import pathlib
    import sys

    backend_dir = pathlib.Path(__file__).resolve().parent
    if str(backend_dir) not in sys.path:
        sys.path.append(str(backend_dir))

    from api import summarize_frames
    from db import append_timeline_event

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
        change_ratio: float = 0.50,
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
    """Continuously read from the webcam, preprocess, and react to scene changes.

    Args:
        camera_source: Index or path to the camera device.
        display_window: Whether to show a live window via OpenCV highgui.
    """
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source {camera_source!r}")

    scene_detector = SceneChangeDetector()
    frames_buffer: list[np.ndarray] = []
    period_start = time.time()
    # reads frames from webcam and projects
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # preprocess live video feed
        frame, gray = preprocess_frame(frame)

        #print("preprocessed frame", frame)
        if display_window:
            try:
                cv2.imshow("BRIO feed", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as exc:
                logging.warning("Disabling display due to OpenCV error: %s", exc)
                display_window = False

        # only record frames when significant scene change is detected
        if scene_detector.detect(frame, gray):
            print("detected scene change", frame)
            frames_buffer.append(frame.copy())

        if frames_buffer and (time.time() - period_start) >= 3600:
            process_frame_batch(frames_buffer)
            frames_buffer.clear()
            period_start = time.time()

        # if familiar face detected, run the facial recognition process and record conversation
        if face_detected(frame):
            process_face(frame)
        else:
            continue

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
    append_timeline_event(summary)

#facial similarity check
def face_detected(frame):
    #detect face in frame using deepface library
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        detections = DeepFace.extract_faces(
            img_path=rgb_frame,
            detector_backend="retinaface",
            enforce_detection=False,
        )
    except Exception as exc:
        print(f"DeepFace detection error: {exc}")
        return False

    has_face = bool(detections)
    if has_face:
        print(True)
    return has_face

    #compare with similarity from database recorded facial profiles

#run live audio processing, send to LLM for summarization, process text, store
def process_face(frame):
    #process face in frame
    pass


def record_frame(frame):
    append_timeline_event("frame captured")


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