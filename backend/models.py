"""
Data models for the livefeed processing system.
Simple dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any


@dataclass
class FacialProfile:
    """Profile for a recognized person."""
    profile_id: str
    name: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    encounter_count: int = 0
    conversation_summaries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryRecord:
    """Record of a summarized interaction."""
    record_id: str
    record_type: str  # "audio", "scene", "combined"
    summary_text: str
    timestamp: datetime
    start_time: datetime
    end_time: datetime
    confidence: float = 0.0
    profile_ids: List[str] = field(default_factory=list)
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationBuffer:
    """Temporary buffer for ongoing conversations."""
    buffer_id: str
    participants: List[str]
    start_time: datetime
    last_update: datetime
    expiry_time: datetime
    segments: List[Dict[str, Any]] = field(default_factory=list)
    max_duration_minutes: int = 5


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    confidence: float
    timestamp: datetime
    speaker_id: Optional[str] = None
    language: str = "en"
    duration_ms: int = 0


@dataclass
class FaceRecognitionResult:
    """Result from face recognition."""
    profile_id: Optional[str]
    is_match: bool
    confidence: float
    embedding: List[float]
    bounding_box: Optional[tuple] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AudioSummary:
    """Summary of an audio conversation."""
    summary_id: str
    summary_text: str
    confidence: float
    participants: List[str]
    start_time: datetime
    end_time: datetime
    key_topics: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSummary:
    """Summary of a visual scene."""
    summary_id: str
    description: str
    timestamp: datetime
    confidence: float
    participants: List[str] = field(default_factory=list)
    objects_detected: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
