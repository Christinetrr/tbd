import os
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from bson import ObjectId
from pymongo import MongoClient, ReturnDocument

_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
_db_name = os.getenv("MONGO_DB", "tbd")
db = _client[_db_name]

_TIMELINE_ID = "timeline"


def _ensure_dt(ts: Optional[datetime]) -> datetime:
    return ts if ts else datetime.utcnow()


def append_timeline_event(caption: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    event = {"caption": caption, "timestamp": _ensure_dt(timestamp)}
    db.timelines.update_one({"_id": _TIMELINE_ID}, {"$push": {"events": event}}, upsert=True)
    return event


def get_timeline() -> Iterable[Dict[str, Any]]:
    doc = db.timelines.find_one({"_id": _TIMELINE_ID}, {"_id": 0, "events": 1})
    return doc.get("events", []) if doc else []


def create_profile(name: str, relation: str) -> Dict[str, Any]:
    profile = {"name": name, "relation": relation, "conversations": []}
    insert_result = db.profiles.insert_one(profile)
    profile["_id"] = insert_result.inserted_id
    return profile


def _as_object_id(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(str(value))


def add_conversation(profile_id: Any, summary: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    event = {"summary": summary, "timestamp": _ensure_dt(timestamp)}
    updated = db.profiles.find_one_and_update(
        {"_id": _as_object_id(profile_id)},
        {"$push": {"conversations": event}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise ValueError("profile not found")
    return event


def get_profile(profile_id: Any) -> Optional[Dict[str, Any]]:
    return db.profiles.find_one({"_id": _as_object_id(profile_id)})


def find_profile_by_name(name: str) -> Optional[Dict[str, Any]]:
    return db.profiles.find_one({"name": name})


def list_profiles() -> Iterable[Dict[str, Any]]:
    return db.profiles.find()


def ensure_profile(name: str, relation: str) -> Dict[str, Any]:
    profile = find_profile_by_name(name)
    return profile if profile else create_profile(name, relation)

