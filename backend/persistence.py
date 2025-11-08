"""
Persistence module for storing embeddings, summaries, conversation buffers,
and facial profiles in MongoDB database.
"""
import json
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, PyMongoError

from models import (
    FacialProfile, SummaryRecord, ConversationBuffer,
    FaceRecognitionResult, AudioSummary, SceneSummary
)

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple vector store for face embeddings using numpy.
    In production, use a proper vector database like Pinecone, Weaviate, or FAISS.
    """
    
    def __init__(self, dimension: int = 128):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimension of embedding vectors
        """
        self.dimension = dimension
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized vector store: dimension={dimension}")
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """
        Add an embedding with metadata.
        
        Args:
            embedding: Embedding vector
            metadata: Associated metadata (profile_id, etc.)
        """
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}")
        
        self.embeddings.append(embedding)
        self.metadata.append(metadata)
        
        logger.debug(f"Added embedding for profile: {metadata.get('profile_id')}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of results with metadata and similarity scores
        """
        if not self.embeddings:
            return []
        
        # Compute cosine similarities
        embeddings_matrix = np.array(self.embeddings)
        similarities = np.dot(embeddings_matrix, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter by threshold and prepare results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                results.append({
                    "metadata": self.metadata[idx],
                    "similarity": similarity,
                    "index": int(idx)
                })
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "dimension": self.dimension
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.dimension = data["dimension"]
        logger.info(f"Vector store loaded from {filepath}")


class DatabaseManager:
    """
    Manages MongoDB database for storing summaries, profiles, and conversation buffers.
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "livefeed_db"
    ):
        """
        Initialize MongoDB database manager.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Database name
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.db = None
        
        self._initialize_database()
        logger.info(f"Initialized MongoDB manager: {database_name}")
    
    def _initialize_database(self):
        """Connect to MongoDB and create indexes."""
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            self.db = self.client[self.database_name]
            
            # Create collections (automatically created on first insert, but explicit is better)
            self.profiles = self.db.facial_profiles
            self.summaries = self.db.summary_records
            self.conversations = self.db.conversation_buffers
            self.profile_summaries = self.db.profile_summaries
            
            # Create indexes for better query performance
            self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("Continuing without database connection")
            self.client = None
            self.db = None
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {e}", exc_info=True)
            self.client = None
            self.db = None
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        if self.db is None:
            return
        
        try:
            # Facial profiles indexes
            self.profiles.create_index("profile_id", unique=True)
            self.profiles.create_index("last_seen", name="last_seen_idx")
            self.profiles.create_index("encounter_count", name="encounter_count_idx")
            
            # Summary records indexes
            self.summaries.create_index("record_id", unique=True)
            self.summaries.create_index("timestamp", name="timestamp_idx")
            self.summaries.create_index("record_type", name="record_type_idx")
            self.summaries.create_index("profile_ids", name="profile_ids_idx")
            
            # Conversation buffers indexes
            self.conversations.create_index("buffer_id", unique=True)
            self.conversations.create_index("expiry_time", name="expiry_time_idx")
            
            # Profile-summary associations indexes
            self.profile_summaries.create_index(
                [("profile_id", ASCENDING), ("summary_id", ASCENDING)],
                name="profile_summary_idx"
            )
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def save_facial_profile(self, profile: FacialProfile):
        """
        Save or update a facial profile.
        
        Args:
            profile: FacialProfile to save
        """
        if self.db is None:
            logger.warning("Database not available, skipping profile save")
            return
        
        try:
            # Convert profile to MongoDB document
            # Note: embeddings are stored in vector store, not here
            doc = {
                "profile_id": profile.profile_id,
                "name": profile.name,
                "first_seen": profile.first_seen,
                "last_seen": profile.last_seen,
                "encounter_count": profile.encounter_count,
                "conversation_summaries": profile.conversation_summaries,
                "metadata": profile.metadata
            }
            
            # Upsert (update if exists, insert if not)
            self.profiles.update_one(
                {"profile_id": profile.profile_id},
                {"$set": doc},
                upsert=True
            )
            
            logger.debug(f"Saved facial profile: {profile.profile_id}")
        except PyMongoError as e:
            logger.error(f"Error saving profile: {e}", exc_info=True)
    
    def get_facial_profile(self, profile_id: str) -> Optional[FacialProfile]:
        """
        Retrieve a facial profile by ID.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            FacialProfile or None if not found
        """
        if self.db is None:
            return None
        
        try:
            doc = self.profiles.find_one({"profile_id": profile_id})
            
            if doc is None:
                return None
            
            # Convert MongoDB document to FacialProfile
            profile = FacialProfile(
                profile_id=doc["profile_id"],
                name=doc.get("name"),
                first_seen=doc["first_seen"],
                last_seen=doc["last_seen"],
                encounter_count=doc["encounter_count"],
                conversation_summaries=doc.get("conversation_summaries", []),
                metadata=doc.get("metadata", {})
            )
            
            return profile
        except PyMongoError as e:
            logger.error(f"Error retrieving profile: {e}", exc_info=True)
            return None
    
    def save_summary_record(self, summary: SummaryRecord):
        """
        Save a summary record.
        
        Args:
            summary: SummaryRecord to save
        """
        if self.db is None:
            logger.warning("Database not available, skipping summary save")
            return
        
        try:
            # Convert to MongoDB document
            doc = {
                "record_id": summary.record_id,
                "record_type": summary.record_type,
                "summary_text": summary.summary_text,
                "timestamp": summary.timestamp,
                "start_time": summary.start_time,
                "end_time": summary.end_time,
                "confidence": summary.confidence,
                "profile_ids": summary.profile_ids,
                "location": summary.location,
                "tags": summary.tags,
                "metadata": summary.metadata
            }
            
            # Upsert summary
            self.summaries.update_one(
                {"record_id": summary.record_id},
                {"$set": doc},
                upsert=True
            )
            
            # Create profile-summary associations
            for profile_id in summary.profile_ids:
                self.profile_summaries.update_one(
                    {
                        "profile_id": profile_id,
                        "summary_id": summary.record_id
                    },
                    {
                        "$set": {
                            "profile_id": profile_id,
                            "summary_id": summary.record_id,
                            "timestamp": summary.timestamp
                        }
                    },
                    upsert=True
                )
            
            logger.debug(f"Saved summary record: {summary.record_id}")
        except PyMongoError as e:
            logger.error(f"Error saving summary: {e}", exc_info=True)
    
    def get_summaries_by_profile(
        self,
        profile_id: str,
        limit: int = 10
    ) -> List[SummaryRecord]:
        """
        Get summaries associated with a profile.
        
        Args:
            profile_id: Profile identifier
            limit: Maximum number of summaries to retrieve
            
        Returns:
            List of SummaryRecord objects
        """
        if self.db is None:
            return []
        
        try:
            # Get summary IDs for this profile
            associations = self.profile_summaries.find(
                {"profile_id": profile_id}
            ).sort("timestamp", DESCENDING).limit(limit)
            
            summary_ids = [assoc["summary_id"] for assoc in associations]
            
            # Get the actual summaries
            summaries = []
            for doc in self.summaries.find({"record_id": {"$in": summary_ids}}):
                summary = SummaryRecord(
                    record_id=doc["record_id"],
                    record_type=doc["record_type"],
                    summary_text=doc["summary_text"],
                    timestamp=doc["timestamp"],
                    start_time=doc["start_time"],
                    end_time=doc["end_time"],
                    confidence=doc["confidence"],
                    profile_ids=doc.get("profile_ids", []),
                    location=doc.get("location"),
                    tags=doc.get("tags", []),
                    metadata=doc.get("metadata", {})
                )
                summaries.append(summary)
            
            # Sort by timestamp descending
            summaries.sort(key=lambda s: s.timestamp, reverse=True)
            
            return summaries[:limit]
        except PyMongoError as e:
            logger.error(f"Error retrieving summaries: {e}", exc_info=True)
            return []
    
    def save_conversation_buffer(
        self,
        buffer_id: str,
        buffer: ConversationBuffer
    ):
        """
        Save a conversation buffer with TTL.
        
        Args:
            buffer_id: Unique identifier for buffer
            buffer: ConversationBuffer to save
        """
        if self.db is None:
            logger.warning("Database not available, skipping buffer save")
            return
        
        try:
            expiry_time = buffer.start_time + timedelta(minutes=buffer.max_duration_minutes)
            
            # Serialize segments
            segments_data = [
                {
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "timestamp": seg.timestamp,
                    "speaker_id": seg.speaker_id,
                    "language": seg.language,
                    "duration_ms": seg.duration_ms
                }
                for seg in buffer.segments
            ]
            
            doc = {
                "buffer_id": buffer_id,
                "participants": buffer.participants,
                "start_time": buffer.start_time,
                "last_update": buffer.last_update,
                "expiry_time": expiry_time,
                "segments": segments_data,
                "max_duration_minutes": buffer.max_duration_minutes
            }
            
            # Upsert with TTL
            self.conversations.update_one(
                {"buffer_id": buffer_id},
                {"$set": doc},
                upsert=True
            )
            
            logger.debug(f"Saved conversation buffer: {buffer_id}")
        except PyMongoError as e:
            logger.error(f"Error saving conversation buffer: {e}", exc_info=True)
    
    def get_conversation_buffer(self, buffer_id: str) -> Optional[ConversationBuffer]:
        """
        Retrieve a conversation buffer.
        
        Args:
            buffer_id: Buffer identifier
            
        Returns:
            ConversationBuffer or None if not found or expired
        """
        if self.db is None:
            return None
        
        try:
            doc = self.conversations.find_one({"buffer_id": buffer_id})
            
            if doc is None:
                return None
            
            # Check if expired
            if datetime.now() > doc["expiry_time"]:
                self.delete_conversation_buffer(buffer_id)
                return None
            
            # Deserialize segments
            from models import TranscriptionResult
            segments = [
                TranscriptionResult(
                    text=seg["text"],
                    confidence=seg["confidence"],
                    timestamp=seg["timestamp"],
                    speaker_id=seg.get("speaker_id"),
                    language=seg.get("language", "en"),
                    duration_ms=seg.get("duration_ms", 0)
                )
                for seg in doc["segments"]
            ]
            
            buffer = ConversationBuffer(
                segments=segments,
                participants=doc["participants"],
                start_time=doc["start_time"],
                last_update=doc["last_update"],
                max_duration_minutes=doc.get("max_duration_minutes", 30)
            )
            
            return buffer
        except PyMongoError as e:
            logger.error(f"Error retrieving conversation buffer: {e}", exc_info=True)
            return None
    
    def delete_conversation_buffer(self, buffer_id: str):
        """Delete an expired conversation buffer."""
        if self.db is None:
            return
        
        try:
            self.conversations.delete_one({"buffer_id": buffer_id})
            logger.debug(f"Deleted conversation buffer: {buffer_id}")
        except PyMongoError as e:
            logger.error(f"Error deleting buffer: {e}", exc_info=True)
    
    def cleanup_expired_buffers(self):
        """Remove all expired conversation buffers."""
        if self.db is None:
            return
        
        try:
            result = self.conversations.delete_many(
                {"expiry_time": {"$lt": datetime.now()}}
            )
            
            if result.deleted_count > 0:
                logger.info(f"Cleaned up {result.deleted_count} expired conversation buffers")
        except PyMongoError as e:
            logger.error(f"Error cleaning up buffers: {e}", exc_info=True)
    
    def get_all_profiles(self, limit: int = 100) -> List[FacialProfile]:
        """
        Get all facial profiles.
        
        Args:
            limit: Maximum number of profiles to return
            
        Returns:
            List of FacialProfile objects
        """
        if self.db is None:
            return []
        
        try:
            profiles = []
            for doc in self.profiles.find().limit(limit):
                profile = FacialProfile(
                    profile_id=doc["profile_id"],
                    name=doc.get("name"),
                    first_seen=doc["first_seen"],
                    last_seen=doc["last_seen"],
                    encounter_count=doc["encounter_count"],
                    conversation_summaries=doc.get("conversation_summaries", []),
                    metadata=doc.get("metadata", {})
                )
                profiles.append(profile)
            
            return profiles
        except PyMongoError as e:
            logger.error(f"Error retrieving profiles: {e}", exc_info=True)
            return []
    
    def close(self):
        """Close database connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


class PersistenceManager:
    """
    Unified persistence manager coordinating MongoDB database and vector store.
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "livefeed_db",
        vector_store_path: str = "face_embeddings.pkl"
    ):
        """
        Initialize persistence manager.
        
        Args:
            connection_string: MongoDB connection string
            database_name: MongoDB database name
            vector_store_path: Path to vector store file
        """
        self.db = DatabaseManager(connection_string, database_name)
        self.vector_store = VectorStore(dimension=128)
        self.vector_store_path = vector_store_path
        
        # Load existing vector store if available
        if Path(vector_store_path).exists():
            try:
                self.vector_store.load(vector_store_path)
            except Exception as e:
                logger.warning(f"Failed to load vector store: {e}")
        
        logger.info("Initialized persistence manager with MongoDB")
    
    def store_face_recognition_result(
        self,
        result: FaceRecognitionResult,
        create_new_profile: bool = True
    ) -> str:
        """
        Store face recognition result and update or create profile.
        
        Args:
            result: Face recognition result
            create_new_profile: Create new profile if no match found
            
        Returns:
            Profile ID (existing or newly created)
        """
        if result.is_match and result.profile_id:
            # Update existing profile
            profile = self.db.get_facial_profile(result.profile_id)
            if profile:
                profile.update_last_seen()
                if result.detection.embedding is not None:
                    profile.add_embedding(result.detection.embedding)
                self.db.save_facial_profile(profile)
                return result.profile_id
        
        elif create_new_profile and result.detection.embedding is not None:
            # Create new profile for unknown face
            import uuid
            profile_id = f"profile_{uuid.uuid4().hex[:8]}"
            
            profile = FacialProfile(
                profile_id=profile_id,
                embeddings=[result.detection.embedding],
                first_seen=result.timestamp,
                last_seen=result.timestamp,
                encounter_count=1
            )
            
            self.db.save_facial_profile(profile)
            
            # Add to vector store
            self.vector_store.add(
                result.detection.embedding,
                {"profile_id": profile_id}
            )
            self._save_vector_store()
            
            logger.info(f"Created new facial profile: {profile_id}")
            return profile_id
        
        return ""
    
    def store_audio_summary(
        self,
        summary: AudioSummary,
        buffer_id: Optional[str] = None
    ) -> str:
        """
        Store audio summary and optionally link to conversation buffer.
        
        Args:
            summary: Audio summary to store
            buffer_id: Optional conversation buffer ID
            
        Returns:
            Summary record ID
        """
        import uuid
        record_id = f"audio_{uuid.uuid4().hex[:8]}"
        
        summary_record = SummaryRecord(
            record_id=record_id,
            record_type="audio",
            summary_text=summary.summary_text,
            timestamp=datetime.now(),
            start_time=summary.start_time,
            end_time=summary.end_time,
            confidence=summary.confidence,
            profile_ids=summary.participants,
            metadata={
                "buffer_id": buffer_id,
                "transcription_count": len(summary.transcriptions)
            }
        )
        
        self.db.save_summary_record(summary_record)
        logger.info(f"Stored audio summary: {record_id}")
        return record_id
    
    def store_scene_summary(self, summary: SceneSummary) -> str:
        """
        Store scene summary.
        
        Args:
            summary: Scene summary to store
            
        Returns:
            Summary record ID
        """
        import uuid
        record_id = f"scene_{uuid.uuid4().hex[:8]}"
        
        summary_record = SummaryRecord(
            record_id=record_id,
            record_type="scene",
            summary_text=summary.summary_text,
            timestamp=datetime.now(),
            start_time=summary.start_time,
            end_time=summary.end_time,
            confidence=summary.confidence,
            profile_ids=summary.participant_ids,
            location=summary.location,
            tags=summary.tags,
            metadata={
                "frame_count": len(summary.frame_ids),
                "frame_ids": summary.frame_ids
            }
        )
        
        self.db.save_summary_record(summary_record)
        logger.info(f"Stored scene summary: {record_id}")
        return record_id
    
    def get_profile_history(self, profile_id: str) -> Dict[str, Any]:
        """
        Get complete history for a profile.
        
        Args:
            profile_id: Profile identifier
            
        Returns:
            Dictionary with profile and associated summaries
        """
        profile = self.db.get_facial_profile(profile_id)
        summaries = self.db.get_summaries_by_profile(profile_id)
        
        return {
            "profile": profile,
            "summaries": summaries
        }
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        try:
            self.vector_store.save(self.vector_store_path)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}", exc_info=True)
    
    def close(self):
        """Close all persistence connections."""
        self._save_vector_store()
        self.db.close()
        logger.info("Persistence manager closed")
