"""
MongoDB Schema for Assistive Memory System
Manages profiles and timelines with efficient querying and indexing.
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryDatabase:
    """
    Database manager for assistive memory system.
    Handles profiles (people) and timelines (events).
    """
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "assistive_memory"
    ):
        """
        Initialize MongoDB connection and setup collections.
        
        Args:
            connection_string: MongoDB connection URI
            database_name: Name of the database
        """
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.db = self.client[database_name]
            
            # Collections
            self.profiles = self.db.profiles
            self.timelines = self.db.timelines
            
            # Verify connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB: {database_name}")
            
            # Setup indexes
            self._create_indexes()
            
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for efficient querying."""
        try:
            # Profile indexes
            self.profiles.create_index("name", unique=True)
            self.profiles.create_index("relation")
            self.profiles.create_index("conversations.timestamp", name="conversation_timestamp")
            
            # Timeline indexes
            self.timelines.create_index("events.timestamp", name="event_timestamp")
            
            logger.info("Indexes created successfully")
            
        except PyMongoError as e:
            logger.error(f"Error creating indexes: {e}")
    
    # ==================== PROFILE OPERATIONS ====================
    
    def insert_profile(
        self,
        name: str,
        relation: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Insert a new profile.
        
        Args:
            name: Person's name (unique identifier)
            relation: Relationship to user (e.g., "Colleague", "Friend")
            embedding: Optional face embedding vector (e.g., from DeepFace)
            metadata: Optional additional data (e.g., contact info, notes)
        
        Returns:
            Profile ID if successful, None otherwise
        """
        try:
            profile = {
                "name": name,
                "relation": relation,
                "conversations": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Add optional fields
            if embedding is not None:
                profile["embedding"] = embedding
            
            if metadata is not None:
                profile["metadata"] = metadata
            
            result = self.profiles.insert_one(profile)
            logger.info(f"Profile created: {name}")
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            logger.warning(f"Profile already exists: {name}")
            return None
        except PyMongoError as e:
            logger.error(f"Error inserting profile: {e}")
            return None
    
    def update_profile(
        self,
        name: str,
        relation: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing profile.
        
        Args:
            name: Person's name
            relation: New relationship (optional)
            embedding: Updated face embedding (optional)
            metadata: Updated metadata (optional)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            update_fields = {"updated_at": datetime.utcnow()}
            
            if relation is not None:
                update_fields["relation"] = relation
            
            if embedding is not None:
                update_fields["embedding"] = embedding
            
            if metadata is not None:
                update_fields["metadata"] = metadata
            
            result = self.profiles.update_one(
                {"name": name},
                {"$set": update_fields}
            )
            
            if result.matched_count > 0:
                logger.info(f"Profile updated: {name}")
                return True
            else:
                logger.warning(f"Profile not found: {name}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error updating profile: {e}")
            return False
    
    def add_conversation(
        self,
        name: str,
        summary: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Add a conversation to a profile.
        
        Args:
            name: Person's name
            summary: Conversation summary
            timestamp: When the conversation occurred (defaults to now)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conversation = {
                "summary": summary,
                "timestamp": timestamp or datetime.utcnow()
            }
            
            result = self.profiles.update_one(
                {"name": name},
                {
                    "$push": {"conversations": conversation},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            
            if result.matched_count > 0:
                logger.info(f"Conversation added for: {name}")
                return True
            else:
                logger.warning(f"Profile not found: {name}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error adding conversation: {e}")
            return False
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a profile by name.
        
        Args:
            name: Person's name
        
        Returns:
            Profile document or None if not found
        """
        try:
            profile = self.profiles.find_one({"name": name})
            if profile:
                profile["_id"] = str(profile["_id"])  # Convert ObjectId to string
            return profile
        except PyMongoError as e:
            logger.error(f"Error retrieving profile: {e}")
            return None
    
    def get_all_profiles(self, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all profiles, optionally filtered by relation.
        
        Args:
            relation: Filter by relationship type (optional)
        
        Returns:
            List of profile documents
        """
        try:
            query = {"relation": relation} if relation else {}
            profiles = list(self.profiles.find(query))
            
            # Convert ObjectId to string
            for profile in profiles:
                profile["_id"] = str(profile["_id"])
            
            return profiles
        except PyMongoError as e:
            logger.error(f"Error retrieving profiles: {e}")
            return []
    
    def search_by_embedding(
        self,
        query_embedding: List[float],
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for profiles by face embedding similarity.
        Note: This is a basic implementation. For production, use vector databases
        like MongoDB Atlas Vector Search or FAISS for better performance.
        
        Args:
            query_embedding: Face embedding vector to search for
            threshold: Minimum similarity threshold (0-1)
            limit: Maximum number of results
        
        Returns:
            List of matching profiles with similarity scores
        """
        try:
            import numpy as np
            
            # Get all profiles with embeddings
            profiles = list(self.profiles.find({"embedding": {"$exists": True}}))
            
            results = []
            for profile in profiles:
                # Calculate cosine similarity
                embedding = np.array(profile["embedding"])
                query = np.array(query_embedding)
                
                similarity = np.dot(embedding, query) / (
                    np.linalg.norm(embedding) * np.linalg.norm(query)
                )
                
                if similarity >= threshold:
                    profile["_id"] = str(profile["_id"])
                    profile["similarity"] = float(similarity)
                    results.append(profile)
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            return results[:limit]
            
        except ImportError:
            logger.error("NumPy required for embedding search")
            return []
        except PyMongoError as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    # ==================== TIMELINE OPERATIONS ====================
    
    def add_event(
        self,
        caption: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add an event to the timeline.
        
        Args:
            caption: Event description
            timestamp: When the event occurred (defaults to now)
            metadata: Optional additional data (e.g., location, participants)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            event = {
                "caption": caption,
                "timestamp": timestamp or datetime.utcnow()
            }
            
            if metadata is not None:
                event["metadata"] = metadata
            
            # Timeline uses a single document with array of events
            # Update or create the timeline document
            result = self.timelines.update_one(
                {},  # Match any document (we only have one timeline)
                {
                    "$push": {"events": event},
                    "$set": {"updated_at": datetime.utcnow()}
                },
                upsert=True  # Create if doesn't exist
            )
            
            logger.info(f"Event added: {caption}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Error adding event: {e}")
            return False
    
    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get events from timeline, optionally filtered by date range.
        
        Args:
            start_date: Start of date range (optional)
            end_date: End of date range (optional)
            limit: Maximum number of events to return
        
        Returns:
            List of events
        """
        try:
            # Build aggregation pipeline
            pipeline = []
            
            # Unwind the events array
            pipeline.append({"$unwind": "$events"})
            
            # Filter by date range if provided
            if start_date or end_date:
                match_filter = {}
                if start_date:
                    match_filter["$gte"] = start_date
                if end_date:
                    match_filter["$lte"] = end_date
                
                pipeline.append({
                    "$match": {"events.timestamp": match_filter}
                })
            
            # Sort by timestamp (descending)
            pipeline.append({"$sort": {"events.timestamp": DESCENDING}})
            
            # Limit results
            pipeline.append({"$limit": limit})
            
            # Project only the event data
            pipeline.append({"$replaceRoot": {"newRoot": "$events"}})
            
            events = list(self.timelines.aggregate(pipeline))
            return events
            
        except PyMongoError as e:
            logger.error(f"Error retrieving events: {e}")
            return []
    
    def get_daily_interactions(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get all interactions (conversations and events) for a specific day.
        
        Args:
            date: Date to query (defaults to today)
        
        Returns:
            Dictionary with conversations and events for the day
        """
        try:
            target_date = date or datetime.utcnow()
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Get conversations from profiles
            conversations_pipeline = [
                {"$unwind": "$conversations"},
                {
                    "$match": {
                        "conversations.timestamp": {
                            "$gte": start_of_day,
                            "$lt": end_of_day
                        }
                    }
                },
                {
                    "$project": {
                        "name": 1,
                        "relation": 1,
                        "summary": "$conversations.summary",
                        "timestamp": "$conversations.timestamp"
                    }
                },
                {"$sort": {"timestamp": DESCENDING}}
            ]
            
            conversations = list(self.profiles.aggregate(conversations_pipeline))
            
            # Get events from timeline
            events = self.get_events(start_date=start_of_day, end_date=end_of_day)
            
            return {
                "date": start_of_day.isoformat(),
                "conversations": conversations,
                "events": events,
                "total_conversations": len(conversations),
                "total_events": len(events)
            }
            
        except PyMongoError as e:
            logger.error(f"Error retrieving daily interactions: {e}")
            return {"conversations": [], "events": [], "total_conversations": 0, "total_events": 0}
    
    # ==================== UTILITY OPERATIONS ====================
    
    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""
        try:
            result = self.profiles.delete_one({"name": name})
            if result.deleted_count > 0:
                logger.info(f"Profile deleted: {name}")
                return True
            else:
                logger.warning(f"Profile not found: {name}")
                return False
        except PyMongoError as e:
            logger.error(f"Error deleting profile: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all data (use with caution!)."""
        try:
            self.profiles.delete_many({})
            self.timelines.delete_many({})
            logger.info("All data cleared")
        except PyMongoError as e:
            logger.error(f"Error clearing data: {e}")
    
    def close(self):
        """Close MongoDB connection."""
        self.client.close()
        logger.info("Database connection closed")


# ==================== EXAMPLE USAGE ====================

def example_usage():
    """Demonstrate the assistive memory system."""
    
    # Initialize database
    db = MemoryDatabase()
    
    print("=" * 60)
    print("Assistive Memory System - Example Usage")
    print("=" * 60)
    
    # Example 1: Insert profiles
    print("\n1. Creating Profiles...")
    
    # Simulate a DeepFace embedding (512-dimensional vector)
    import random
    fake_embedding = [random.random() for _ in range(512)]
    
    db.insert_profile(
        name="Alex Smith",
        relation="Colleague",
        embedding=fake_embedding,
        metadata={"department": "Engineering", "email": "alex@company.com"}
    )
    
    db.insert_profile(
        name="Jamie Chen",
        relation="Friend",
        metadata={"phone": "555-0123"}
    )
    
    db.insert_profile(
        name="Dr. Sarah Johnson",
        relation="Doctor",
        metadata={"specialty": "Cardiology"}
    )
    
    # Example 2: Add conversations
    print("\n2. Adding Conversations...")
    
    db.add_conversation(
        name="Alex Smith",
        summary="Discussed project updates and upcoming deadlines",
        timestamp=datetime(2025, 11, 8, 14, 30, 0)
    )
    
    db.add_conversation(
        name="Alex Smith",
        summary="Coffee break chat about weekend plans",
        timestamp=datetime(2025, 11, 8, 10, 15, 0)
    )
    
    db.add_conversation(
        name="Jamie Chen",
        summary="Planned dinner for next Friday",
        timestamp=datetime(2025, 11, 8, 18, 0, 0)
    )
    
    # Example 3: Add timeline events
    print("\n3. Adding Timeline Events...")
    
    db.add_event(
        caption="Met Alex for coffee",
        timestamp=datetime(2025, 11, 8, 10, 15, 0),
        metadata={"location": "Starbucks on Main St"}
    )
    
    db.add_event(
        caption="Team meeting - Project review",
        timestamp=datetime(2025, 11, 8, 14, 0, 0),
        metadata={"duration": "1 hour", "attendees": ["Alex Smith", "Others"]}
    )
    
    db.add_event(
        caption="Doctor appointment",
        timestamp=datetime(2025, 11, 8, 16, 30, 0),
        metadata={"location": "Medical Center"}
    )
    
    # Example 4: Query a profile
    print("\n4. Retrieving Profile...")
    profile = db.get_profile("Alex Smith")
    if profile:
        print(f"   Name: {profile['name']}")
        print(f"   Relation: {profile['relation']}")
        print(f"   Conversations: {len(profile['conversations'])}")
        for conv in profile['conversations']:
            print(f"     - {conv['timestamp']}: {conv['summary']}")
    
    # Example 5: Get daily interactions
    print("\n5. Daily Interactions (Nov 8, 2025)...")
    daily = db.get_daily_interactions(datetime(2025, 11, 8))
    print(f"   Date: {daily['date']}")
    print(f"   Total Conversations: {daily['total_conversations']}")
    print(f"   Total Events: {daily['total_events']}")
    
    print("\n   Conversations:")
    for conv in daily['conversations']:
        print(f"     - {conv['name']} ({conv['relation']}): {conv['summary']}")
    
    print("\n   Events:")
    for event in daily['events']:
        print(f"     - {event['caption']}")
    
    # Example 6: Get all profiles by relation
    print("\n6. All Colleagues...")
    colleagues = db.get_all_profiles(relation="Colleague")
    for colleague in colleagues:
        print(f"   - {colleague['name']}")
    
    # Example 7: Search by embedding (if you had a face detection)
    print("\n7. Embedding Search Example...")
    # Simulate searching for a similar face
    query_embedding = [random.random() for _ in range(512)]
    matches = db.search_by_embedding(query_embedding, threshold=0.0, limit=3)
    print(f"   Found {len(matches)} matches")
    for match in matches:
        print(f"     - {match['name']}: {match['similarity']:.3f} similarity")
    
    # Close connection
    db.close()
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
