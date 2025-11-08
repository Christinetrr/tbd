"""
MongoDB Integration Test Script for Assistive Memory System
Tests conversation/event appending with duplicate detection and aggregation queries.
"""

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, DuplicateKeyError
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryIntegrationTest:
    """Test suite for assistive memory MongoDB operations."""
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "assistive_memory_test"
    ):
        """Initialize test database connection."""
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self.db = self.client[database_name]
            
            # Collections
            self.profiles = self.db.profiles
            self.timelines = self.db.timelines
            
            # Verify connection
            self.client.server_info()
            logger.info(f"Connected to test database: {database_name}")
            
            # Setup indexes
            self._create_indexes()
            
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for efficient querying."""
        try:
            self.profiles.create_index("name", unique=True)
            self.profiles.create_index("conversations.timestamp")
            self.timelines.create_index("profile_id")
            self.timelines.create_index("events.timestamp")
            logger.info("Indexes created")
        except PyMongoError as e:
            logger.error(f"Error creating indexes: {e}")
    
    def append_conversation(
        self,
        profile_id: str,
        summary: str,
        timestamp: datetime
    ) -> bool:
        """
        Append conversation to profile with duplicate detection.
        
        Args:
            profile_id: Profile ObjectId as string or name
            summary: Conversation summary text
            timestamp: Timezone-aware datetime
        
        Returns:
            True if appended, False if duplicate or error
        """
        try:
            # Check if profile exists
            profile = self.profiles.find_one({"name": profile_id})
            if not profile:
                logger.error(f"Profile not found: {profile_id}")
                return False
            
            # Check for duplicate (same timestamp and summary)
            duplicate = self.profiles.find_one({
                "name": profile_id,
                "conversations": {
                    "$elemMatch": {
                        "summary": summary,
                        "timestamp": timestamp
                    }
                }
            })
            
            if duplicate:
                logger.warning(f"Duplicate conversation detected for {profile_id}, skipping")
                return False
            
            # Append conversation
            result = self.profiles.update_one(
                {"name": profile_id},
                {
                    "$push": {
                        "conversations": {
                            "summary": summary,
                            "timestamp": timestamp
                        }
                    },
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"Appended conversation for {profile_id}: {summary[:50]}...")
                return True
            else:
                logger.warning(f"Failed to append conversation for {profile_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error appending conversation: {e}")
            return False
    
    def append_event(
        self,
        profile_id: str,
        caption: str,
        timestamp: datetime
    ) -> bool:
        """
        Append event to timeline with duplicate detection.
        
        Args:
            profile_id: Profile identifier (name)
            caption: Event description
            timestamp: Timezone-aware datetime
        
        Returns:
            True if appended, False if duplicate or error
        """
        try:
            # Get profile to ensure it exists
            profile = self.profiles.find_one({"name": profile_id})
            if not profile:
                logger.error(f"Profile not found: {profile_id}")
                return False
            
            profile_object_id = profile["_id"]
            
            # Check for duplicate event (same caption and timestamp)
            duplicate = self.timelines.find_one({
                "profile_id": profile_object_id,
                "events": {
                    "$elemMatch": {
                        "caption": caption,
                        "timestamp": timestamp
                    }
                }
            })
            
            if duplicate:
                logger.warning(f"Duplicate event detected for {profile_id}, skipping")
                return False
            
            # Append event (upsert timeline document if doesn't exist)
            result = self.timelines.update_one(
                {"profile_id": profile_object_id},
                {
                    "$push": {
                        "events": {
                            "caption": caption,
                            "timestamp": timestamp
                        }
                    },
                    "$set": {"updated_at": datetime.now(timezone.utc)}
                },
                upsert=True
            )
            
            if result.modified_count > 0 or result.upserted_id:
                logger.info(f"Appended event for {profile_id}: {caption[:50]}...")
                return True
            else:
                logger.warning(f"Failed to append event for {profile_id}")
                return False
                
        except PyMongoError as e:
            logger.error(f"Error appending event: {e}")
            return False
    
    def get_daily_overview(self, target_date: datetime) -> Dict[str, Any]:
        """
        Aggregate daily interactions by profile.
        
        Args:
            target_date: Date to query (timezone-aware)
        
        Returns:
            Dictionary with profile interactions and counts
        """
        try:
            # Define date range
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Aggregate conversations
            conversation_pipeline = [
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
                    "$group": {
                        "_id": "$_id",
                        "profile_name": {"$first": "$name"},
                        "relation": {"$first": "$relation"},
                        "unique_conversations": {"$addToSet": "$conversations.summary"},
                        "conversation_count": {"$sum": 1}
                    }
                }
            ]
            
            conversations = list(self.profiles.aggregate(conversation_pipeline))
            
            # Aggregate events
            event_pipeline = [
                {"$unwind": "$events"},
                {
                    "$match": {
                        "events.timestamp": {
                            "$gte": start_of_day,
                            "$lt": end_of_day
                        }
                    }
                },
                {
                    "$lookup": {
                        "from": "profiles",
                        "localField": "profile_id",
                        "foreignField": "_id",
                        "as": "profile"
                    }
                },
                {"$unwind": "$profile"},
                {
                    "$group": {
                        "_id": "$profile_id",
                        "profile_name": {"$first": "$profile.name"},
                        "unique_events": {"$addToSet": "$events.caption"},
                        "event_count": {"$sum": 1}
                    }
                }
            ]
            
            events = list(self.timelines.aggregate(event_pipeline))
            
            # Combine results
            overview = {
                "date": start_of_day.isoformat(),
                "conversations": conversations,
                "events": events,
                "total_conversation_count": sum(c["conversation_count"] for c in conversations),
                "total_event_count": sum(e["event_count"] for e in events),
                "total_unique_profiles": len(set(
                    [c["profile_name"] for c in conversations] + 
                    [e["profile_name"] for e in events]
                ))
            }
            
            return overview
            
        except PyMongoError as e:
            logger.error(f"Error generating daily overview: {e}")
            return {
                "conversations": [],
                "events": [],
                "total_conversation_count": 0,
                "total_event_count": 0,
                "total_unique_profiles": 0
            }
    
    def setup_test_data(self):
        """Create test profiles."""
        logger.info("Setting up test data...")
        
        test_profiles = [
            {"name": "Alice Johnson", "relation": "Colleague"},
            {"name": "Bob Smith", "relation": "Friend"},
            {"name": "Carol Davis", "relation": "Manager"},
            {"name": "David Lee", "relation": "Client"}
        ]
        
        for profile in test_profiles:
            try:
                profile["conversations"] = []
                profile["created_at"] = datetime.now(timezone.utc)
                profile["updated_at"] = datetime.now(timezone.utc)
                
                self.profiles.insert_one(profile)
                logger.info(f"Created profile: {profile['name']}")
            except DuplicateKeyError:
                logger.info(f"Profile already exists: {profile['name']}")
    
    def verify_no_duplicates(self) -> bool:
        """Verify no duplicate conversations or events exist."""
        logger.info("Verifying no duplicates...")
        
        # Check for duplicate conversations
        duplicate_conversations = list(self.profiles.aggregate([
            {"$unwind": "$conversations"},
            {
                "$group": {
                    "_id": {
                        "name": "$name",
                        "summary": "$conversations.summary",
                        "timestamp": "$conversations.timestamp"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$match": {"count": {"$gt": 1}}}
        ]))
        
        if duplicate_conversations:
            logger.error(f"Found {len(duplicate_conversations)} duplicate conversations!")
            return False
        
        # Check for duplicate events
        duplicate_events = list(self.timelines.aggregate([
            {"$unwind": "$events"},
            {
                "$group": {
                    "_id": {
                        "profile_id": "$profile_id",
                        "caption": "$events.caption",
                        "timestamp": "$events.timestamp"
                    },
                    "count": {"$sum": 1}
                }
            },
            {"$match": {"count": {"$gt": 1}}}
        ]))
        
        if duplicate_events:
            logger.error(f"Found {len(duplicate_events)} duplicate events!")
            return False
        
        logger.info("✅ No duplicates found")
        return True
    
    def cleanup(self):
        """Clean up test data."""
        self.profiles.delete_many({})
        self.timelines.delete_many({})
        logger.info("Test data cleaned up")
    
    def close(self):
        """Close database connection."""
        self.client.close()
        logger.info("Database connection closed")


def simulate_processing_cycles(test_db: MemoryIntegrationTest, num_cycles: int = 20):
    """
    Simulate processing cycles with face recognition and transcription.
    
    Args:
        test_db: Test database instance
        num_cycles: Number of cycles to simulate
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"SIMULATING {num_cycles} PROCESSING CYCLES")
    logger.info(f"{'='*60}\n")
    
    # Hardcoded profile IDs (simulating DeepFace matches)
    profile_pool = ["Alice Johnson", "Bob Smith", "Carol Davis", "David Lee"]
    
    # Dummy conversation summaries
    conversation_templates = [
        "Project update discussion",
        "Reviewed quarterly goals",
        "Casual coffee chat",
        "Technical problem solving session",
        "Team collaboration meeting",
        "Brainstorming new features",
        "Status check-in",
        "Design review feedback",
        "Planning next sprint",
        "Client requirements discussion"
    ]
    
    # Dummy event captions
    event_templates = [
        "Met for coffee",
        "Lunch meeting",
        "Video call",
        "Office hallway conversation",
        "Team standup",
        "Code review session",
        "Whiteboard brainstorm",
        "Workshop attendance",
        "Conference call",
        "Quick sync-up"
    ]
    
    success_count = {"conversations": 0, "events": 0}
    duplicate_count = {"conversations": 0, "events": 0}
    
    base_time = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
    
    for i in range(num_cycles):
        logger.info(f"\n--- Cycle {i+1}/{num_cycles} ---")
        
        # Simulate DeepFace match (hardcoded profile_id)
        matched_profile = random.choice(profile_pool)
        logger.info(f"🔍 Face detected: {matched_profile}")
        
        # Generate timestamp (spread throughout the day)
        timestamp = base_time + timedelta(minutes=i*30)
        
        # Generate conversation summary
        summary = random.choice(conversation_templates)
        
        # Attempt to append conversation
        conversation_added = test_db.append_conversation(
            profile_id=matched_profile,
            summary=summary,
            timestamp=timestamp
        )
        
        if conversation_added:
            success_count["conversations"] += 1
        else:
            duplicate_count["conversations"] += 1
        
        # Generate event caption
        caption = random.choice(event_templates)
        
        # Attempt to append event
        event_added = test_db.append_event(
            profile_id=matched_profile,
            caption=caption,
            timestamp=timestamp
        )
        
        if event_added:
            success_count["events"] += 1
        else:
            duplicate_count["events"] += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Conversations added: {success_count['conversations']}")
    logger.info(f"⚠️  Duplicate conversations skipped: {duplicate_count['conversations']}")
    logger.info(f"✅ Events added: {success_count['events']}")
    logger.info(f"⚠️  Duplicate events skipped: {duplicate_count['events']}")
    logger.info(f"{'='*60}\n")
    
    return success_count, duplicate_count, base_time


def run_tests():
    """Main test runner."""
    logger.info("\n" + "="*60)
    logger.info("ASSISTIVE MEMORY MONGODB INTEGRATION TEST")
    logger.info("="*60 + "\n")
    
    # Initialize test database
    test_db = MemoryIntegrationTest(
        connection_string="mongodb://localhost:27017/",
        database_name="assistive_memory_test"
    )
    
    try:
        # Clean up any existing test data
        test_db.cleanup()
        
        # Setup test profiles
        test_db.setup_test_data()
        
        # Simulate processing cycles
        success_count, duplicate_count, base_time = simulate_processing_cycles(test_db, num_cycles=20)
        
        # Test duplicate detection by trying to add same data again
        logger.info("\n--- Testing Duplicate Detection ---")
        logger.info("Attempting to add duplicate entries...")
        
        # Get actual data from database to test duplicates
        carol = test_db.profiles.find_one({"name": "Carol Davis"})
        if carol and carol["conversations"]:
            first_conv = carol["conversations"][0]
            dup_conv = test_db.append_conversation(
                "Carol Davis",
                first_conv["summary"],
                first_conv["timestamp"]
            )
        else:
            dup_conv = False
        
        carol_timeline = test_db.timelines.find_one({"profile_id": carol["_id"]})
        if carol_timeline and carol_timeline["events"]:
            first_event = carol_timeline["events"][0]
            dup_event = test_db.append_event(
                "Carol Davis",
                first_event["caption"],
                first_event["timestamp"]
            )
        else:
            dup_event = False
        
        assert not dup_conv, "Duplicate conversation should be rejected!"
        assert not dup_event, "Duplicate event should be rejected!"
        logger.info("✅ Duplicate detection working correctly")
        
        # Test non-existent profile error handling
        logger.info("\n--- Testing Error Handling ---")
        invalid_result = test_db.append_conversation(
            "NonExistent Person",
            "This should fail",
            datetime.now(timezone.utc)
        )
        assert not invalid_result, "Non-existent profile should return False!"
        logger.info("✅ Error handling working correctly")
        
        # Verify no duplicates
        assert test_db.verify_no_duplicates(), "Duplicate check failed!"
        
        # Generate daily overview
        logger.info("\n--- Generating Daily Overview ---")
        today = datetime.now(timezone.utc)
        overview = test_db.get_daily_overview(today)
        
        logger.info(f"Date: {overview['date']}")
        logger.info(f"Total unique profiles: {overview['total_unique_profiles']}")
        logger.info(f"Total conversations: {overview['total_conversation_count']}")
        logger.info(f"Total events: {overview['total_event_count']}")
        
        logger.info("\nBreakdown by profile:")
        for conv in overview['conversations']:
            logger.info(f"  {conv['profile_name']} ({conv['relation']}): "
                       f"{conv['conversation_count']} conversations")
        
        for event in overview['events']:
            logger.info(f"  {event['profile_name']}: {event['event_count']} events")
        
        # Assertions
        logger.info("\n--- Running Assertions ---")
        
        assert overview['total_conversation_count'] == success_count['conversations'], \
            f"Expected {success_count['conversations']} conversations, got {overview['total_conversation_count']}"
        logger.info(f"✅ Conversation count matches: {overview['total_conversation_count']}")
        
        assert overview['total_event_count'] == success_count['events'], \
            f"Expected {success_count['events']} events, got {overview['total_event_count']}"
        logger.info(f"✅ Event count matches: {overview['total_event_count']}")
        
        assert overview['total_unique_profiles'] > 0, "Should have at least one profile"
        logger.info(f"✅ Unique profiles found: {overview['total_unique_profiles']}")
        
        # Verify unique counts
        for conv in overview['conversations']:
            assert len(conv['unique_conversations']) <= conv['conversation_count'], \
                "Unique summaries can't exceed total count"
        logger.info("✅ Unique conversation counts valid")
        
        for event in overview['events']:
            assert len(event['unique_events']) <= event['event_count'], \
                "Unique events can't exceed total count"
        logger.info("✅ Unique event counts valid")
        
        logger.info("\n" + "="*60)
        logger.info("ALL TESTS PASSED! ✅")
        logger.info("="*60 + "\n")
        
    except AssertionError as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up test data...")
        test_db.cleanup()
        test_db.close()


if __name__ == "__main__":
    run_tests()
