"""
Integration test to connect video processing with database.
Tests how her data_processing.py works with your persistence.py and memory_schema.py
"""

import cv2
import numpy as np
from datetime import datetime, timezone
from data_processing import preprocess_frame, SceneChangeDetector
from persistence import PersistenceManager
from memory_schema import MemoryDatabase

def test_scene_detection_with_database():
    """Test scene change detection and save to database."""
    
    print("="*60)
    print("TEST 1: Scene Detection → Database Integration")
    print("="*60)
    
    # Initialize database
    db = MemoryDatabase(
        connection_string="mongodb://localhost:27017/",
        database_name="integration_test"
    )
    
    # Create a test profile
    db.insert_profile(
        name="Test User",
        relation="Self"
    )
    
    # Simulate scene changes
    detector = SceneChangeDetector()
    
    # Create test frames (simulating camera feed)
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)  # Different scene
    
    print("\n1. Processing frame 1...")
    processed1, gray1 = preprocess_frame(frame1)
    change1 = detector.detect(processed1, gray1)
    print(f"   Scene change detected: {change1}")
    
    print("\n2. Processing frame 2 (very different)...")
    processed2, gray2 = preprocess_frame(frame2)
    change2 = detector.detect(processed2, gray2)
    print(f"   Scene change detected: {change2}")
    
    if change2:
        # Record the scene change as an event
        db.add_event(
            caption="Significant scene change detected",
            timestamp=datetime.now(timezone.utc),
            metadata={"frame_shape": str(frame2.shape)}
        )
        print("   ✅ Scene change saved to timeline!")
    
    # Verify it was saved
    events = db.get_events()
    print(f"\n3. Total events in database: {len(events)}")
    if events:
        print(f"   Latest event: {events[0]['caption']}")
    
    db.close()
    print("\n✅ Test 1 complete!\n")


def test_face_detection_integration():
    """Test how face detection would work with facial profiles database."""
    
    print("="*60)
    print("TEST 2: Face Detection → Facial Profiles Database")
    print("="*60)
    
    # Initialize livefeed database
    persistence = PersistenceManager(
        connection_string="mongodb://localhost:27017/",
        database_name="integration_test_livefeed",
        vector_store_path="test_embeddings.pkl"
    )
    
    print("\n1. Simulating face detection...")
    
    # In real implementation, this would be from DeepFace
    fake_embedding = np.random.rand(512).tolist()
    
    print("   Face detected! Creating profile...")
    
    # Search for similar face (this would normally use the embedding)
    # For now, we'll just create a new profile
    
    from models import FacialProfile
    profile = FacialProfile(
        profile_id="profile_test_001",
        name="Unknown Person",
        first_seen=datetime.now(timezone.utc),
        last_seen=datetime.now(timezone.utc),
        encounter_count=1,
        conversation_summaries=[]
    )
    
    persistence.db.save_facial_profile(profile)
    print("   ✅ Profile saved!")
    
    # Verify
    retrieved = persistence.db.get_facial_profile("profile_test_001")
    if retrieved:
        print(f"\n2. Profile retrieved from database:")
        print(f"   Name: {retrieved.name}")
        print(f"   First seen: {retrieved.first_seen}")
        print(f"   Encounters: {retrieved.encounter_count}")
    
    persistence.close()
    print("\n✅ Test 2 complete!\n")


def test_conversation_recording():
    """Test how conversation recording would work with both databases."""
    
    print("="*60)
    print("TEST 3: Conversation Recording → Multiple Databases")
    print("="*60)
    
    # Use memory schema for user-facing data
    memory_db = MemoryDatabase(
        connection_string="mongodb://localhost:27017/",
        database_name="integration_test"
    )
    
    # Use persistence for internal livefeed data
    livefeed_db = PersistenceManager(
        connection_string="mongodb://localhost:27017/",
        database_name="integration_test_livefeed",
        vector_store_path="test_embeddings.pkl"
    )
    
    print("\n1. Face detected → Recording conversation...")
    
    # Add conversation to memory schema (user-facing)
    memory_db.add_conversation(
        name="Test User",
        summary="Discussed the weather and weekend plans",
        timestamp=datetime.now(timezone.utc)
    )
    print("   ✅ Conversation added to assistive memory!")
    
    # Also add to livefeed database for processing
    from models import SummaryRecord
    summary = SummaryRecord(
        record_id="summary_test_001",
        record_type="audio",
        summary_text="Discussed the weather and weekend plans",
        timestamp=datetime.now(timezone.utc),
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        confidence=0.95,
        profile_ids=["profile_test_001"]
    )
    
    livefeed_db.db.save_summary_record(summary)
    print("   ✅ Summary added to livefeed database!")
    
    # Verify both
    profile = memory_db.get_profile("Test User")
    print(f"\n2. Conversations for Test User: {len(profile['conversations'])}")
    
    summaries = livefeed_db.db.get_summaries_by_profile("profile_test_001")
    print(f"   Summaries for profile_test_001: {len(summaries)}")
    
    memory_db.close()
    livefeed_db.close()
    print("\n✅ Test 3 complete!\n")


def show_integration_summary():
    """Show how the pieces fit together."""
    
    print("\n" + "="*60)
    print("INTEGRATION SUMMARY")
    print("="*60)
    
    print("""
Her Implementation (data_processing.py):
├─ ✅ preprocess_frame()        → Cleans video frames
├─ ✅ SceneChangeDetector       → Detects scene changes
├─ ✅ webcam_processing()       → Main capture loop
├─ ✅ Camera detection          → Finds Logitech Brio
└─ ❌ Placeholders (need integration):
    ├─ face_detected()          → Needs: DeepFace + persistence.py
    ├─ process_face()           → Needs: memory_schema.py for conversations
    └─ record_frame()           → Needs: memory_schema.py for events

Your Database Implementation:
├─ ✅ persistence.py            → Livefeed system
│   ├─ facial_profiles          → Store detected faces
│   ├─ summary_records          → Store conversation summaries
│   └─ conversation_buffers     → Temporary conversation data
│
└─ ✅ memory_schema.py          → Assistive memory system
    ├─ profiles                 → People with conversations
    └─ timelines                → Events throughout the day

Integration Points:
1. face_detected(frame) should:
   - Use DeepFace to get face embedding
   - Search persistence.vector_store for similar faces
   - Return profile_id if match found

2. process_face(frame) should:
   - Get/create profile from persistence.py
   - Start audio recording
   - Add conversation to memory_schema.py

3. record_frame(frame) should:
   - Save frame as event in memory_schema.timelines
   - Store metadata (timestamp, scene summary)

Next Steps:
1. Install DeepFace: pip install deepface
2. Integrate face_detected() with your vector store
3. Connect process_face() to both databases
4. Test end-to-end with real webcam
""")
    
    print("="*60 + "\n")


def cleanup_test_databases():
    """Clean up test databases."""
    print("Cleaning up test databases...")
    
    memory_db = MemoryDatabase(database_name="integration_test")
    memory_db.clear_all_data()
    memory_db.close()
    
    livefeed_db = PersistenceManager(
        database_name="integration_test_livefeed",
        vector_store_path="test_embeddings.pkl"
    )
    livefeed_db.db.client.drop_database("integration_test_livefeed")
    livefeed_db.close()
    
    import os
    if os.path.exists("test_embeddings.pkl"):
        os.remove("test_embeddings.pkl")
    
    print("✅ Cleanup complete!")


if __name__ == "__main__":
    try:
        # Run integration tests
        test_scene_detection_with_database()
        test_face_detection_integration()
        test_conversation_recording()
        
        # Show summary
        show_integration_summary()
        
    finally:
        # Clean up
        cleanup_test_databases()
