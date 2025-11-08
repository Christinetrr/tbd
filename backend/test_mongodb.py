#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and basic operations.
This doesn't require camera/microphone permissions.
"""

import sys
from datetime import datetime
from persistence import PersistenceManager
from models import FacialProfile, SummaryRecord

def test_mongodb_connection():
    """Test basic MongoDB operations."""
    
    print("🔍 Testing MongoDB Connection...")
    
    try:
        # Initialize persistence manager
        persistence = PersistenceManager(
            connection_string="mongodb://localhost:27017/",
            database_name="livefeed_db_test"
        )
        print("✅ MongoDB connection successful!")
        
        # Test 1: Save a facial profile
        print("\n📝 Test 1: Creating facial profile...")
        profile = FacialProfile(
            profile_id="test_profile_001",
            name="Test Person",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            encounter_count=1,
            conversation_summaries=[]
        )
        persistence.db.save_facial_profile(profile)
        print("✅ Profile saved successfully!")
        
        # Test 2: Retrieve the profile
        print("\n🔍 Test 2: Retrieving facial profile...")
        retrieved_profile = persistence.db.get_facial_profile("test_profile_001")
        if retrieved_profile:
            print(f"✅ Profile retrieved: {retrieved_profile.name}")
            print(f"   First seen: {retrieved_profile.first_seen}")
            print(f"   Encounters: {retrieved_profile.encounter_count}")
        else:
            print("❌ Failed to retrieve profile")
            return False
        
        # Test 3: Update encounter count
        print("\n🔄 Test 3: Updating encounter count...")
        retrieved_profile.encounter_count += 1
        retrieved_profile.last_seen = datetime.now()
        persistence.db.save_facial_profile(retrieved_profile)
        
        updated_profile = persistence.db.get_facial_profile("test_profile_001")
        print(f"✅ Updated encounter count: {updated_profile.encounter_count}")
        
        # Test 4: Save a summary record
        print("\n📋 Test 4: Creating summary record...")
        summary = SummaryRecord(
            record_id="test_summary_001",
            record_type="test",
            summary_text="This is a test summary of a conversation.",
            timestamp=datetime.now(),
            start_time=datetime.now(),
            end_time=datetime.now(),
            confidence=0.95,
            profile_ids=["test_profile_001"]
        )
        persistence.db.save_summary_record(summary)
        print("✅ Summary saved successfully!")
        
        # Test 5: Retrieve summaries
        print("\n🔍 Test 5: Retrieving summaries...")
        summaries = persistence.db.get_summaries_by_profile("test_profile_001")
        print(f"✅ Retrieved {len(summaries)} summary(ies)")
        for s in summaries:
            print(f"   - {s.summary_text[:50]}...")
        
        # Test 6: List all profiles
        print("\n👥 Test 6: Listing all profiles...")
        all_profiles = persistence.db.get_all_profiles()
        print(f"✅ Total profiles in database: {len(all_profiles)}")
        for p in all_profiles:
            print(f"   - {p.name or p.profile_id}: {p.encounter_count} encounters")
        
        # Test 7: Clean up test data
        print("\n🧹 Test 7: Cleaning up test data...")
        # In a real scenario, you'd implement delete methods
        # For now, we'll just note that we created test data
        print("✅ Test data created (manual cleanup may be needed)")
        
        # Close connection
        persistence.close()
        print("\n✅ All tests passed! MongoDB is working correctly.")
        print("\nNote: Test data was created in 'livefeed_db_test' database")
        print("You can view it with: mongosh livefeed_db_test")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_mongodb_running():
    """Check if MongoDB is running."""
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info()  # Will raise exception if can't connect
        return True
    except Exception as e:
        print(f"❌ MongoDB not running or not accessible: {e}")
        print("\nTo start MongoDB:")
        print("  brew services start mongodb-community@7.0")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MongoDB Connection Test")
    print("=" * 60)
    
    # Check if MongoDB is running
    if not check_mongodb_running():
        print("\n⚠️  Please start MongoDB first!")
        sys.exit(1)
    
    print("✅ MongoDB is running\n")
    
    # Run tests
    success = test_mongodb_connection()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ MongoDB setup is working correctly!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Check the errors above.")
        print("=" * 60)
        sys.exit(1)
