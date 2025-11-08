# Assistive Memory System - MongoDB Schema Reference

## Overview
MongoDB schema for an assistive memory system that tracks people (profiles) and events (timelines).

## Collections

### 1. `profiles` Collection
Stores information about people and their conversations.

**Schema:**
```javascript
{
  _id: ObjectId,
  name: String,              // Unique - person's name
  relation: String,          // e.g., "Colleague", "Friend", "Doctor"
  conversations: [           // Array of conversation objects
    {
      summary: String,       // e.g., "Discussed project updates"
      timestamp: DateTime    // ISO format
    }
  ],
  embedding: [Float],        // Optional: Face recognition vector (512D from DeepFace)
  metadata: Object,          // Optional: Additional data (contact info, notes, etc.)
  created_at: DateTime,
  updated_at: DateTime
}
```

**Indexes:**
- `name` (unique)
- `relation`
- `conversations.timestamp`

---

### 2. `timelines` Collection
Stores daily events and activities.

**Schema:**
```javascript
{
  _id: ObjectId,
  events: [                  // Array of event objects
    {
      caption: String,       // e.g., "Met Alex for coffee"
      timestamp: DateTime,   // ISO format
      metadata: Object       // Optional: location, participants, duration, etc.
    }
  ],
  updated_at: DateTime
}
```

**Indexes:**
- `events.timestamp`

---

## Key Functions

### Profile Operations

#### `insert_profile(name, relation, embedding=None, metadata=None)`
Create a new profile.
```python
db.insert_profile(
    name="Alex Smith",
    relation="Colleague",
    embedding=[0.1, 0.2, ...],  # 512D vector from DeepFace
    metadata={"email": "alex@company.com", "department": "Engineering"}
)
```

#### `update_profile(name, relation=None, embedding=None, metadata=None)`
Update an existing profile.
```python
db.update_profile(
    name="Alex Smith",
    relation="Friend",
    metadata={"phone": "555-0123"}
)
```

#### `add_conversation(name, summary, timestamp=None)`
Add a conversation to a profile.
```python
db.add_conversation(
    name="Alex Smith",
    summary="Discussed project deadlines",
    timestamp=datetime(2025, 11, 8, 14, 30, 0)
)
```

#### `get_profile(name)`
Retrieve a profile by name.
```python
profile = db.get_profile("Alex Smith")
# Returns: {'name': 'Alex Smith', 'relation': 'Colleague', 'conversations': [...], ...}
```

#### `get_all_profiles(relation=None)`
Get all profiles, optionally filtered by relation.
```python
colleagues = db.get_all_profiles(relation="Colleague")
```

#### `search_by_embedding(query_embedding, threshold=0.7, limit=5)`
Search for profiles by face embedding similarity (for face recognition).
```python
matches = db.search_by_embedding(
    query_embedding=[0.1, 0.2, ...],
    threshold=0.7,
    limit=5
)
# Returns: [{'name': 'Alex Smith', 'similarity': 0.85, ...}, ...]
```

---

### Timeline Operations

#### `add_event(caption, timestamp=None, metadata=None)`
Add an event to the timeline.
```python
db.add_event(
    caption="Met Alex for coffee",
    timestamp=datetime(2025, 11, 8, 10, 15, 0),
    metadata={"location": "Starbucks on Main St"}
)
```

#### `get_events(start_date=None, end_date=None, limit=100)`
Get events from timeline, optionally filtered by date range.
```python
today = datetime(2025, 11, 8)
events = db.get_events(
    start_date=today,
    end_date=today + timedelta(days=1)
)
```

---

### Query Operations

#### `get_daily_interactions(date=None)`
Get all interactions (conversations + events) for a specific day.
```python
daily = db.get_daily_interactions(datetime(2025, 11, 8))
# Returns:
# {
#   'date': '2025-11-08T00:00:00',
#   'conversations': [...],
#   'events': [...],
#   'total_conversations': 3,
#   'total_events': 5
# }
```

---

## Sample Queries

### 1. Get all conversations with a specific person
```python
profile = db.get_profile("Alex Smith")
for conv in profile['conversations']:
    print(f"{conv['timestamp']}: {conv['summary']}")
```

### 2. Get today's interactions
```python
daily = db.get_daily_interactions()
print(f"Conversations: {daily['total_conversations']}")
print(f"Events: {daily['total_events']}")
```

### 3. Find all friends
```python
friends = db.get_all_profiles(relation="Friend")
for friend in friends:
    print(friend['name'])
```

### 4. Search for a face
```python
# After detecting a face with DeepFace
from deepface import DeepFace

# Get embedding from detected face
result = DeepFace.represent(img_path="face.jpg", model_name="Facenet512")
embedding = result[0]["embedding"]

# Search database
matches = db.search_by_embedding(embedding, threshold=0.7)
if matches:
    print(f"Recognized: {matches[0]['name']} ({matches[0]['similarity']:.2f})")
else:
    print("Unknown person")
```

### 5. Get recent events
```python
from datetime import datetime, timedelta

# Last 7 days
week_ago = datetime.utcnow() - timedelta(days=7)
recent_events = db.get_events(start_date=week_ago)

for event in recent_events:
    print(f"{event['timestamp']}: {event['caption']}")
```

---

## Direct MongoDB Queries

### View all profiles
```bash
mongosh assistive_memory
db.profiles.find().pretty()
```

### View profiles with conversations today
```javascript
db.profiles.aggregate([
  { $unwind: "$conversations" },
  { 
    $match: { 
      "conversations.timestamp": {
        $gte: new Date("2025-11-08T00:00:00Z"),
        $lt: new Date("2025-11-09T00:00:00Z")
      }
    }
  }
])
```

### View all events
```bash
db.timelines.findOne().events
```

### Count total conversations
```javascript
db.profiles.aggregate([
  { $project: { count: { $size: "$conversations" } } },
  { $group: { _id: null, total: { $sum: "$count" } } }
])
```

---

## Error Handling

The system includes error handling for:
- **Duplicate profiles**: Returns `None` if profile already exists
- **Profile not found**: Returns `False` or `None` for update/get operations
- **MongoDB connection errors**: Logged with detailed error messages
- **Missing dependencies**: Graceful fallback (e.g., NumPy for embedding search)

---

## Integration with DeepFace

Example workflow for face recognition:

```python
from deepface import DeepFace
from memory_schema import MemoryDatabase

db = MemoryDatabase()

# 1. When you meet someone new
img_path = "person.jpg"
result = DeepFace.represent(img_path, model_name="Facenet512")
embedding = result[0]["embedding"]

# 2. Search if they're in the database
matches = db.search_by_embedding(embedding, threshold=0.7)

if matches:
    # Known person
    name = matches[0]['name']
    print(f"Welcome back, {name}!")
    
    # Add conversation
    db.add_conversation(
        name=name,
        summary="Casual chat in hallway"
    )
else:
    # New person - create profile
    name = input("Who is this? ")
    relation = input("Relation? ")
    
    db.insert_profile(
        name=name,
        relation=relation,
        embedding=embedding
    )
```

---

## Performance Tips

1. **Use indexes**: Already created automatically on initialization
2. **Limit results**: Use `limit` parameter in queries
3. **Date ranges**: Always specify date ranges for timeline queries
4. **Vector search**: For production, consider MongoDB Atlas Vector Search or separate FAISS index
5. **Batch operations**: Use `insert_many` for bulk profile creation

---

## Running the Example

```bash
# Make sure MongoDB is running
brew services start mongodb-community@7.0

# Run the example
python memory_schema.py
```

---

## Database Commands

```bash
# View databases
mongosh --eval "show dbs"

# Connect to database
mongosh assistive_memory

# View collections
show collections

# Count profiles
db.profiles.countDocuments({})

# Drop database (careful!)
db.dropDatabase()
```
