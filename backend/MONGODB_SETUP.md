# MongoDB Setup Guide

This guide will help you set up MongoDB for the livefeed processing pipeline.

## Quick Start

### Option 1: Local MongoDB Installation

#### macOS
```bash
# Install MongoDB using Homebrew
brew tap mongodb/brew
brew install mongodb-community@7.0

# Start MongoDB service
brew services start mongodb-community@7.0

# Verify installation
mongosh --eval "db.version()"
```

#### Linux (Ubuntu/Debian)
```bash
# Import MongoDB public GPG key
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
   sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify installation
mongosh --eval "db.version()"
```

#### Windows
1. Download MongoDB Community Server from https://www.mongodb.com/try/download/community
2. Run the installer and follow the setup wizard
3. Add MongoDB to PATH: `C:\Program Files\MongoDB\Server\7.0\bin`
4. Start MongoDB: `mongod`

### Option 2: MongoDB Atlas (Cloud)

1. **Create Free Account**
   - Go to https://www.mongodb.com/cloud/atlas
   - Sign up for free tier (512 MB storage)

2. **Create Cluster**
   - Click "Build a Database"
   - Choose "Free" tier
   - Select your preferred cloud provider and region
   - Click "Create"

3. **Configure Access**
   - Add your IP address to whitelist (or use 0.0.0.0/0 for testing)
   - Create database user with username and password

4. **Get Connection String**
   - Click "Connect" → "Connect your application"
   - Copy connection string (looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)

5. **Update Configuration**
   ```json
   {
     "persistence": {
       "mongodb_connection_string": "mongodb+srv://username:password@cluster.mongodb.net/",
       "mongodb_database_name": "livefeed_db"
     }
   }
   ```

### Option 3: Docker

```bash
# Run MongoDB in Docker container
docker run -d \
  --name mongodb \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:7.0

# Verify container is running
docker ps | grep mongodb

# Connection string for Docker setup
# mongodb://admin:password@localhost:27017/
```

## Python Driver Installation

```bash
pip install pymongo
```

## Verify Connection

Test your MongoDB connection:

```python
from pymongo import MongoClient

# For local MongoDB
client = MongoClient("mongodb://localhost:27017/")

# For MongoDB Atlas
# client = MongoClient("mongodb+srv://username:password@cluster.mongodb.net/")

# For Docker with authentication
# client = MongoClient("mongodb://admin:password@localhost:27017/")

# Test connection
try:
    client.admin.command('ping')
    print("MongoDB connection successful!")
    
    # List databases
    print("Databases:", client.list_database_names())
    
except Exception as e:
    print(f"MongoDB connection failed: {e}")

client.close()
```

## Configuration

### config.json

```json
{
  "persistence": {
    "mongodb_connection_string": "mongodb://localhost:27017/",
    "mongodb_database_name": "livefeed_db",
    "vector_store_path": "face_embeddings.pkl",
    "conversation_buffer_ttl_minutes": 30,
    "auto_cleanup_interval_seconds": 300
  }
}
```

### Environment Variables (Recommended for Credentials)

Create `.env` file:
```bash
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_DATABASE_NAME=livefeed_db

# For MongoDB Atlas with credentials
# MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/

# For Docker with authentication
# MONGODB_CONNECTION_STRING=mongodb://admin:password@localhost:27017/
```

Update your code to use environment variables:
```python
import os
from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv(
    "MONGODB_CONNECTION_STRING",
    "mongodb://localhost:27017/"
)
database_name = os.getenv("MONGODB_DATABASE_NAME", "livefeed_db")
```

## Database Schema

The pipeline creates these collections automatically:

### facial_profiles
```javascript
{
  profile_id: "profile_abc123",      // Unique identifier
  name: "John Doe",                   // Optional name
  first_seen: ISODate("2025-11-08"), // First encounter
  last_seen: ISODate("2025-11-08"),  // Last encounter
  encounter_count: 5,                 // Number of encounters
  conversation_summaries: [],         // Array of summary IDs
  metadata: {}                        // Additional metadata
}
```

### summary_records
```javascript
{
  record_id: "audio_xyz789",          // Unique identifier
  record_type: "audio",               // "audio", "scene", or "combined"
  summary_text: "Discussion about...", // Generated summary
  timestamp: ISODate("2025-11-08"),   // Creation timestamp
  start_time: ISODate("2025-11-08"),  // Event start
  end_time: ISODate("2025-11-08"),    // Event end
  confidence: 0.95,                   // Summary confidence
  profile_ids: ["profile_abc123"],    // Associated profiles
  location: "Office",                 // Optional location
  tags: ["meeting", "work"],          // Tags
  metadata: {}                        // Additional metadata
}
```

### conversation_buffers
```javascript
{
  buffer_id: "current_conversation",  // Unique identifier
  participants: ["profile_abc123"],   // Array of profile IDs
  start_time: ISODate("2025-11-08"), // Buffer start
  last_update: ISODate("2025-11-08"), // Last update
  expiry_time: ISODate("2025-11-08"), // TTL expiration
  segments: [                          // Transcription segments
    {
      text: "Hello...",
      confidence: 0.95,
      timestamp: ISODate("2025-11-08"),
      speaker_id: "profile_abc123",
      language: "en",
      duration_ms: 1500
    }
  ],
  max_duration_minutes: 30
}
```

### profile_summaries (associations)
```javascript
{
  profile_id: "profile_abc123",
  summary_id: "audio_xyz789",
  timestamp: ISODate("2025-11-08")
}
```

## Indexes

The system automatically creates these indexes for performance:

```javascript
// facial_profiles
db.facial_profiles.createIndex({ profile_id: 1 }, { unique: true })
db.facial_profiles.createIndex({ last_seen: 1 })
db.facial_profiles.createIndex({ encounter_count: 1 })

// summary_records
db.summary_records.createIndex({ record_id: 1 }, { unique: true })
db.summary_records.createIndex({ timestamp: 1 })
db.summary_records.createIndex({ record_type: 1 })
db.summary_records.createIndex({ profile_ids: 1 })

// conversation_buffers
db.conversation_buffers.createIndex({ buffer_id: 1 }, { unique: true })
db.conversation_buffers.createIndex({ expiry_time: 1 })

// profile_summaries
db.profile_summaries.createIndex({ profile_id: 1, summary_id: 1 })
```

## MongoDB Compass (GUI Tool)

MongoDB Compass is a free GUI for viewing and managing your data:

1. Download from https://www.mongodb.com/try/download/compass
2. Install and open Compass
3. Connect using your connection string
4. Browse collections, run queries, create indexes

## Common Operations

### View Data in MongoDB Shell

```bash
# Connect to MongoDB
mongosh

# Switch to database
use livefeed_db

# Show collections
show collections

# View profiles
db.facial_profiles.find().pretty()

# View summaries
db.summary_records.find().limit(5).pretty()

# Count documents
db.facial_profiles.countDocuments()

# Find specific profile
db.facial_profiles.findOne({ profile_id: "profile_abc123" })

# Get recent summaries
db.summary_records.find().sort({ timestamp: -1 }).limit(10)

# Find summaries for a profile
db.profile_summaries.find({ profile_id: "profile_abc123" })
```

### Backup and Restore

```bash
# Backup entire database
mongodump --db livefeed_db --out ./backup

# Restore database
mongorestore --db livefeed_db ./backup/livefeed_db

# Export collection to JSON
mongoexport --db livefeed_db --collection facial_profiles --out profiles.json

# Import collection from JSON
mongoimport --db livefeed_db --collection facial_profiles --file profiles.json
```

## Security Best Practices

### 1. Enable Authentication

```bash
# Create admin user
mongosh admin
db.createUser({
  user: "admin",
  pwd: "secure_password",
  roles: ["userAdminAnyDatabase", "dbAdminAnyDatabase", "readWriteAnyDatabase"]
})

# Create app-specific user
use livefeed_db
db.createUser({
  user: "livefeed_app",
  pwd: "app_password",
  roles: ["readWrite"]
})

# Restart MongoDB with authentication
mongod --auth
```

Update connection string:
```
mongodb://livefeed_app:app_password@localhost:27017/livefeed_db
```

### 2. Use Environment Variables

Never commit credentials to git:
```bash
# .env
MONGODB_CONNECTION_STRING=mongodb://livefeed_app:app_password@localhost:27017/livefeed_db
```

### 3. Enable TLS/SSL (Production)

For MongoDB Atlas, TLS is enabled by default. For local deployments:

```bash
mongod --tlsMode requireTLS --tlsCertificateKeyFile /path/to/cert.pem
```

## Troubleshooting

### Connection Refused

```bash
# Check if MongoDB is running
ps aux | grep mongod

# Start MongoDB
brew services start mongodb-community@7.0  # macOS
sudo systemctl start mongod                # Linux

# Check logs
tail -f /usr/local/var/log/mongodb/mongo.log  # macOS
tail -f /var/log/mongodb/mongod.log           # Linux
```

### Authentication Failed

- Verify username and password in connection string
- Ensure user has correct permissions
- Check database name in connection string

### Cannot Connect to Atlas

- Verify IP whitelist includes your IP
- Check username/password (no special characters that need URL encoding)
- Ensure cluster is running (not paused)

### Slow Queries

- Check indexes are created: `db.collection.getIndexes()`
- Use MongoDB Compass to analyze query performance
- Enable profiling: `db.setProfilingLevel(1, { slowms: 100 })`

## Performance Tuning

### Connection Pooling

```python
from pymongo import MongoClient

client = MongoClient(
    "mongodb://localhost:27017/",
    maxPoolSize=50,          # Maximum connections
    minPoolSize=10,          # Minimum connections
    maxIdleTimeMS=45000,     # Close idle connections after 45s
    serverSelectionTimeoutMS=5000  # Timeout for server selection
)
```

### Batch Operations

For inserting multiple documents:
```python
# Instead of multiple insert_one calls
db.collection.insert_many([doc1, doc2, doc3])
```

### Caching

Consider caching frequently accessed profiles in memory.

## Migration from SQLite

If you have existing SQLite data to migrate:

```python
import sqlite3
from pymongo import MongoClient

# Connect to both databases
sqlite_conn = sqlite3.connect("livefeed_data.db")
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client.livefeed_db

# Migrate profiles
cursor = sqlite_conn.execute("SELECT * FROM facial_profiles")
for row in cursor:
    doc = {
        "profile_id": row[0],
        "name": row[1],
        "first_seen": row[2],
        "last_seen": row[3],
        "encounter_count": row[4],
        "metadata": json.loads(row[5])
    }
    mongo_db.facial_profiles.update_one(
        {"profile_id": doc["profile_id"]},
        {"$set": doc},
        upsert=True
    )

print("Migration complete!")
```

## Monitoring

### MongoDB Atlas (Cloud)

- Built-in monitoring dashboard
- Performance metrics
- Query profiling
- Alerts and notifications

### Self-Hosted

Use MongoDB's free monitoring:
```bash
mongod --enableFreeMonitoring on
```

Or use tools like:
- **Prometheus + Grafana**: For metrics visualization
- **MongoDB Ops Manager**: Enterprise monitoring (paid)
- **mongostat**: Real-time stats (`mongostat --port 27017`)

## Next Steps

1. ✅ Install MongoDB
2. ✅ Verify connection
3. ✅ Update config.json
4. ✅ Install pymongo: `pip install pymongo`
5. ✅ Run the pipeline: `python data_processing.py`
6. Monitor data in MongoDB Compass or shell

For production deployments, consider:
- Setting up replica sets for high availability
- Configuring sharding for horizontal scaling
- Implementing backup automation
- Setting up monitoring and alerting
