'''
Backend for the application
1) Allow user to set up manual facial profiles
2) Extract from database (throughout the day accumulated data, 
   current conversation audio data, etc.) and build relevant timeline of events
'''

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from typing import Optional, List
import base64
from PIL import Image
import io
import numpy as np

# Import your existing memory schema
from memory_schema import MemoryDatabase

# IMPORTANT: This uses OpenCV as a placeholder for face detection
# 
# TODO FOR COLLEAGUE: Replace extract_face_embedding() function below with:
# - face_recognition library (if you can install it)
# - OR send embeddings directly from Raspberry Pi to /api/profiles/recognize
# - The embedding MUST be 128-dimensional to match the database schema
#
# If using face_recognition:
#   import face_recognition
#   face_encodings = face_recognition.face_encodings(image)
#   embedding = face_encodings[0].tolist()  # 128-dim vector
#
import cv2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/Users/drea3/princeton/tbd/backend/uploads'

# Create uploads folder if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

# Initialize database
db = MemoryDatabase()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_face_embedding(image_path: str) -> Optional[List[float]]:
    """
    Extract facial embedding from image.
    Returns 128-dimensional embedding vector or None if no face found.
    
    NOTE: This is a placeholder implementation using OpenCV for face detection.
    Replace this with their actual embedding model
    that returns a 128-dimensional vector.
    """
    try:
        # Load OpenCV's pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("No face detected in image")
            return None
        
        # Get the first face
        (x, y, w, h) = faces[0]
        face_roi = image[y:y+h, x:x+w]
        
        # PLACEHOLDER: Generate a simple embedding
        # TODO: Replace this with actual face embedding model (FaceNet, ArcFace, etc.)
        # For now, resize face to 128 pixels and flatten as a dummy embedding
        face_resized = cv2.resize(face_roi, (8, 16))  # Small size
        embedding = face_resized.flatten()[:128].tolist()
        
        # Normalize to 0-1 range
        embedding = [float(x) / 255.0 for x in embedding]
        
        # Pad if needed to reach 128 dimensions
        while len(embedding) < 128:
            embedding.append(0.0)
        
        print(f"✅ Face detected, generated {len(embedding)}-dimensional embedding")
        return embedding[:128]  # Ensure exactly 128 dimensions
        
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

# ==================== CORE FUNCTIONS ====================

def setup_facial_profiles():
    """
    Set up manual facial profiles.
    User uploads a facial image, extract facial features into a vector.
    Store vector in database.
    """
    # This function can be called programmatically or used as a helper
    # The actual implementation is in the /api/profiles/setup endpoint below
    pass

# ==================== API ENDPOINTS ====================

@app.route('/api/profiles/setup', methods=['POST'])
def setup_facial_profile():
    """
    Create a new facial profile.
    
    Expected form data:
    - photo: image file
    - name: person's name
    - relation: relationship (Colleague, Friend, Family, etc.)
    - metadata: optional JSON string with additional info
    
    Returns:
    {
      "success": true,
      "profile_id": "...",
      "name": "Alex Smith",
      "embedding_dimensions": 128
    }
    """
    # Validate request
    if 'photo' not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400
    
    if 'name' not in request.form:
        return jsonify({"error": "Name is required"}), 400
    
    if 'relation' not in request.form:
        return jsonify({"error": "Relation is required"}), 400
    
    photo = request.files['photo']
    name = request.form['name']
    relation = request.form['relation']
    metadata = request.form.get('metadata', {})
    
    # Validate file
    if photo.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(photo.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    # Save uploaded photo temporarily
    filename = secure_filename(f"{name}_{photo.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    photo.save(filepath)
    
    try:
        # Extract face embedding
        embedding = extract_face_embedding(filepath)
        
        if embedding is None:
            os.remove(filepath)  # Clean up
            return jsonify({"error": "No face detected in image. Please upload a clear face photo."}), 400
        
        # Store in MongoDB
        profile_id = db.insert_profile(
            name=name,
            relation=relation,
            embedding=embedding,
            metadata=metadata if isinstance(metadata, dict) else {}
        )
        
        if profile_id:
            return jsonify({
                "success": True,
                "profile_id": str(profile_id),
                "name": name,
                "relation": relation,
                "embedding_dimensions": len(embedding),
                "message": f"Profile created for {name}"
            }), 201
        else:
            os.remove(filepath)
            return jsonify({"error": "Failed to create profile. Name might already exist."}), 400
            
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    finally:
        # Optional: keep or delete the uploaded photo
        # os.remove(filepath)  # Uncomment to delete after processing
        pass


@app.route('/api/profiles/recognize', methods=['POST'])
def recognize_face():
    """
    Recognize a person from uploaded photo or embedding.
    This is what the Raspberry Pi camera will call.
    
    Expected JSON (option 1 - send embedding directly):
    {
      "embedding": [0.12, 0.34, ...],  // 128 floats
      "threshold": 0.6  // optional, default 0.6 (lower is stricter for face_recognition)
    }
    
    OR option 2 - send photo:
    Form data with 'photo' file
    
    Returns:
    {
      "recognized": true,
      "name": "Alex Smith",
      "relation": "Colleague",
      "similarity": 0.89,
      "metadata": {...}
    }
    """
    threshold = float(request.form.get('threshold', 0.6)) if request.form else 0.6
    
    # Option 1: Embedding sent directly (from Raspberry Pi)
    if request.is_json:
        data = request.json
        embedding = data.get('embedding')
        threshold = data.get('threshold', 0.6)
        
        if not embedding or len(embedding) != 128:
            return jsonify({"error": "Invalid embedding. Expected 128-dimensional vector."}), 400
    
    # Option 2: Photo uploaded
    elif 'photo' in request.files:
        photo = request.files['photo']
        
        if photo.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(photo.filename):
            return jsonify({"error": f"Invalid file type"}), 400
        
        # Save temporarily
        filename = secure_filename(f"temp_{photo.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(filepath)
        
        try:
            # Extract embedding
            embedding = extract_face_embedding(filepath)
            
            if embedding is None:
                os.remove(filepath)
                return jsonify({"error": "No face detected in image"}), 400
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        return jsonify({"error": "Either 'embedding' JSON or 'photo' file required"}), 400
    
    # Search for matching profile
    matches = db.search_by_embedding(
        query_embedding=embedding,
        threshold=threshold,
        limit=1
    )
    
    if matches and len(matches) > 0:
        best_match = matches[0]
        return jsonify({
            "recognized": True,
            "name": best_match['name'],
            "relation": best_match.get('relation'),
            "similarity": best_match['similarity'],
            "metadata": best_match.get('metadata', {}),
            "profile_id": str(best_match.get('_id', ''))
        })
    else:
        return jsonify({
            "recognized": False,
            "message": "No matching profile found"
        })


@app.route('/api/profiles', methods=['GET'])
def get_all_profiles():
    """Get all profiles or filter by relation."""
    relation = request.args.get('relation')
    profiles = db.get_all_profiles(relation=relation)
    
    # Remove embeddings from response (too large)
    for profile in profiles:
        profile.pop('embedding', None)
        profile['_id'] = str(profile.get('_id', ''))
    
    return jsonify({
        "count": len(profiles),
        "profiles": profiles
    })


@app.route('/api/profiles/<name>', methods=['GET'])
def get_profile(name):
    """Get a specific profile by name."""
    profile = db.get_profile(name)
    
    if profile:
        profile.pop('embedding', None)  # Don't send embedding in response
        profile['_id'] = str(profile.get('_id', ''))
        return jsonify(profile)
    else:
        return jsonify({"error": "Profile not found"}), 404


@app.route('/api/profiles/<name>', methods=['DELETE'])
def delete_profile(name):
    """Delete a profile."""
    success = db.delete_profile(name)
    
    if success:
        return jsonify({"success": True, "message": f"Profile '{name}' deleted"})
    else:
        return jsonify({"error": "Profile not found"}), 404


@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if API is running and database is connected."""
    try:
        # Try to ping database
        profile_count = len(db.get_all_profiles())
        return jsonify({
            "status": "healthy",
            "database": "connected",
            "profile_count": profile_count
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("Starting Assistive Memory Backend...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("API Endpoints:")
    print("  POST /api/profiles/setup - Create new facial profile")
    print("  POST /api/profiles/recognize - Recognize face from photo/embedding")
    print("  GET  /api/profiles - Get all profiles")
    print("  GET  /api/profiles/<name> - Get specific profile")
    print("  DELETE /api/profiles/<name> - Delete profile")
    print("  GET  /api/health - Health check")
    app.run(host='0.0.0.0', port=5001, debug=True)

