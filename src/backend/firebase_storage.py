"""
Firebase Cloud Storage and Firestore integration
"""
import firebase_admin
from firebase_admin import credentials, firestore, storage
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, Any
import os

logger = logging.getLogger(__name__)


class FirebaseStorage:
    """Firebase Cloud Storage and Firestore operations"""
    
    _initialized = False
    _db = None
    _bucket = None
    
    @classmethod
    def initialize(cls):
        """Initialize Firebase app once"""
        if cls._initialized:
            return
        
        try:
            # Method 1: Try credentials from environment variable (JSON string)
            creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
            if creds_json:
                try:
                    creds_dict = json.loads(creds_json)
                    cred = credentials.Certificate(creds_dict)
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase initialized with FIREBASE_CREDENTIALS_JSON env var")
                except json.JSONDecodeError as je:
                    logger.warning(f"Invalid JSON in FIREBASE_CREDENTIALS_JSON: {je}")
                    raise
            
            # Method 2: Try to initialize with credentials file
            else:
                creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
                if creds_path and Path(creds_path).exists():
                    cred = credentials.Certificate(creds_path)
                    firebase_admin.initialize_app(cred)
                    logger.info("Firebase initialized with credentials file")
                else:
                    # Method 3: Try default credentials
                    firebase_admin.initialize_app()
                    logger.info("Firebase initialized with default credentials")
            
            cls._db = firestore.client()
            cls._bucket = storage.bucket()
            cls._initialized = True
            logger.info("Firebase Storage and Firestore ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    @staticmethod
    async def upload_snapshot(image_path: str, search_id: str, match_id: str) -> dict:
        """Upload snapshot to Firebase Storage"""
        try:
            FirebaseStorage.initialize()
            
            image_path = Path(image_path)
            if not image_path.exists():
                return {"success": False, "error": "Image file not found"}
            
            # Upload to Firebase Storage
            blob_path = f"drishti/snapshots/{search_id}/{match_id}.jpg"
            blob = FirebaseStorage._bucket.blob(blob_path)
            blob.upload_from_filename(str(image_path))
            
            # Make it publicly accessible
            blob.make_public()
            
            logger.info(f"Snapshot uploaded to Firebase: {blob_path}")
            
            return {
                "success": True,
                "url": blob.public_url,
                "path": blob_path
            }
        except Exception as e:
            logger.error(f"Failed to upload snapshot: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def save_search_results(results: dict, search_id: str) -> dict:
        """Save search results to Firestore"""
        try:
            FirebaseStorage.initialize()
            
            # Prepare document data
            doc_data = {
                "search_id": search_id,
                "timestamp": datetime.now(),
                "matches": results.get("matches", []),
                "stats": results.get("stats", {}),
                "summary": results.get("summary", {}),
                "status": results.get("status", "completed")
            }
            
            # Save to Firestore
            doc_ref = FirebaseStorage._db.collection("searches").document(search_id)
            doc_ref.set(doc_data)
            
            # Set TTL to 30 days (optional cleanup)
            # Firestore will auto-delete after expiry_time
            doc_ref.update({
                "expiry_time": datetime.now() + timedelta(days=30)
            })
            
            logger.info(f"Results saved to Firestore: {search_id}")
            
            return {
                "success": True,
                "search_id": search_id,
                "url": f"/api/results/{search_id}"
            }
        except Exception as e:
            logger.error(f"Failed to save results to Firestore: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def get_search_results(search_id: str) -> dict:
        """Retrieve search results from Firestore"""
        try:
            FirebaseStorage.initialize()
            
            doc = FirebaseStorage._db.collection("searches").document(search_id).get()
            
            if not doc.exists:
                return {"success": False, "error": "Search results not found"}
            
            data = doc.to_dict()
            
            # Convert timestamp to ISO format
            if "timestamp" in data and hasattr(data["timestamp"], "isoformat"):
                data["timestamp"] = data["timestamp"].isoformat()
            
            logger.info(f"Retrieved results from Firestore: {search_id}")
            
            return {
                "success": True,
                "search_id": search_id,
                "data": data
            }
        except Exception as e:
            logger.error(f"Failed to retrieve results: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    async def delete_search_results(search_id: str) -> bool:
        """Delete search results from Firestore and Storage"""
        try:
            FirebaseStorage.initialize()
            
            # Delete from Firestore
            FirebaseStorage._db.collection("searches").document(search_id).delete()
            
            # Delete snapshots from Storage
            blobs = FirebaseStorage._bucket.list_blobs(prefix=f"drishti/snapshots/{search_id}/")
            for blob in blobs:
                blob.delete()
            
            logger.info(f"Deleted results for search_id: {search_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete results: {e}")
            return False
    
    @staticmethod
    async def get_all_searches() -> dict:
        """Get all search results (paginated)"""
        try:
            FirebaseStorage.initialize()
            
            docs = FirebaseStorage._db.collection("searches").limit(50).stream()
            
            results = []
            for doc in docs:
                data = doc.to_dict()
                if "timestamp" in data and hasattr(data["timestamp"], "isoformat"):
                    data["timestamp"] = data["timestamp"].isoformat()
                results.append(data)
            
            return {
                "success": True,
                "total": len(results),
                "results": results
            }
        except Exception as e:
            logger.error(f"Failed to get all searches: {e}")
            return {"success": False, "error": str(e), "results": []}
