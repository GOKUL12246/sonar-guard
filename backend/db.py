"""
SONAR-GUARD — MongoDB Database Manager
======================================
Connects to MongoDB Atlas for cloud persistence of:
  - Analyzed acoustic sonar contacts (WGS-84 coordinates, dimensions, risk bands)
  - Operator human-in-the-loop verification decisions & active learning logs
  - Mission telemetry & sonar survey logs
"""

import os
from typing import Dict, List, Optional
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# MongoDB Atlas URI (Configurable via Environment Variable)
DEFAULT_MONGO_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://admin:admin123@cluster0.wirpb6n.mongodb.net/sonar_guard_db?retryWrites=true&w=majority&appName=Cluster0"
)

class MongoDBManager:
    def __init__(self, uri: str = DEFAULT_MONGO_URI):
        self.uri = uri
        self.client: Optional[pymongo.MongoClient] = None
        self.db = None
        self.connected = False
        self._init_connection()

    def _init_connection(self):
        try:
            self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=4000)
            # Ping database to verify credentials & network
            self.client.admin.command('ping')
            self.db = self.client["sonar_guard_db"]
            self.connected = True
            print("Successfully connected to MongoDB Atlas (sonar_guard_db)")
        except Exception as exc:
            self.connected = False
            print(f"MongoDB connection notice: Running in local fallback mode ({str(exc)[:90]})")

    def save_contact(self, contact: dict) -> bool:
        """Insert or update an analyzed sonar contact in MongoDB Atlas."""
        if not self.connected or self.db is None:
            return False
        try:
            contacts_col = self.db["contacts"]
            anomaly_id = contact.get("anomaly_id")
            if anomaly_id:
                contacts_col.update_one(
                    {"anomaly_id": anomaly_id},
                    {"$set": contact},
                    upsert=True
                )
            else:
                contacts_col.insert_one(contact)
            return True
        except Exception as exc:
            print(f"MongoDB save_contact error: {exc}")
            return False

    def get_contacts(self, limit: int = 500) -> List[dict]:
        """Fetch all analyzed contacts from MongoDB Atlas."""
        if not self.connected or self.db is None:
            return []
        try:
            contacts_col = self.db["contacts"]
            cursor = contacts_col.find({}, {"_id": 0}).sort("timestamp_utc", pymongo.DESCENDING).limit(limit)
            return list(cursor)
        except Exception as exc:
            print(f"MongoDB get_contacts error: {exc}")
            return []

    def save_feedback(self, feedback: dict) -> bool:
        """Save operator verification decision for active learning."""
        if not self.connected or self.db is None:
            return False
        try:
            feedback_col = self.db["feedback"]
            feedback_col.insert_one(feedback)
            # Also update contact status in contacts collection
            anomaly_id = feedback.get("anomaly_id")
            decision = feedback.get("user_decision")
            if anomaly_id and decision:
                self.db["contacts"].update_one(
                    {"anomaly_id": anomaly_id},
                    {"$set": {"verification_status": decision, "notes": feedback.get("user_notes", "")}}
                )
            return True
        except Exception as exc:
            print(f"MongoDB save_feedback error: {exc}")
            return False

    def get_status(self) -> dict:
        """Return MongoDB Atlas health and collection counts."""
        if not self.connected or self.db is None:
            return {"status": "offline_fallback", "connected": False}
        try:
            contacts_count = self.db["contacts"].count_documents({})
            feedback_count = self.db["feedback"].count_documents({})
            return {
                "status": "connected",
                "connected": True,
                "database": "sonar_guard_db",
                "collections": {
                    "contacts": contacts_count,
                    "feedback": feedback_count,
                }
            }
        except Exception:
            return {"status": "connected", "connected": True, "database": "sonar_guard_db"}

# Singleton instance
mongo_db = MongoDBManager()
