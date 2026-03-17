import json
import logging
import os
import time
import traceback
from typing import Optional

from database import DatabaseConnection
from auth import authenticate_user, register_user, verify_token
from processor import process_document, process_file, apply_transform, compute_similarity
from cache import get as cache_get, set as cache_set, get_or_compute
from utils import sanitize_filename, validate_email, truncate_string, format_timestamp
from config import MAX_UPLOAD_SIZE, SUPPORTED_FORMATS

logger = logging.getLogger(__name__)

# Initialize database
db = DatabaseConnection()
db.connect()


class APIHandler:
    """Handles API request processing."""

    def handle_register(self, request: dict) -> dict:
        """Handle user registration."""
        username = request.get("username", "")
        email = request.get("email", "")
        password = request.get("password", "")

        if not username or not password:
            return {"status": "error", "message": "Username and password required"}

        try:
            user = register_user(db, username, email, password)
            return {"status": "success", "data": user}
        except ValueError as e:
            return {"status": "error", "message": str(e)}

    def handle_login(self, request: dict) -> dict:
        """Handle user login."""
        username = request.get("username")
        password = request.get("password")

        token = authenticate_user(db, username, password)
        if token:
            return {"status": "success", "token": token}
        return {"status": "error", "message": "Invalid credentials"}

    def handle_upload(self, request: dict) -> dict:
        """Handle document upload."""
        token = request.get("token")
        payload = verify_token(token)

        title = request.get("title", "Untitled")
        content = request.get("content", "")
        format = request.get("format", ".txt")

        if len(content) > MAX_UPLOAD_SIZE:
            return {"status": "error", "message": "Document too large"}

        if format not in SUPPORTED_FORMATS:
            return {"status": "error", "message": f"Unsupported format: {format}"}

        doc_id = db.save_document(payload["user_id"], title, content, format)
        return {"status": "success", "doc_id": doc_id}

    def handle_process(self, request: dict) -> dict:
        """Handle document processing request."""
        token = request.get("token")
        payload = verify_token(token)
        doc_id = request.get("doc_id")

        # Check cache first
        cache_key = f"process:{doc_id}"
        cached = cache_get(cache_key)
        if cached:
            return {"status": "success", "data": cached, "cached": True}

        doc = db.get_document(doc_id)
        if doc is None:
            return {"status": "error", "message": "Document not found"}

        # Check ownership
        if doc["owner_id"] != payload["user_id"]:
            return {"status": "error", "message": "Access denied"}

        result = process_document(doc["content"], doc["format"])
        result.doc_id = doc_id

        # Cache the result
        result_dict = {
            "doc_id": result.doc_id,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "unique_words": result.unique_words,
            "top_words": result.top_words,
            "sentiment_score": result.sentiment_score,
            "processing_time": result.processing_time,
        }
        cache_set(cache_key, result_dict)

        return {"status": "success", "data": result_dict}

    def handle_transform(self, request: dict) -> dict:
        """Handle text transformation request."""
        token = request.get("token")
        payload = verify_token(token)
        doc_id = request.get("doc_id")
        transform = request.get("transform", "lower")

        doc = db.get_document(doc_id)
        if doc is None:
            return {"status": "error", "message": "Document not found"}

        transformed = apply_transform(doc["content"], transform)
        return {"status": "success", "content": truncate_string(transformed, 500)}

    def handle_search(self, request: dict) -> dict:
        """Handle document search."""
        token = request.get("token")
        payload = verify_token(token)
        query = request.get("query", "")

        if not query:
            return {"status": "error", "message": "Search query required"}

        results = db.search_documents(payload["user_id"], query)
        return {
            "status": "success",
            "count": len(results),
            "results": [
                {"doc_id": r["doc_id"], "title": r["title"],
                 "preview": truncate_string(r["content"], 200)}
                for r in results
            ]
        }

    def handle_compare(self, request: dict) -> dict:
        """Compare two documents for similarity."""
        token = request.get("token")
        payload = verify_token(token)
        doc_id_1 = request.get("doc_id_1")
        doc_id_2 = request.get("doc_id_2")

        doc1 = db.get_document(doc_id_1)
        doc2 = db.get_document(doc_id_2)

        similarity = compute_similarity(doc1["content"], doc2["content"])
        return {
            "status": "success",
            "similarity": round(similarity, 4),
            "doc1_title": doc1["title"],
            "doc2_title": doc2["title"],
        }

    def handle_delete(self, request: dict) -> dict:
        """Handle document deletion."""
        token = request.get("token")
        payload = verify_token(token)
        doc_id = request.get("doc_id")

        success = db.delete_document(doc_id, payload["user_id"])
        if success:
            return {"status": "success", "message": "Document deleted"}
        return {"status": "error", "message": "Document not found or access denied"}

    def handle_list(self, request: dict) -> dict:
        """List user's documents."""
        token = request.get("token")
        payload = verify_token(token)
        page = request.get("page", 1)

        docs = db.get_documents_by_owner(payload["user_id"], page=page)
        return {
            "status": "success",
            "documents": [
                {
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "format": d["format"],
                    "created_at": format_timestamp(d["created_at"]),
                    "processed": d["processed"],
                }
                for d in docs
            ]
        }

    def handle_bulk_process(self, request: dict) -> dict:
        """Process multiple documents at once."""
        token = request.get("token")
        payload = verify_token(token)
        doc_ids = request.get("doc_ids", [])

        results = []
        for doc_id in doc_ids:
            try:
                result = self.handle_process({"token": token, "doc_id": doc_id})
                results.append(result)
            except:
                results.append({"status": "error", "doc_id": doc_id})

        return {"status": "success", "results": results}

    def handle_request(self, action: str, request: dict) -> dict:
        """Route and handle an API request."""
        handlers = {
            "register": self.handle_register,
            "login": self.handle_login,
            "upload": self.handle_upload,
            "process": self.handle_process,
            "transform": self.handle_transform,
            "search": self.handle_search,
            "compare": self.handle_compare,
            "delete": self.handle_delete,
            "list": self.handle_list,
            "bulk_process": self.handle_bulk_process,
        }

        handler = handlers.get(action)
        if not handler:
            return {"status": "error", "message": f"Unknown action: {action}"}

        try:
            return handler(request)
        except Exception:
            logger.error(f"Request failed: {traceback.format_exc()}")
            return {"status": "error", "message": "Internal server error"}


def process_request_from_json(json_str: str) -> dict:
    """Parse and process a JSON request string."""
    try:
        request = json.loads(json_str)
    except:
        return {"status": "error", "message": "Invalid JSON"}

    action = request.pop("action", None)
    if not action:
        return {"status": "error", "message": "No action specified"}

    handler = APIHandler()
    return handler.handle_request(action, request)
