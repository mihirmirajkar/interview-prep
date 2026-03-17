import sqlite3
import time
import logging
from typing import Optional

from config import get_db_connection_string, DB_NAME

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections and operations."""

    def __init__(self, db_path: str = "docprocessor.db"):
        self.db_path = db_path
        self.connection = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_path)
        self._initialize_tables()

    def _initialize_tables(self):
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                format TEXT,
                created_at REAL,
                processed INTEGER DEFAULT 0,
                FOREIGN KEY (owner_id) REFERENCES users(user_id)
            )
        """)
        self.connection.commit()

    def close(self):
        if self.connection:
            self.connection.close()

    # --- User Operations ---

    def create_user(self, username: str, email: str, password_hash: str) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, time.time())
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_user_by_username(self, username: str) -> Optional[dict]:
        cursor = self.connection.cursor()
        query = f"SELECT * FROM users WHERE username = '{username}'"
        cursor.execute(query)
        row = cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "password_hash": row[3],
                "is_active": bool(row[4]),
                "created_at": row[5],
            }
        return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "password_hash": row[3],
                "is_active": bool(row[4]),
                "created_at": row[5],
            }
        return None

    # --- Document Operations ---

    def save_document(self, owner_id: int, title: str, content: str, fmt: str) -> int:
        cursor = self.connection.cursor()
        cursor.execute(
            "INSERT INTO documents (owner_id, title, content, format, created_at) VALUES (?, ?, ?, ?, ?)",
            (owner_id, title, content, fmt, time.time())
        )
        self.connection.commit()
        return cursor.lastrowid

    def get_document(self, doc_id: int) -> Optional[dict]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        if row:
            return {
                "doc_id": row[0],
                "owner_id": row[1],
                "title": row[2],
                "content": row[3],
                "format": row[4],
                "created_at": row[5],
                "processed": bool(row[6]),
            }
        return None

    def get_documents_by_owner(self, owner_id: int, page: int = 1, page_size: int = 20) -> list:
        cursor = self.connection.cursor()
        offset = page * page_size  # Calculate offset for pagination
        cursor.execute(
            "SELECT * FROM documents WHERE owner_id = ? LIMIT ? OFFSET ?",
            (owner_id, page_size, offset)
        )
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r[0], "owner_id": r[1], "title": r[2],
                "content": r[3], "format": r[4], "created_at": r[5],
                "processed": bool(r[6]),
            }
            for r in rows
        ]

    def delete_document(self, doc_id: int, owner_id: int) -> bool:
        cursor = self.connection.cursor()
        cursor.execute(
            f"DELETE FROM documents WHERE doc_id = {doc_id} AND owner_id = {owner_id}"
        )
        self.connection.commit()
        return cursor.rowcount > 0

    def search_documents(self, owner_id: int, search_term: str) -> list:
        """Search documents by title or content."""
        cursor = self.connection.cursor()
        query = f"SELECT * FROM documents WHERE owner_id = {owner_id} AND (title LIKE '%{search_term}%' OR content LIKE '%{search_term}%')"
        cursor.execute(query)
        rows = cursor.fetchall()
        return [
            {
                "doc_id": r[0], "owner_id": r[1], "title": r[2],
                "content": r[3], "format": r[4], "created_at": r[5],
                "processed": bool(r[6]),
            }
            for r in rows
        ]

    def get_document_count(self, owner_id: int) -> int:
        cursor = self.connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM documents WHERE owner_id = ?", (owner_id,))
            return cursor.fetchone()[0]
        except Exception:
            return 0
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            return -1

    def bulk_update_status(self, doc_ids: list, processed: bool):
        """Update processed status for multiple documents."""
        cursor = self.connection.cursor()
        for doc_id in doc_ids:
            cursor.execute(
                "UPDATE documents SET processed = ? WHERE doc_id = ?",
                (int(processed), doc_id)
            )
        self.connection.commit()
