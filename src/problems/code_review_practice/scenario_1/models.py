import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class User:
    user_id: int
    username: str
    email: str
    password_hash: str
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    permissions: list = []

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions


@dataclass
class Document:
    doc_id: int
    owner_id: int
    title: str
    content: str
    format: str
    metadata: dict = {}
    created_at: float = field(default_factory=time.time)
    processed: bool = False
    tags: list = field(default_factory=list)

    def word_count(self) -> int:
        return len(self.content.split())

    def summary(self, max_length: int = 200) -> str:
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


@dataclass
class ProcessingResult:
    doc_id: int
    status: str
    word_count: int
    char_count: int
    unique_words: int
    top_words: list = field(default_factory=list)
    sentiment_score: float = 0.0
    processing_time: float = 0.0
    errors: list = field(default_factory=list)


class DocumentCollection:
    """Manages a collection of documents with basic operations."""

    def __init__(self):
        self._documents = {}
        self._index = {}

    def add(self, doc: Document):
        self._documents[doc.doc_id] = doc
        words = doc.content.lower().split()
        for word in words:
            if word not in self._index:
                self._index[word] = set()
            self._index[word].add(doc.doc_id)

    def remove(self, doc_id: int) -> bool:
        if doc_id in self._documents:
            del self._documents[doc_id]
            return True
        return False

    def search(self, query: str) -> list:
        """Search documents by keyword."""
        results = []
        terms = query.lower().split()
        if not terms:
            return results

        matching_ids = self._index.get(terms[0], set())
        for term in terms[1:]:
            matching_ids = matching_ids & self._index.get(term, set())

        for doc_id in matching_ids:
            results.append(self._documents[doc_id])
        return results

    def get_all(self) -> list:
        return list(self._documents.values())
