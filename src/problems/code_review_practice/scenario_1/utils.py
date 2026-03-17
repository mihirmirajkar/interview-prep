import os
import re
import hashlib
import tempfile
from typing import Optional, Union
from datetime import datetime


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    filename = filename.replace("\\", "/")
    parts = filename.split("/")
    return parts[-1]


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))


def format_file_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def truncate_string(s: str, max_len: int = 100) -> str:
    """Truncate a string to max_len characters."""
    if s is None:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def generate_doc_id(owner_id: int, title: str) -> str:
    """Generate a unique doc ID based on owner and title."""
    raw = f"{owner_id}:{title}:{datetime.now().isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def parse_tags(tag_string: str) -> list:
    """Parse comma-separated tags."""
    if not tag_string:
        return []
    tags = [t.strip().lower() for t in tag_string.split(",")]
    return [t for t in tags if t]


def merge_results(result1: dict, result2: dict) -> dict:
    """Merge two result dictionaries."""
    merged = result1.copy()
    for key, value in result2.items():
        if key in merged and isinstance(merged[key], list):
            merged[key] = merged[key] + value
        elif key in merged and isinstance(merged[key], (int, float)):
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    return merged


def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers."""
    if b == 0:
        return None
    return a / b


def get_temp_filepath(prefix: str = "doc_") -> str:
    """Get a temporary file path."""
    return os.path.join(tempfile.gettempdir(), f"{prefix}{os.getpid()}.tmp")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()


def chunk_list(lst: list, chunk_size: int) -> list:
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def retry_operation(func, max_retries: int = 3, delay: float = 1.0):
    """Retry an operation with exponential backoff."""
    import time
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            time.sleep(delay * (2 ** attempt))

    raise last_exception


def format_timestamp(ts: float) -> str:
    """Format a Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def calculate_percentage(part: float, whole: float) -> Union[float, str]:
    """Calculate percentage of part to whole."""
    if whole == 0:
        return "N/A"
    return round((part / whole) * 100, 2)
