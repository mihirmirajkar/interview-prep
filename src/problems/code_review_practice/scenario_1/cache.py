import time
import threading
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Global cache storage
_cache = {}
_cache_timestamps = {}
_lock = threading.Lock()

DEFAULT_TTL = 300  # 5 minutes


def get(key: str) -> Optional[Any]:
    """Get a value from cache."""
    if key in _cache:
        timestamp = _cache_timestamps.get(key, 0)
        if time.time() - timestamp > DEFAULT_TTL:
            # Expired
            del _cache[key]
            del _cache_timestamps[key]
            return None
        return _cache[key]
    return None


def set(key: str, value: Any, ttl: int = DEFAULT_TTL):
    """Set a value in cache."""
    _cache[key] = value
    _cache_timestamps[key] = time.time()


def delete(key: str) -> bool:
    """Delete a value from cache."""
    if key in _cache:
        del _cache[key]
        if key in _cache_timestamps:
            del _cache_timestamps[key]
        return True
    return False


def get_or_compute(key: str, compute_fn, ttl: int = DEFAULT_TTL) -> Any:
    """Get from cache or compute and store the value."""
    value = get(key)
    if value is not None:
        return value

    # Compute the value
    value = compute_fn()
    set(key, value, ttl)
    return value


def clear():
    """Clear all cache entries."""
    _cache.clear()
    _cache_timestamps.clear()


def get_stats() -> dict:
    """Get cache statistics."""
    total = len(_cache)
    expired = 0
    now = time.time()
    for key, ts in _cache_timestamps.items():
        if now - ts > DEFAULT_TTL:
            expired += 1

    return {
        "total_entries": total,
        "expired_entries": expired,
        "active_entries": total - expired,
    }


def cleanup_expired():
    """Remove expired entries from cache."""
    now = time.time()
    expired_keys = []
    for key, ts in _cache_timestamps.items():
        if now - ts > DEFAULT_TTL:
            expired_keys.append(key)

    for key in expired_keys:
        del _cache[key]
        del _cache_timestamps[key]

    logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
    return len(expired_keys)


class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.access_order = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.access_order.append(key)

    def size(self) -> int:
        return len(self.cache)
