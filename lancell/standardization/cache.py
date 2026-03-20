"""SQLite-backed persistent cache for API results.

Keyed by (resolver_name, input_value, namespace). TTL-based expiration.
Location: ``~/.cache/lancell/standardization.db``
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# TODO: Consider moving this cache to S3 for a persistent multi-user cache.
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lancell"
DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 days


@dataclass
class CacheEntry:
    resolver: str
    key: str
    namespace: str
    value: dict
    created_at: float
    ttl: float


class StandardizationCache:
    """Thread-safe SQLite cache with WAL mode."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        default_ttl: float = DEFAULT_TTL_SECONDS,
    ):
        if db_path is None:
            db_path = DEFAULT_CACHE_DIR / "standardization.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                resolver TEXT NOT NULL,
                key TEXT NOT NULL,
                namespace TEXT NOT NULL DEFAULT '',
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl REAL NOT NULL,
                PRIMARY KEY (resolver, key, namespace)
            )
            """
        )
        self._conn.commit()

    def get(self, resolver: str, key: str, namespace: str = "") -> CacheEntry | None:
        """Return a cached entry if it exists and has not expired."""
        row = self._conn.execute(
            "SELECT value, created_at, ttl FROM cache WHERE resolver=? AND key=? AND namespace=?",
            (resolver, key, namespace),
        ).fetchone()
        if row is None:
            return None
        value_json, created_at, ttl = row
        if time.time() - created_at > ttl:
            # Expired — delete and return None
            self._conn.execute(
                "DELETE FROM cache WHERE resolver=? AND key=? AND namespace=?",
                (resolver, key, namespace),
            )
            self._conn.commit()
            return None
        return CacheEntry(
            resolver=resolver,
            key=key,
            namespace=namespace,
            value=json.loads(value_json),
            created_at=created_at,
            ttl=ttl,
        )

    def put(
        self,
        resolver: str,
        key: str,
        value: dict,
        namespace: str = "",
        ttl: float | None = None,
    ) -> None:
        """Insert or replace a cache entry."""
        if ttl is None:
            ttl = self._default_ttl
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache (resolver, key, namespace, value, created_at, ttl)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (resolver, key, namespace, json.dumps(value), time.time(), ttl),
        )
        self._conn.commit()

    def clear(self, resolver: str | None = None) -> int:
        """Delete cached entries. If *resolver* is given, delete only that resolver's entries.

        Returns the number of rows deleted.
        """
        if resolver is not None:
            cur = self._conn.execute("DELETE FROM cache WHERE resolver=?", (resolver,))
        else:
            cur = self._conn.execute("DELETE FROM cache")
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()


# Module-level singleton for convenience
_default_cache: StandardizationCache | None = None


def get_cache() -> StandardizationCache:
    """Return (and lazily create) the module-level default cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = StandardizationCache()
    return _default_cache
