from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit


BASE_DIR = Path(__file__).resolve().parent


def canonical_thread_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return value
    if value.startswith("/"):
        value = f"https://www.reddit.com{value}"
    parts = urlsplit(value)
    normalized = urlunsplit((parts.scheme or "https", parts.netloc or "www.reddit.com", parts.path, "", ""))
    return normalized.rstrip("/")


@dataclass(frozen=True)
class PostAttempt:
    thread_url: str
    thread_id: str
    subreddit: str
    outcome: str
    reason: str = ""
    dry_run: bool = True
    reddit_comment_id: str = ""
    reply_text: str = ""
    error_message: str = ""


class BotState:
    """SQLite-backed cross-community memory for duplicate prevention and audit logs."""

    def __init__(self, database_path: Path, schema_path: Path | None = None):
        self.database_path = Path(database_path)
        self.schema_path = schema_path or BASE_DIR / "schema.sql"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        schema = self.schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema)

    def has_posted_thread(self, thread_url: str) -> bool:
        normalized = canonical_thread_url(thread_url)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM posted_threads WHERE thread_url = ? LIMIT 1",
                (normalized,),
            ).fetchone()
        return row is not None

    def record_argument_pattern(
        self,
        *,
        subreddit: str,
        thread_id: str,
        thread_url: str,
        topic: str,
        position_a: str,
        position_b: str,
        factual_question: str,
        confidence: float,
        raw_summary: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO argument_patterns (
                    subreddit,
                    thread_id,
                    thread_url,
                    topic,
                    position_a,
                    position_b,
                    factual_question,
                    confidence,
                    raw_summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subreddit,
                    thread_id,
                    canonical_thread_url(thread_url),
                    topic,
                    position_a,
                    position_b,
                    factual_question,
                    confidence,
                    raw_summary,
                ),
            )

    def record_post_attempt(self, attempt: PostAttempt) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO post_history (
                    thread_url,
                    thread_id,
                    subreddit,
                    outcome,
                    reason,
                    dry_run,
                    reddit_comment_id,
                    reply_text,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_thread_url(attempt.thread_url),
                    attempt.thread_id,
                    attempt.subreddit,
                    attempt.outcome,
                    attempt.reason,
                    1 if attempt.dry_run else 0,
                    attempt.reddit_comment_id,
                    attempt.reply_text,
                    attempt.error_message,
                ),
            )

    def record_posted_thread(
        self,
        *,
        thread_url: str,
        thread_id: str,
        subreddit: str,
        reddit_comment_id: str,
        reply_text: str,
        source_count: int,
        confidence: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO posted_threads (
                    thread_url,
                    thread_id,
                    subreddit,
                    reddit_comment_id,
                    dry_run,
                    reply_text,
                    source_count,
                    confidence,
                    status
                )
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, 'posted')
                ON CONFLICT(thread_url) DO NOTHING
                """,
                (
                    canonical_thread_url(thread_url),
                    thread_id,
                    subreddit,
                    reddit_comment_id,
                    reply_text,
                    source_count,
                    confidence,
                ),
            )

    def count_live_posts_today(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM post_history
                WHERE outcome = 'posted'
                  AND dry_run = 0
                  AND date(attempted_at) = date('now')
                """
            ).fetchone()
        return int(row["count"] if row else 0)

    def recent_post_attempts(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM post_history
                ORDER BY attempted_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return list(rows)
