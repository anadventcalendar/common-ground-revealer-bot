from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

try:
    import praw
except ImportError:  # pragma: no cover - handled when live posting is attempted.
    praw = None

from state import BotState, PostAttempt, canonical_thread_url


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostResult:
    outcome: str
    reason: str
    reddit_comment_id: str = ""


class RedditPoster:
    """Centralized Reddit posting path with dry-run and daily cap enforcement."""

    def __init__(self, state: BotState, settings: Any):
        self.state = state
        self.settings = settings
        self._last_live_post_monotonic = 0.0

    def daily_cap_reached(self) -> bool:
        return self.state.count_live_posts_today() >= self.settings.max_daily_posts

    def post_reply(
        self,
        submission: Any,
        *,
        reply_text: str,
        confidence: float,
        source_count: int,
    ) -> PostResult:
        thread_url = _submission_thread_url(submission)
        thread_id = str(getattr(submission, "id", ""))
        subreddit = str(getattr(getattr(submission, "subreddit", ""), "display_name", "")) or str(
            getattr(submission, "subreddit", "")
        )

        if self.state.has_posted_thread(thread_url):
            reason = "This thread URL is already recorded in the global posted_threads table."
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="skipped_duplicate",
                    reason=reason,
                    dry_run=self.settings.dry_run,
                    reply_text=reply_text,
                )
            )
            logger.info("Skipping duplicate thread: %s", thread_url)
            return PostResult("skipped_duplicate", reason)

        if self.settings.dry_run:
            logger.info("DRY_RUN: would post in %s:\n%s", thread_url, reply_text)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="dry_run",
                    reason="DRY_RUN=true; no Reddit comment was submitted.",
                    dry_run=True,
                    reply_text=reply_text,
                )
            )
            return PostResult("dry_run", "DRY_RUN=true; no Reddit comment was submitted.")

        if self.daily_cap_reached():
            reason = f"Daily live-post cap reached ({self.settings.max_daily_posts})."
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="skipped_daily_cap",
                    reason=reason,
                    dry_run=False,
                    reply_text=reply_text,
                )
            )
            logger.warning(reason)
            return PostResult("skipped_daily_cap", reason)

        if praw is None:
            reason = "praw is not installed. Run: python -m pip install -r requirements.txt"
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="failed",
                    reason=reason,
                    dry_run=False,
                    reply_text=reply_text,
                    error_message=reason,
                )
            )
            return PostResult("failed", reason)

        elapsed = time.monotonic() - self._last_live_post_monotonic
        if elapsed < self.settings.min_seconds_between_live_posts:
            reason = (
                "Live post cooldown is active; skipping instead of posting too quickly. "
                f"Elapsed={elapsed:.1f}s required={self.settings.min_seconds_between_live_posts}s."
            )
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="skipped_cooldown",
                    reason=reason,
                    dry_run=False,
                    reply_text=reply_text,
                )
            )
            logger.warning(reason)
            return PostResult("skipped_cooldown", reason)

        try:
            comment = submission.reply(reply_text)
            reddit_comment_id = str(getattr(comment, "id", ""))
            self._last_live_post_monotonic = time.monotonic()
            self.state.record_posted_thread(
                thread_url=thread_url,
                thread_id=thread_id,
                subreddit=subreddit,
                reddit_comment_id=reddit_comment_id,
                reply_text=reply_text,
                source_count=source_count,
                confidence=confidence,
            )
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="posted",
                    reason="Posted successfully.",
                    dry_run=False,
                    reddit_comment_id=reddit_comment_id,
                    reply_text=reply_text,
                )
            )
            logger.info("Posted reply to %s as comment %s", thread_url, reddit_comment_id)
            return PostResult("posted", "Posted successfully.", reddit_comment_id)
        except Exception as exc:
            reason = _reddit_exception_reason(exc)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="failed",
                    reason=reason,
                    dry_run=False,
                    reply_text=reply_text,
                    error_message=str(exc),
                )
            )
            logger.exception("Reddit post failed for %s", thread_url)
            return PostResult("failed", reason)


def _submission_thread_url(submission: Any) -> str:
    permalink = getattr(submission, "permalink", "")
    if permalink:
        return canonical_thread_url(str(permalink))
    url = getattr(submission, "url", "")
    return canonical_thread_url(str(url))


def _reddit_exception_reason(exc: Exception) -> str:
    if praw is not None and isinstance(exc, praw.exceptions.RedditAPIException):
        messages = []
        for item in exc.items:
            messages.append(f"{item.error_type}: {item.message}")
        return "; ".join(messages) or str(exc)
    return str(exc)
