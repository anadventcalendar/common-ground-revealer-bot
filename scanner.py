from __future__ import annotations

import logging
from typing import Any

from analyzer import GeminiAnalyzer, ThreadSnapshot, CommentSnapshot
from poster import RedditPoster
from state import BotState, PostAttempt, canonical_thread_url


logger = logging.getLogger(__name__)


class RedditScanner:
    """Scheduled cross-subreddit scanner."""

    def __init__(
        self,
        *,
        reddit: Any,
        state: BotState,
        analyzer: GeminiAnalyzer,
        poster: RedditPoster,
        settings: Any,
    ):
        self.reddit = reddit
        self.state = state
        self.analyzer = analyzer
        self.poster = poster
        self.settings = settings

    def scan_once(self) -> None:
        logger.info(
            "Starting proactive scan across %s subreddits with query: %s",
            len(self.settings.subreddits),
            self.settings.search_query,
        )
        processed_candidates = 0

        for subreddit_name in self.settings.subreddits:
            if processed_candidates >= self.settings.max_candidates_per_scan:
                logger.info("Reached MAX_CANDIDATES_PER_SCAN=%s", self.settings.max_candidates_per_scan)
                return
            if not self.settings.dry_run and self.poster.daily_cap_reached():
                logger.warning("Daily live-post cap reached before scanning r/%s", subreddit_name)
                return

            for submission in self._search_subreddit(subreddit_name):
                if processed_candidates >= self.settings.max_candidates_per_scan:
                    return
                processed_candidates += 1
                self._process_submission(submission)

        logger.info("Scan complete. Processed %s candidate threads.", processed_candidates)

    def _search_subreddit(self, subreddit_name: str) -> list[Any]:
        logger.info("Searching r/%s via Reddit search API", subreddit_name)
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            return list(
                subreddit.search(
                    self.settings.search_query,
                    sort=self.settings.reddit_sort,
                    time_filter=self.settings.reddit_time_filter,
                    limit=self.settings.search_limit_per_subreddit,
                )
            )
        except Exception as exc:
            logger.exception("Search failed for r/%s: %s", subreddit_name, exc)
            return []

    def _process_submission(self, submission: Any) -> None:
        thread_url = canonical_thread_url(str(getattr(submission, "permalink", "")))
        thread_id = str(getattr(submission, "id", ""))
        subreddit = _submission_subreddit_name(submission)

        if not self._is_active_candidate(submission):
            return

        if self.state.has_posted_thread(thread_url):
            logger.info("Already posted in thread globally, skipping: %s", thread_url)
            return

        try:
            snapshot = self._build_thread_snapshot(submission)
        except Exception as exc:
            logger.exception("Could not build thread snapshot for %s: %s", thread_url, exc)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=thread_url,
                    thread_id=thread_id,
                    subreddit=subreddit,
                    outcome="failed_snapshot",
                    reason="Could not load thread comments.",
                    dry_run=self.settings.dry_run,
                    error_message=str(exc),
                )
            )
            return

        question = self.analyzer.identify_factual_question(snapshot)
        self.state.record_argument_pattern(
            subreddit=snapshot.subreddit,
            thread_id=snapshot.thread_id,
            thread_url=snapshot.thread_url,
            topic=question.topic,
            position_a=question.position_a,
            position_b=question.position_b,
            factual_question=question.factual_question,
            confidence=question.confidence,
            raw_summary=question.reason,
        )

        if not question.should_continue:
            logger.info("Step 1 skipped %s: %s", snapshot.thread_url, question.reason)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=snapshot.thread_url,
                    thread_id=snapshot.thread_id,
                    subreddit=snapshot.subreddit,
                    outcome="skipped_no_question",
                    reason=question.reason,
                    dry_run=self.settings.dry_run,
                )
            )
            return

        polling = self.analyzer.search_polling_data(question.factual_question)
        if not polling.found:
            logger.info("Step 2 skipped %s: %s", snapshot.thread_url, polling.reason)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=snapshot.thread_url,
                    thread_id=snapshot.thread_id,
                    subreddit=snapshot.subreddit,
                    outcome="skipped_no_polling",
                    reason=polling.reason,
                    dry_run=self.settings.dry_run,
                )
            )
            return

        reply = self.analyzer.verify_and_generate_reply(snapshot, question, polling)
        if not reply.should_post or reply.confidence < self.settings.minimum_confidence:
            reason = reply.reason or "Generated reply did not meet confidence or safety rules."
            logger.info("Step 3 skipped %s: %s", snapshot.thread_url, reason)
            self.state.record_post_attempt(
                PostAttempt(
                    thread_url=snapshot.thread_url,
                    thread_id=snapshot.thread_id,
                    subreddit=snapshot.subreddit,
                    outcome="skipped_low_confidence",
                    reason=reason,
                    dry_run=self.settings.dry_run,
                    reply_text=reply.reply_text,
                )
            )
            return

        self.poster.post_reply(
            submission,
            reply_text=reply.reply_text,
            confidence=reply.confidence,
            source_count=len(polling.sources),
        )

    def _is_active_candidate(self, submission: Any) -> bool:
        if bool(getattr(submission, "locked", False)):
            return False
        if bool(getattr(submission, "archived", False)):
            return False
        if int(getattr(submission, "num_comments", 0) or 0) < self.settings.min_thread_comments:
            return False
        if int(getattr(submission, "score", 0) or 0) < self.settings.min_thread_score:
            return False
        return True

    def _build_thread_snapshot(self, submission: Any) -> ThreadSnapshot:
        submission.comment_sort = "confidence"
        comments_root = getattr(submission, "comments", None)
        if comments_root is not None:
            comments_root.replace_more(limit=0)

        comments: list[CommentSnapshot] = []
        for comment in comments_root.list() if comments_root is not None else []:
            body = str(getattr(comment, "body", "") or "").strip()
            if not body or body in {"[deleted]", "[removed]"}:
                continue
            author = getattr(comment, "author", None)
            comments.append(
                CommentSnapshot(
                    author=str(author) if author else "[deleted]",
                    score=int(getattr(comment, "score", 0) or 0),
                    body=_compact_text(body, max_chars=900),
                )
            )
            if len(comments) >= self.settings.candidate_comment_limit:
                break

        return ThreadSnapshot(
            subreddit=_submission_subreddit_name(submission),
            thread_id=str(getattr(submission, "id", "")),
            thread_url=canonical_thread_url(str(getattr(submission, "permalink", ""))),
            title=str(getattr(submission, "title", "")).strip(),
            body=_compact_text(str(getattr(submission, "selftext", "") or ""), max_chars=1800),
            score=int(getattr(submission, "score", 0) or 0),
            comment_count=int(getattr(submission, "num_comments", 0) or 0),
            comments=comments,
        )


def _submission_subreddit_name(submission: Any) -> str:
    subreddit = getattr(submission, "subreddit", "")
    return str(getattr(subreddit, "display_name", "")) or str(subreddit)


def _compact_text(text: str, *, max_chars: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars] + "..."
