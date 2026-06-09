from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poster import RedditPoster
from state import BotState, PostAttempt


class FakeSubreddit:
    display_name = "politics"


class FakeSubmission:
    id = "abc"
    permalink = "/r/politics/comments/abc/example"
    subreddit = FakeSubreddit()

    def __init__(self):
        self.reply_called = False

    def reply(self, text):
        self.reply_called = True
        raise AssertionError("reply should not be called in this test")


class RedditPosterTests(unittest.TestCase):
    def make_state(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        state = BotState(Path(tmpdir.name) / "bot.sqlite3", ROOT / "schema.sql")
        state.initialize()
        return state

    def settings(self, dry_run):
        return SimpleNamespace(
            dry_run=dry_run,
            max_daily_posts=1,
            min_seconds_between_live_posts=60,
        )

    def test_dry_run_records_attempt_without_replying(self):
        state = self.make_state()
        submission = FakeSubmission()
        poster = RedditPoster(state, self.settings(dry_run=True))

        result = poster.post_reply(
            submission,
            reply_text="A sourced neutral reply: https://example.com",
            confidence=0.9,
            source_count=1,
        )

        self.assertEqual(result.outcome, "dry_run")
        self.assertFalse(submission.reply_called)
        attempts = state.recent_post_attempts()
        self.assertEqual(attempts[0]["outcome"], "dry_run")

    def test_daily_cap_blocks_live_post_before_replying(self):
        state = self.make_state()
        state.record_post_attempt(
            PostAttempt(
                thread_url="https://www.reddit.com/r/politics/comments/old/example",
                thread_id="old",
                subreddit="politics",
                outcome="posted",
                dry_run=False,
            )
        )
        submission = FakeSubmission()
        poster = RedditPoster(state, self.settings(dry_run=False))

        result = poster.post_reply(
            submission,
            reply_text="A sourced neutral reply: https://example.com",
            confidence=0.9,
            source_count=1,
        )

        self.assertEqual(result.outcome, "skipped_daily_cap")
        self.assertFalse(submission.reply_called)


if __name__ == "__main__":
    unittest.main()
