from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from state import BotState, PostAttempt, canonical_thread_url


class BotStateTests(unittest.TestCase):
    def make_state(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        state = BotState(Path(tmpdir.name) / "bot.sqlite3", ROOT / "schema.sql")
        state.initialize()
        return state

    def test_canonical_thread_url_removes_query_and_trailing_slash(self):
        self.assertEqual(
            canonical_thread_url("https://www.reddit.com/r/politics/comments/abc/title/?utm_source=x"),
            "https://www.reddit.com/r/politics/comments/abc/title",
        )

    def test_posted_thread_is_global_not_subreddit_scoped(self):
        state = self.make_state()
        url = "https://www.reddit.com/r/politics/comments/abc/example"
        state.record_posted_thread(
            thread_url=url,
            thread_id="abc",
            subreddit="politics",
            reddit_comment_id="comment1",
            reply_text="hello",
            source_count=1,
            confidence=0.9,
        )

        self.assertTrue(state.has_posted_thread(url))
        self.assertTrue(state.has_posted_thread(url + "/?context=3"))

    def test_live_daily_count_ignores_dry_run_attempts(self):
        state = self.make_state()
        state.record_post_attempt(
            PostAttempt(
                thread_url="https://www.reddit.com/r/news/comments/abc/example",
                thread_id="abc",
                subreddit="news",
                outcome="dry_run",
                dry_run=True,
            )
        )
        self.assertEqual(state.count_live_posts_today(), 0)

        state.record_post_attempt(
            PostAttempt(
                thread_url="https://www.reddit.com/r/news/comments/def/example",
                thread_id="def",
                subreddit="news",
                outcome="posted",
                dry_run=False,
            )
        )
        self.assertEqual(state.count_live_posts_today(), 1)


if __name__ == "__main__":
    unittest.main()
