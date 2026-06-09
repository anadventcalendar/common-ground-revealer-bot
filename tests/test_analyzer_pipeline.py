from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyzer import (
    AnalysisResult,
    CommentSnapshot,
    FactualQuestionResult,
    GeminiAnalyzer,
    GeneratedReplyResult,
    PollingDataResult,
    Source,
    ThreadSnapshot,
)


def sample_thread():
    return ThreadSnapshot(
        subreddit="politics",
        thread_id="abc",
        thread_url="https://www.reddit.com/r/politics/comments/abc/example",
        title="Policy argument",
        body="",
        score=10,
        comment_count=100,
        comments=[CommentSnapshot(author="user", score=5, body="People disagree about support.")],
    )


class NoPollingAnalyzer(GeminiAnalyzer):
    def __init__(self):
        self.minimum_confidence = 0.78
        self.minimum_relevance = 0.70

    def identify_factual_question(self, thread):
        return FactualQuestionResult(
            should_continue=True,
            topic="policy",
            position_a="one view",
            position_b="opposing view",
            factual_question="Do voters support this policy?",
            confidence=0.9,
        )

    def search_polling_data(self, factual_question):
        return PollingDataResult(found=False, reason="No directly relevant polling found.")

    def verify_and_generate_reply(self, thread, question, polling):
        raise AssertionError("Step 3 should not run when Step 2 finds no polling")


class LowConfidenceAnalyzer(NoPollingAnalyzer):
    def search_polling_data(self, factual_question):
        return PollingDataResult(
            found=True,
            factual_question=factual_question,
            confidence=0.9,
            relevance_score=0.9,
            sources=[Source(title="Poll", url="https://example.com/poll")],
            raw_json={"found": True},
        )

    def verify_and_generate_reply(self, thread, question, polling):
        return GeneratedReplyResult(
            should_post=True,
            reply_text="A sourced neutral reply: https://example.com/poll",
            confidence=0.5,
        )


class AnalyzerPipelineTests(unittest.TestCase):
    def test_no_polling_data_means_no_post(self):
        result = NoPollingAnalyzer().analyze_thread(sample_thread())

        self.assertIsInstance(result, AnalysisResult)
        self.assertFalse(result.should_post)
        self.assertIn("polling", result.reason.lower())

    def test_low_confidence_reply_means_no_post(self):
        result = LowConfidenceAnalyzer().analyze_thread(sample_thread())

        self.assertFalse(result.should_post)
        self.assertIn("below", result.reason.lower())


if __name__ == "__main__":
    unittest.main()
