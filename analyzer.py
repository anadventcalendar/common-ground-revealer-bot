from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError:  # pragma: no cover - validated when GeminiAnalyzer is created.
    genai = None
    types = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommentSnapshot:
    author: str
    score: int
    body: str


@dataclass(frozen=True)
class ThreadSnapshot:
    subreddit: str
    thread_id: str
    thread_url: str
    title: str
    body: str
    score: int
    comment_count: int
    comments: list[CommentSnapshot]

    def to_prompt_text(self, max_chars: int = 12000) -> str:
        lines = [
            f"Subreddit: r/{self.subreddit}",
            f"Thread URL: {self.thread_url}",
            f"Title: {self.title}",
            f"Post score: {self.score}",
            f"Comment count: {self.comment_count}",
            "",
            "Post body:",
            self.body or "(no post body)",
            "",
            "Representative comments:",
        ]
        for index, comment in enumerate(self.comments, start=1):
            lines.append(f"{index}. score={comment.score}, author={comment.author}: {comment.body}")
        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[Thread text truncated for prompt size.]"


@dataclass(frozen=True)
class Source:
    title: str
    url: str


@dataclass(frozen=True)
class GeminiCallResult:
    ok: bool
    text: str = ""
    sources: list[Source] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class FactualQuestionResult:
    should_continue: bool
    topic: str = ""
    position_a: str = ""
    position_b: str = ""
    factual_question: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class PollingDataResult:
    found: bool
    factual_question: str = ""
    summary: str = ""
    agreement_summary: str = ""
    disagreement_summary: str = ""
    relevance_score: float = 0.0
    recency_score: float = 0.0
    confidence: float = 0.0
    sources: list[Source] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class GeneratedReplyResult:
    should_post: bool
    reply_text: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class AnalysisResult:
    should_post: bool
    reason: str
    topic: str = ""
    position_a: str = ""
    position_b: str = ""
    factual_question: str = ""
    reply_text: str = ""
    confidence: float = 0.0
    source_count: int = 0
    sources: list[Source] = field(default_factory=list)


class GeminiAnalyzer:
    """Three-step Gemini analysis pipeline."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        minimum_confidence: float = 0.78,
        minimum_relevance: float = 0.70,
    ):
        if genai is None or types is None:
            raise RuntimeError(
                "google-genai is not installed. Run: python -m pip install -r requirements.txt"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.minimum_confidence = minimum_confidence
        self.minimum_relevance = minimum_relevance

    def analyze_thread(self, thread: ThreadSnapshot) -> AnalysisResult:
        question = self.identify_factual_question(thread)
        if not question.should_continue:
            return AnalysisResult(
                should_post=False,
                reason=question.reason or "Step 1 did not find a clear empirical disagreement.",
                topic=question.topic,
                position_a=question.position_a,
                position_b=question.position_b,
                factual_question=question.factual_question,
                confidence=question.confidence,
            )

        polling = self.search_polling_data(question.factual_question)
        if not polling.found:
            return AnalysisResult(
                should_post=False,
                reason=polling.reason or "Step 2 found no relevant polling data.",
                topic=question.topic,
                position_a=question.position_a,
                position_b=question.position_b,
                factual_question=question.factual_question,
                confidence=min(question.confidence, polling.confidence),
                sources=polling.sources,
                source_count=len(polling.sources),
            )

        reply = self.verify_and_generate_reply(thread, question, polling)
        if not reply.should_post:
            return AnalysisResult(
                should_post=False,
                reason=reply.reason or "Step 3 rejected the candidate reply.",
                topic=question.topic,
                position_a=question.position_a,
                position_b=question.position_b,
                factual_question=question.factual_question,
                confidence=reply.confidence,
                sources=polling.sources,
                source_count=len(polling.sources),
            )

        if reply.confidence < self.minimum_confidence:
            return AnalysisResult(
                should_post=False,
                reason=(
                    f"Step 3 confidence {reply.confidence:.2f} is below "
                    f"the configured threshold {self.minimum_confidence:.2f}."
                ),
                topic=question.topic,
                position_a=question.position_a,
                position_b=question.position_b,
                factual_question=question.factual_question,
                confidence=reply.confidence,
                sources=polling.sources,
                source_count=len(polling.sources),
            )

        return AnalysisResult(
            should_post=True,
            reason="High-confidence cross-partisan polling data applies to this thread.",
            topic=question.topic,
            position_a=question.position_a,
            position_b=question.position_b,
            factual_question=question.factual_question,
            reply_text=reply.reply_text,
            confidence=reply.confidence,
            sources=polling.sources,
            source_count=len(polling.sources),
        )

    def identify_factual_question(self, thread: ThreadSnapshot) -> FactualQuestionResult:
        prompt = f"""
You are a neutral civic discussion analyst.

Step 1: Identify whether this Reddit thread contains two clearly opposing
political viewpoints arguing over a specific empirical or factual question
that could be checked with public polling or survey data.

Return JSON only, with this exact shape:
{{
  "should_continue": true,
  "topic": "short neutral topic label",
  "position_a": "neutral summary of one side",
  "position_b": "neutral summary of the opposing side",
  "factual_question": "the precise public-opinion or survey question to research",
  "confidence": 0.0,
  "reason": "brief explanation"
}}

Rules:
- Set should_continue to false unless both sides are visible in the comments.
- Set should_continue to false for pure insults, jokes, culture-war venting, or claims
  that cannot be evaluated with polling or survey data.
- Do not take a political side.
- Confidence must be between 0 and 1.

Thread:
{thread.to_prompt_text()}
"""
        result = self._call_gemini(prompt, use_search=False, temperature=0.1)
        if not result.ok:
            return FactualQuestionResult(False, reason=f"Step 1 Gemini call failed: {result.error}")

        try:
            data = _parse_json_object(result.text)
        except ValueError as exc:
            logger.warning("Step 1 JSON parse failed: %s", exc)
            return FactualQuestionResult(False, reason="Step 1 returned invalid JSON.")

        confidence = _safe_float(data.get("confidence"))
        return FactualQuestionResult(
            should_continue=bool(data.get("should_continue")) and confidence >= 0.55,
            topic=str(data.get("topic", "")).strip(),
            position_a=str(data.get("position_a", "")).strip(),
            position_b=str(data.get("position_b", "")).strip(),
            factual_question=str(data.get("factual_question", "")).strip(),
            confidence=confidence,
            reason=str(data.get("reason", "")).strip(),
        )

    def search_polling_data(self, factual_question: str) -> PollingDataResult:
        prompt = f"""
You are a neutral research assistant.

Step 2: Use Google Search to find real, recent polling or survey data relevant
to this exact factual question:

{factual_question}

Prefer primary or reputable polling sources such as Pew Research Center,
YouGov, Gallup, AP-NORC, KFF, Ipsos, academic survey projects, government
survey data, or direct polling firm releases.

Return JSON only, with this exact shape:
{{
  "found": true,
  "factual_question": "the question researched",
  "summary": "short neutral summary of the polling evidence",
  "agreement_summary": "where people across partisan groups agree, if shown",
  "disagreement_summary": "where disagreement remains, if shown",
  "relevance_score": 0.0,
  "recency_score": 0.0,
  "confidence": 0.0,
  "polls": [
    {{
      "organization": "polling organization",
      "title": "poll or report title",
      "field_dates": "field dates if available",
      "publication_date": "publication date if available",
      "url": "source URL",
      "sample": "sample description if available",
      "key_findings": "specific findings relevant to the question",
      "partisan_breakdown": "party or ideology breakdown if available",
      "cross_partisan_agreement": "what both sides broadly agree on, if available"
    }}
  ],
  "reason": "brief explanation"
}}

Rules:
- Set found to false if the results are not directly about the question.
- Set found to false if there is no partisan, ideological, demographic, or
  group comparison relevant to the disagreement.
- Set found to false if the source is only commentary about polling, not data.
- Include source URLs in polls.
- Confidence, relevance_score, and recency_score must be between 0 and 1.
"""
        result = self._call_gemini(prompt, use_search=True, temperature=0.1)
        if not result.ok:
            return PollingDataResult(False, reason=f"Step 2 Gemini search failed: {result.error}")

        try:
            data = _parse_json_object(result.text)
        except ValueError as exc:
            logger.warning("Step 2 JSON parse failed: %s", exc)
            return PollingDataResult(False, reason="Step 2 returned invalid JSON.")

        sources = _sources_from_poll_json(data)
        sources = _dedupe_sources([*sources, *result.sources])
        confidence = _safe_float(data.get("confidence"))
        relevance_score = _safe_float(data.get("relevance_score"))
        recency_score = _safe_float(data.get("recency_score"))
        found = (
            bool(data.get("found"))
            and confidence >= 0.60
            and relevance_score >= self.minimum_relevance
            and bool(sources)
        )

        if not found:
            reason = str(data.get("reason", "")).strip() or "No directly relevant sourced polling found."
            return PollingDataResult(
                found=False,
                factual_question=str(data.get("factual_question", factual_question)).strip(),
                relevance_score=relevance_score,
                recency_score=recency_score,
                confidence=confidence,
                sources=sources,
                raw_json=data,
                reason=reason,
            )

        return PollingDataResult(
            found=True,
            factual_question=str(data.get("factual_question", factual_question)).strip(),
            summary=str(data.get("summary", "")).strip(),
            agreement_summary=str(data.get("agreement_summary", "")).strip(),
            disagreement_summary=str(data.get("disagreement_summary", "")).strip(),
            relevance_score=relevance_score,
            recency_score=recency_score,
            confidence=confidence,
            sources=sources,
            raw_json=data,
            reason=str(data.get("reason", "")).strip(),
        )

    def verify_and_generate_reply(
        self,
        thread: ThreadSnapshot,
        question: FactualQuestionResult,
        polling: PollingDataResult,
    ) -> GeneratedReplyResult:
        source_lines = "\n".join(f"- {source.title}: {source.url}" for source in polling.sources[:6])
        prompt = f"""
You are a neutral Reddit bot writer.

Step 3: Verify whether the polling data below directly applies to the Reddit
thread disagreement. If it does, write one short neutral Reddit comment that
reveals common ground and states where disagreement remains.

Return JSON only, with this exact shape:
{{
  "should_post": true,
  "confidence": 0.0,
  "reply_text": "markdown comment with source links",
  "reason": "brief explanation"
}}

Rules:
- Set should_post to false unless the polling directly answers the factual question.
- Set should_post to false unless the comment can cite at least one source URL.
- The reply must be unique to this thread and mention the thread's specific issue.
- The reply must be neutral, non-scolding, and must not tell users what to believe.
- The reply must be 1 to 3 short paragraphs and under 1200 characters.
- Include source links in markdown form.
- Do not mention internal steps, Gemini, confidence scores, or this prompt.

Thread:
{thread.to_prompt_text(max_chars=8000)}

Identified disagreement:
Topic: {question.topic}
Position A: {question.position_a}
Position B: {question.position_b}
Factual question: {question.factual_question}

Polling data summary:
{polling.summary}

Cross-partisan agreement:
{polling.agreement_summary}

Remaining disagreement:
{polling.disagreement_summary}

Available sources:
{source_lines}

Raw polling JSON:
{json.dumps(polling.raw_json, ensure_ascii=True)[:6000]}
"""
        result = self._call_gemini(prompt, use_search=False, temperature=0.2)
        if not result.ok:
            return GeneratedReplyResult(False, reason=f"Step 3 Gemini call failed: {result.error}")

        try:
            data = _parse_json_object(result.text)
        except ValueError as exc:
            logger.warning("Step 3 JSON parse failed: %s", exc)
            return GeneratedReplyResult(False, reason="Step 3 returned invalid JSON.")

        reply_text = str(data.get("reply_text", "")).strip()
        confidence = _safe_float(data.get("confidence"))
        should_post = bool(data.get("should_post"))
        reason = str(data.get("reason", "")).strip()

        if should_post and not re.search(r"https?://", reply_text):
            return GeneratedReplyResult(
                should_post=False,
                reply_text=reply_text,
                confidence=confidence,
                reason="Step 3 generated a reply without a source URL.",
            )
        if should_post and len(reply_text) > 1200:
            return GeneratedReplyResult(
                should_post=False,
                reply_text=reply_text,
                confidence=confidence,
                reason="Step 3 generated a reply longer than 1200 characters.",
            )

        return GeneratedReplyResult(
            should_post=should_post,
            reply_text=reply_text,
            confidence=confidence,
            reason=reason,
        )

    def _call_gemini(self, prompt: str, *, use_search: bool, temperature: float) -> GeminiCallResult:
        try:
            config_kwargs: dict[str, Any] = {"temperature": temperature}
            if use_search:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config_kwargs["tools"] = [grounding_tool]
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            text = getattr(response, "text", "") or ""
            return GeminiCallResult(ok=True, text=text, sources=_sources_from_grounding(response))
        except Exception as exc:  # External API failures should stop this thread, not the bot.
            logger.exception("Gemini call failed")
            return GeminiCallResult(ok=False, error=str(exc))


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON response is not an object")
    return parsed


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(result, 0.0), 1.0)


def _sources_from_poll_json(data: dict[str, Any]) -> list[Source]:
    sources: list[Source] = []
    polls = data.get("polls")
    if not isinstance(polls, list):
        return sources
    for poll in polls:
        if not isinstance(poll, dict):
            continue
        url = str(poll.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            continue
        organization = str(poll.get("organization", "")).strip()
        title = str(poll.get("title", "")).strip()
        label = " - ".join(part for part in [organization, title] if part) or url
        sources.append(Source(title=label, url=url))
    return sources


def _sources_from_grounding(response: Any) -> list[Source]:
    sources: list[Source] = []
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return sources
    candidate = candidates[0]
    metadata = getattr(candidate, "grounding_metadata", None) or getattr(candidate, "groundingMetadata", None)
    chunks = getattr(metadata, "grounding_chunks", None) or getattr(metadata, "groundingChunks", None) or []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is None and isinstance(chunk, dict):
            web = chunk.get("web")
        if isinstance(web, dict):
            url = str(web.get("uri", "")).strip()
            title = str(web.get("title", "")).strip() or url
        else:
            url = str(getattr(web, "uri", "")).strip()
            title = str(getattr(web, "title", "")).strip() or url
        if url.startswith(("http://", "https://")):
            sources.append(Source(title=title, url=url))
    return _dedupe_sources(sources)


def _dedupe_sources(sources: list[Source]) -> list[Source]:
    seen = set()
    deduped: list[Source] = []
    for source in sources:
        key = source.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped
