from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled by runtime validation.
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SUBREDDITS = [
    "changemyview",
    "NeutralPolitics",
    "moderatepolitics",
    "PoliticalDiscussion",
    "AskConservatives",
    "AskALiberal",
    "worldnews",
    "news",
    "geopolitics",
    "economics",
    "politics",
    "canada",
]

DEFAULT_SEARCH_QUERY = (
    'democrat OR republican OR liberal OR conservative OR "public opinion" '
    'OR poll OR survey OR policy OR voters'
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = default
    else:
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None:
        result = max(minimum, result)
    return result


def _env_float(name: str, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = default
    else:
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number") from exc
    return min(max(result, minimum), maximum)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _env_subreddits() -> list[str]:
    raw = os.getenv("SUBREDDITS")
    if raw is None or raw.strip() == "":
        return list(DEFAULT_SUBREDDITS)
    return [item.strip().removeprefix("r/") for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    reddit_client_id: str
    reddit_client_secret: str
    reddit_username: str
    reddit_password: str
    reddit_user_agent: str
    gemini_api_key: str
    dry_run: bool
    max_daily_posts: int
    scan_interval_minutes: int
    database_path: Path
    subreddits: list[str]
    search_query: str
    search_limit_per_subreddit: int
    max_candidates_per_scan: int
    candidate_comment_limit: int
    min_thread_comments: int
    min_thread_score: int
    reddit_sort: str
    reddit_time_filter: str
    gemini_model: str
    minimum_confidence: float
    minimum_relevance: float
    min_seconds_between_live_posts: int
    timezone: str

    def validate_runtime_credentials(self) -> None:
        missing = []
        required = {
            "REDDIT_CLIENT_ID": self.reddit_client_id,
            "REDDIT_CLIENT_SECRET": self.reddit_client_secret,
            "REDDIT_USERNAME": self.reddit_username,
            "REDDIT_PASSWORD": self.reddit_password,
            "REDDIT_USER_AGENT": self.reddit_user_agent,
            "GEMINI_API_KEY": self.gemini_api_key,
        }
        for name, value in required.items():
            if not value:
                missing.append(name)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing required .env values: {joined}. "
                "Copy .env.example to .env and fill in these values before running the bot."
            )


def load_settings() -> Settings:
    if load_dotenv is not None:
        load_dotenv(BASE_DIR / ".env")

    configured_cap = _env_int("MAX_DAILY_POSTS", 10, minimum=0)
    hard_capped_daily_posts = min(configured_cap, 10)

    return Settings(
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", "").strip(),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        reddit_username=os.getenv("REDDIT_USERNAME", "").strip(),
        reddit_password=os.getenv("REDDIT_PASSWORD", "").strip(),
        reddit_user_agent=os.getenv("REDDIT_USER_AGENT", "").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        dry_run=_env_bool("DRY_RUN", True),
        max_daily_posts=hard_capped_daily_posts,
        scan_interval_minutes=_env_int("SCAN_INTERVAL_MINUTES", 30, minimum=1),
        database_path=_env_path("DATABASE_PATH", BASE_DIR / "common_ground_bot.sqlite3"),
        subreddits=_env_subreddits(),
        search_query=os.getenv("SEARCH_QUERY", DEFAULT_SEARCH_QUERY).strip() or DEFAULT_SEARCH_QUERY,
        search_limit_per_subreddit=_env_int("SEARCH_LIMIT_PER_SUBREDDIT", 15, minimum=1),
        max_candidates_per_scan=_env_int("MAX_CANDIDATES_PER_SCAN", 25, minimum=1),
        candidate_comment_limit=_env_int("CANDIDATE_COMMENT_LIMIT", 30, minimum=5),
        min_thread_comments=_env_int("MIN_THREAD_COMMENTS", 20, minimum=0),
        min_thread_score=_env_int("MIN_THREAD_SCORE", 0),
        reddit_sort=os.getenv("REDDIT_SORT", "comments").strip() or "comments",
        reddit_time_filter=os.getenv("REDDIT_TIME_FILTER", "day").strip() or "day",
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash",
        minimum_confidence=_env_float("MINIMUM_CONFIDENCE", 0.78),
        minimum_relevance=_env_float("MINIMUM_RELEVANCE", 0.70),
        min_seconds_between_live_posts=_env_int("MIN_SECONDS_BETWEEN_LIVE_POSTS", 60, minimum=0),
        timezone=os.getenv("TIMEZONE", "UTC").strip() or "UTC",
    )
