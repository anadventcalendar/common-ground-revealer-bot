from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

try:
    import praw
except ImportError:  # pragma: no cover - handled by main.
    praw = None

try:
    from apscheduler.schedulers.blocking import BlockingScheduler
except ImportError:  # pragma: no cover - handled by main.
    BlockingScheduler = None

from analyzer import GeminiAnalyzer
from config import load_settings
from poster import RedditPoster
from scanner import RedditScanner
from state import BotState


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common Ground Revealer Reddit bot")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run one proactive scan and exit. This is the safest first test mode.",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the SQLite database and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def build_reddit_client(settings):
    if praw is None:
        raise RuntimeError("praw is not installed. Run: python -m pip install -r requirements.txt")
    return praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        username=settings.reddit_username,
        password=settings.reddit_password,
        user_agent=settings.reddit_user_agent,
        ratelimit_seconds=600,
    )


def build_scanner(settings, state: BotState) -> RedditScanner:
    reddit = build_reddit_client(settings)
    analyzer = GeminiAnalyzer(
        settings.gemini_api_key,
        settings.gemini_model,
        minimum_confidence=settings.minimum_confidence,
        minimum_relevance=settings.minimum_relevance,
    )
    poster = RedditPoster(state, settings)
    return RedditScanner(
        reddit=reddit,
        state=state,
        analyzer=analyzer,
        poster=poster,
        settings=settings,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    settings = load_settings()

    state = BotState(settings.database_path)
    state.initialize()
    logger.info("SQLite state initialized at %s", settings.database_path)

    if args.init_db:
        logger.info("Database initialization complete.")
        return 0

    settings.validate_runtime_credentials()
    scanner = build_scanner(settings, state)

    logger.info(
        "Common Ground Revealer starting. DRY_RUN=%s max_daily_posts=%s scan_interval=%s minutes",
        settings.dry_run,
        settings.max_daily_posts,
        settings.scan_interval_minutes,
    )

    if args.run_once:
        scanner.scan_once()
        return 0

    if BlockingScheduler is None:
        raise RuntimeError("APScheduler is not installed. Run: python -m pip install -r requirements.txt")

    timezone = ZoneInfo(settings.timezone)
    scheduler = BlockingScheduler(timezone=timezone)
    scheduler.add_job(
        scanner.scan_once,
        trigger="interval",
        minutes=settings.scan_interval_minutes,
        id="proactive_cross_subreddit_scan",
        name="scheduled cross-subreddit scan",
        next_run_time=datetime.now(timezone),
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )

    def shutdown(signum, frame):
        logger.info("Received signal %s, shutting down scheduler.", signum)
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    scheduler.start()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except Exception as exc:
        logger.exception("Bot exited with an error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
