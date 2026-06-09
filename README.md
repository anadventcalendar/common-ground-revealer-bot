# Common Ground Revealer Bot

This repository contains the source code for the Reddit bot that will access the Reddit API.

The bot performs low-volume scheduled scans of configured subreddits, evaluates candidate threads, and may submit a single neutral source-cited comment when strict checks pass. Credentials and private runtime configuration are not included in this repository.

## API Use

- Reads subreddit listings and thread comments to find candidate discussions.
- Skips locked and archived threads.
- Checks global SQLite state before posting so the same thread is not handled twice.
- Submits a Reddit comment only when dry-run mode is disabled, the daily cap allows it, and the analysis pipeline returns a high-confidence result.

## Safety Controls

- `DRY_RUN=true` by default.
- `MAX_DAILY_POSTS` defaults to `10` and is capped at `10`.
- Live posts are rate-limited.
- Previously handled thread URLs are tracked globally across all configured subreddits.
- Reddit and Gemini failures are logged and do not trigger aggressive retries.
- No API credentials, tokens, or private `.env` files are committed.

## File Overview

- `main.py` - entry point; runs a one-time scan or starts the scheduler.
- `scanner.py` - scheduled search across configured subreddits.
- `state.py` - SQLite database tracking posts and history across communities.
- `analyzer.py` - Gemini pipeline for identifying factual questions, retrieving polling data, and generating replies.
- `poster.py` - dry-run behavior, daily cap enforcement, rate limiting, and live Reddit posting.
- `config.py` - subreddit list and runtime settings loaded from `.env`.
- `schema.sql` - database schema.
- `tests/` - offline unit tests that do not require Reddit or Gemini credentials.
