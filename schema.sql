PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS posted_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_url TEXT NOT NULL UNIQUE,
    thread_id TEXT,
    subreddit TEXT NOT NULL,
    posted_at TEXT NOT NULL DEFAULT (datetime('now')),
    reddit_comment_id TEXT,
    dry_run INTEGER NOT NULL DEFAULT 0,
    reply_text TEXT NOT NULL,
    source_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'posted'
);

CREATE TABLE IF NOT EXISTS argument_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at TEXT NOT NULL DEFAULT (datetime('now')),
    subreddit TEXT NOT NULL,
    thread_id TEXT,
    thread_url TEXT NOT NULL,
    topic TEXT,
    position_a TEXT,
    position_b TEXT,
    factual_question TEXT,
    confidence REAL NOT NULL DEFAULT 0.0,
    raw_summary TEXT
);

CREATE TABLE IF NOT EXISTS post_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempted_at TEXT NOT NULL DEFAULT (datetime('now')),
    thread_url TEXT NOT NULL,
    thread_id TEXT,
    subreddit TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT,
    dry_run INTEGER NOT NULL DEFAULT 1,
    reddit_comment_id TEXT,
    reply_text TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_posted_threads_url
    ON posted_threads(thread_url);

CREATE INDEX IF NOT EXISTS idx_argument_patterns_topic
    ON argument_patterns(topic);

CREATE INDEX IF NOT EXISTS idx_argument_patterns_thread
    ON argument_patterns(thread_url);

CREATE INDEX IF NOT EXISTS idx_post_history_attempted_at
    ON post_history(attempted_at);

CREATE INDEX IF NOT EXISTS idx_post_history_thread
    ON post_history(thread_url);
