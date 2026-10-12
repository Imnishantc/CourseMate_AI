"""Small local SQLite store for CourseMate chat conversations."""

import json
import os
import sqlite3
from pathlib import Path


def _database_path():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
        base = base / "CourseMateAI"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        base = base / "coursemate-ai"
    base.mkdir(parents=True, exist_ok=True)
    return base / "history.sqlite3"


def _connect():
    connection = sqlite3.connect(_database_path(), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            document_digest TEXT NOT NULL,
            document_name TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    connection.execute(
        """CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            sources TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    connection.commit()
    return connection


def list_conversations():
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC, created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def create_conversation(conversation_id, document_digest, document_name):
    with _connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, document_digest, document_name, title) VALUES (?, ?, ?, ?)",
            (conversation_id, document_digest, document_name, document_name),
        )


def get_conversation(conversation_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None


def get_messages(conversation_id):
    with _connect() as connection:
        rows = connection.execute(
            "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [
            {"role": row["role"], "content": row["content"], "sources": json.loads(row["sources"])}
            for row in rows
        ]


def add_message(conversation_id, role, content, sources=None):
    with _connect() as connection:
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(sources or [], ensure_ascii=False)),
        )
        conversation = connection.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE conversation_id = ? AND role = 'user'",
            (conversation_id,),
        ).fetchone()
        if conversation["count"] == 1 and role == "user":
            connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (content[:64], conversation_id),
            )
        connection.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )


def clear_conversation(conversation_id):
    with _connect() as connection:
        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        connection.execute(
            "UPDATE conversations SET title = document_name, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )

