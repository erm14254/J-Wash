import json
import logging
import errno
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import numpy as np

import config

DB_PATH = config.DATA_DIR / "jlens.db"
FRAMES_DIR = config.DATA_DIR / "frames"
log = logging.getLogger(__name__)


class FrameStorageError(RuntimeError):
    """Frame archive publication or loading failed coherently."""


class FramesNotAttached(ValueError):
    """No frame archive is currently attached to the message."""


class FramePointerTurnover(RuntimeError):
    """The committed frame pointer changed too often while reading."""


class FrameFileMissing(FrameStorageError):
    """The committed frame archive is missing from disk."""


MAX_FRAME_ERROR_CHARS = 512
MUTATION_STATES = frozenset({"committed", "not_committed", "stale", "superseded", "ambiguous"})


@dataclass(frozen=True)
class StorageMutationOutcome:
    state: str
    mutation: str
    entity_id: str | None = None
    expected_version: int | None = None
    observed_version: int | None = None
    error_kind: str | None = None
    error_message: str | None = None
    value: object | None = None

    def __post_init__(self):
        if self.state not in MUTATION_STATES:
            raise ValueError("invalid storage mutation state")
        for value in (self.state, self.mutation, self.entity_id, self.error_kind, self.error_message):
            if value is not None and (not isinstance(value, str) or len(value) > MAX_FRAME_ERROR_CHARS):
                raise ValueError("storage mutation strings must be bounded")
        def primitive(value):
            return (
                value is None
                or isinstance(value, (bool, int, float))
                or (isinstance(value, str) and len(value) <= MAX_FRAME_ERROR_CHARS)
                or (isinstance(value, tuple) and all(primitive(item) for item in value))
            )
        if not primitive(self.value):
            raise ValueError("storage mutation value must be immutable and primitive")


SUPPORTED_FRAME_PHASES = frozenset({"reading", "thinking", "prompt", "gen"})


def _normalize_frame_layer_key(key):
    """Return canonical integer layer IDs for supported archive map keys.

    Production archives historically store frame-layer maps with string keys
    like "0"; remediation heads may have written integer keys.  Bool values are
    rejected even though bool is an int subclass.
    """
    if isinstance(key, bool):
        raise ValueError("invalid frame layer key")
    if isinstance(key, int):
        if key < 0:
            raise ValueError("invalid frame layer key")
        return key
    if isinstance(key, str):
        if key == "0":
            return 0
        if key and key.isascii() and key[0] in "123456789" and key.isdecimal():
            return int(key)
        raise ValueError("invalid frame layer key")
    raise ValueError("invalid frame layer key")


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    incarnation_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_tombstones (
    mutation_id TEXT PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    incarnation_id TEXT NOT NULL,
    deleted_version INTEGER NOT NULL,
    deleted_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES messages(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT,
    frames_file TEXT,
    created_at TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Store:
    def __init__(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(SCHEMA)
        self._ensure_conversation_version(conn)
        self._ensure_conversation_incarnations(conn)
        self._ensure_tombstone_incarnations(conn)
        self._ensure_message_version(conn)
        conn.commit()

    @staticmethod
    def _ensure_conversation_version(conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "version" not in cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN version INTEGER NOT NULL DEFAULT 1")

    @staticmethod
    def _ensure_conversation_incarnations(conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "incarnation_id" not in cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN incarnation_id TEXT")
        rows = conn.execute(
            "SELECT id FROM conversations WHERE incarnation_id IS NULL OR incarnation_id = ''"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE conversations SET incarnation_id = ? WHERE id = ?",
                (uuid.uuid4().hex, row[0]),
            )

    @staticmethod
    def _ensure_tombstone_incarnations(conn):
        info = conn.execute("PRAGMA table_info(conversation_tombstones)").fetchall()
        cols = {row[1] for row in info}
        mutation_is_pk = any(row[1] == "mutation_id" and row[5] == 1 for row in info)
        if "incarnation_id" in cols and mutation_is_pk:
            return
        conn.execute("ALTER TABLE conversation_tombstones RENAME TO conversation_tombstones_legacy")
        conn.execute(
            "CREATE TABLE conversation_tombstones ("
            "mutation_id TEXT PRIMARY KEY, conversation_id INTEGER NOT NULL, "
            "incarnation_id TEXT NOT NULL, deleted_version INTEGER NOT NULL, deleted_at TEXT NOT NULL)"
        )
        legacy = conn.execute(
            "SELECT conversation_id, mutation_id, deleted_version, deleted_at "
            "FROM conversation_tombstones_legacy"
        ).fetchall()
        for row in legacy:
            conn.execute(
                "INSERT INTO conversation_tombstones VALUES (?, ?, ?, ?, ?)",
                (row[1], row[0], uuid.uuid4().hex, row[2], row[3]),
            )
        conn.execute("DROP TABLE conversation_tombstones_legacy")

    @staticmethod
    def _ensure_message_version(conn):
        cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        if "version" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN version INTEGER NOT NULL DEFAULT 0")

    @staticmethod
    def _unlink_best_effort(path):
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            log.warning("failed to retire unreferenced frame file %s", path, exc_info=True)

    @staticmethod
    def _fsync_dir_best_effort(path):
        try:
            fd = os.open(str(path), os.O_RDONLY)
        except (OSError, AttributeError):
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _discard_conn(self, conn=None):
        conn = conn or getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if hasattr(self._local, "conn"):
            del self._local.conn

    def _abort_transaction_or_discard(self, conn):
        try:
            if conn.in_transaction:
                conn.rollback()
            if conn.in_transaction:
                raise sqlite3.OperationalError("rollback left transaction open")
            conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'").fetchone()
            return True
        except Exception:
            log.warning("discarding poisoned sqlite connection after transaction abort failure", exc_info=True)
            self._discard_conn(conn)
            return False

    def _rollback_or_discard(self, conn):
        return self._abort_transaction_or_discard(conn)

    @contextmanager
    def _independent_read_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def _fresh_conn(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _candidate_referenced(self, filename):
        if not filename:
            return False
        try:
            with self._independent_read_connection() as read:
                row = read.execute("SELECT 1 FROM messages WHERE frames_file = ? LIMIT 1", (filename,)).fetchone()
                return row is not None
        except Exception:
            log.warning("could not prove candidate %s is unreferenced; preserving it", filename, exc_info=True)
            return True

    def _unlink_candidate_if_unreferenced(self, filename, path):
        if path is not None and not self._candidate_referenced(filename):
            self._unlink_best_effort(path)

    @staticmethod
    def _bounded_error(exc):
        return str(exc)[:MAX_FRAME_ERROR_CHARS]

    @classmethod
    def _primitive_value(cls, value):
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:MAX_FRAME_ERROR_CHARS]
        if isinstance(value, (tuple, list)):
            return tuple(cls._primitive_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                (str(key)[:MAX_FRAME_ERROR_CHARS], cls._primitive_value(item))
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        raise TypeError("storage mutation outcome values must be primitive")

    def _mutation_outcome(self, state, mutation, *, entity_id=None, expected_version=None,
                          observed_version=None, exc=None, message=None, value=None):
        return StorageMutationOutcome(
            state=state,
            mutation=str(mutation)[:MAX_FRAME_ERROR_CHARS],
            entity_id=(str(entity_id)[:MAX_FRAME_ERROR_CHARS] if entity_id is not None else None),
            expected_version=expected_version,
            observed_version=observed_version,
            error_kind=(exc.__class__.__name__[:MAX_FRAME_ERROR_CHARS] if exc is not None else None),
            error_message=self._bounded_error(exc) if exc is not None else (
                str(message)[:MAX_FRAME_ERROR_CHARS] if message is not None else None
            ),
            value=self._primitive_value(value),
        )

    def _reconcile_mutation(self, mutation, entity_id, query, params, classify, *,
                            expected_version=None, operational_exc=None,
                            fallback_value=None, fallback_observed_version=None):
        """Read durable state independently and convert every failure to data."""
        try:
            with self._independent_read_connection() as read:
                row = read.execute(query, params).fetchone()
            state, observed_version, value, message = classify(row)
            return self._mutation_outcome(
                state, mutation, entity_id=entity_id,
                expected_version=expected_version,
                observed_version=observed_version, value=value,
                exc=operational_exc if state in {"not_committed", "ambiguous"} else None,
                message=message,
            )
        except BaseException as exc:
            log.warning("storage mutation reconciliation failed for %s %s", mutation, entity_id,
                        exc_info=True)
            return self._mutation_outcome(
                "ambiguous", mutation, entity_id=entity_id,
                expected_version=expected_version,
                observed_version=fallback_observed_version,
                exc=exc, value=fallback_value,
            )

    def create_conversation(self, title, tags=None):
        mutation = "create_conversation"
        try:
            conn = self._conn()
            now = _now()
            tags_sql = json.dumps(tags or [])
            incarnation_id = uuid.uuid4().hex
            conversation_id = int(conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM conversations"
            ).fetchone()[0])
        except BaseException as exc:
            return self._mutation_outcome("not_committed", mutation, exc=exc)
        intended = (conversation_id, title, tags_sql, now, now, 1, incarnation_id)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO conversations (id, title, tags, created_at, updated_at, version, incarnation_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", intended,
            )
            conn.commit()
            return self._mutation_outcome(
                "committed", mutation, entity_id=conversation_id,
                observed_version=1, value=conversation_id,
            )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            def classify(row):
                if row is None:
                    return "not_committed", None, None, None
                actual = tuple(row[key] for key in (
                    "id", "title", "tags", "created_at", "updated_at", "version", "incarnation_id"
                ))
                if actual == intended:
                    return "committed", row["version"], conversation_id, None
                return "ambiguous", row["version"], None, "conversation ID contains conflicting durable state"
            return self._reconcile_mutation(
                mutation, conversation_id,
                "SELECT id, title, tags, created_at, updated_at, version, incarnation_id FROM conversations WHERE id = ?",
                (conversation_id,), classify, expected_version=1, operational_exc=exc,
            )

    def update_conversation(self, conversation_id, title=None, tags=None):
        mutation = "update_conversation"
        try:
            conn = self._conn()
            now = _now()
        except BaseException as exc:
            return self._mutation_outcome("ambiguous", mutation, entity_id=conversation_id, exc=exc)
        original = intended = None
        expected_version = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT title, tags, created_at, updated_at, version, incarnation_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if current is None:
                self._abort_transaction_or_discard(conn)
                return self._mutation_outcome("superseded", mutation, entity_id=conversation_id)
            expected_version = current["version"]
            original = tuple(current[key] for key in ("title", "tags", "created_at", "updated_at", "version", "incarnation_id"))
            new_title = current["title"] if title is None else title
            new_tags = current["tags"] if tags is None else json.dumps(tags)
            intended = (new_title, new_tags, current["created_at"], now, expected_version + 1, current["incarnation_id"])
            cur = conn.execute(
                "UPDATE conversations SET title = ?, tags = ?, updated_at = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (new_title, new_tags, now, conversation_id, expected_version),
            )
            if cur.rowcount != 1:
                self._abort_transaction_or_discard(conn)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=conversation_id,
                    expected_version=expected_version,
                )
            conn.commit()
            return self._mutation_outcome(
                "committed", mutation, entity_id=conversation_id,
                expected_version=expected_version, observed_version=expected_version + 1,
            )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            def classify(row):
                if row is None:
                    return "superseded", None, None, None
                observed = row["version"]
                actual = tuple(row[key] for key in ("title", "tags", "created_at", "updated_at", "version", "incarnation_id"))
                if original is None:
                    return "ambiguous", observed, None, "conversation pre-mutation state is unavailable"
                if actual == intended:
                    return "committed", observed, None, None
                if actual == original:
                    return "not_committed", observed, None, None
                return "superseded", observed, None, "conversation contains different durable state"
            return self._reconcile_mutation(
                mutation, conversation_id,
                "SELECT title, tags, created_at, updated_at, version, incarnation_id FROM conversations WHERE id = ?",
                (conversation_id,), classify, expected_version=expected_version, operational_exc=exc,
            )

    def delete_conversation(self, conversation_id):
        mutation = "delete_conversation"
        mutation_id = uuid.uuid4().hex
        deleted_at = _now()
        try:
            conn = self._conn()
        except BaseException as exc:
            return self._mutation_outcome("ambiguous", mutation, entity_id=conversation_id, exc=exc)
        rows = []
        original_version = intended_version = None
        incarnation_id = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            conversation = conn.execute(
                "SELECT version, incarnation_id FROM conversations WHERE id = ?", (conversation_id,),
            ).fetchone()
            if conversation is None:
                self._abort_transaction_or_discard(conn)
                return self._mutation_outcome("superseded", mutation, entity_id=conversation_id)
            original_version = conversation["version"]
            incarnation_id = conversation["incarnation_id"]
            intended_version = original_version + 1
            rows = conn.execute(
                "SELECT frames_file FROM messages WHERE conversation_id = ? AND frames_file IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            conn.execute(
                "INSERT INTO conversation_tombstones "
                "(mutation_id, conversation_id, incarnation_id, deleted_version, deleted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (mutation_id, conversation_id, incarnation_id, intended_version, deleted_at),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND incarnation_id = ? AND version = ?",
                (conversation_id, incarnation_id, original_version),
            )
            conn.commit()
            outcome = self._mutation_outcome(
                "committed", mutation, entity_id=conversation_id,
                expected_version=original_version, observed_version=intended_version,
                value=mutation_id,
            )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            def classify(row):
                observed = row["conversation_version"]
                own_tombstone = (
                    row["tombstone_mutation_id"] == mutation_id
                    and row["tombstone_conversation_id"] == conversation_id
                    and row["tombstone_incarnation_id"] == incarnation_id
                    and row["deleted_version"] == intended_version
                )
                if own_tombstone:
                    return "committed", intended_version, mutation_id, None
                other_delete = bool(row["other_tombstone_count"])
                if (
                    row["conversation_incarnation"] == incarnation_id
                    and observed == original_version
                    and not other_delete
                ):
                    return "not_committed", observed, mutation_id, None
                if row["conversation_incarnation"] == incarnation_id and observed is not None:
                    return "superseded", observed, mutation_id, "conversation contains newer durable state"
                if observed is not None and row["conversation_incarnation"] != incarnation_id:
                    return "superseded", observed, mutation_id, "conversation ID contains a newer incarnation"
                if other_delete:
                    return "superseded", row["other_deleted_version"], mutation_id, "another delete won"
                return "ambiguous", observed, mutation_id, "conversation incarnation cannot be causally classified"
            outcome = self._reconcile_mutation(
                mutation, conversation_id,
                "SELECT (SELECT version FROM conversations WHERE id = ?) AS conversation_version, "
                "(SELECT incarnation_id FROM conversations WHERE id = ?) AS conversation_incarnation, "
                "(SELECT mutation_id FROM conversation_tombstones WHERE mutation_id = ?) AS tombstone_mutation_id, "
                "(SELECT conversation_id FROM conversation_tombstones WHERE mutation_id = ?) AS tombstone_conversation_id, "
                "(SELECT incarnation_id FROM conversation_tombstones WHERE mutation_id = ?) AS tombstone_incarnation_id, "
                "(SELECT deleted_version FROM conversation_tombstones WHERE mutation_id = ?) AS deleted_version, "
                "(SELECT COUNT(*) FROM conversation_tombstones WHERE conversation_id = ? AND incarnation_id = ? "
                "AND mutation_id != ?) AS other_tombstone_count, "
                "(SELECT MAX(deleted_version) FROM conversation_tombstones WHERE conversation_id = ? "
                "AND incarnation_id = ? AND mutation_id != ?) AS other_deleted_version",
                (
                    conversation_id, conversation_id, mutation_id, mutation_id, mutation_id,
                    mutation_id, conversation_id, incarnation_id, mutation_id,
                    conversation_id, incarnation_id, mutation_id,
                ), classify,
                expected_version=original_version, operational_exc=exc,
                fallback_value=mutation_id, fallback_observed_version=original_version,
            )
        if outcome.state == "committed":
            for row in rows:
                self._unlink_best_effort(FRAMES_DIR / row["frames_file"])
        return outcome

    def list_conversations(self, query=None, limit=200):
        conn = self._conn()
        if query:
            hits = conn.execute(
                """
                SELECT messages_fts.rowid AS mid,
                       snippet(messages_fts, 0, '[', ']', '…', 12) AS snip,
                       rank
                FROM messages_fts
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT 500
                """,
                (query,),
            ).fetchall()
            best = {}
            for hit in hits:
                row = conn.execute(
                    "SELECT conversation_id FROM messages WHERE id = ?", (hit["mid"],)
                ).fetchone()
                if row and row["conversation_id"] not in best:
                    best[row["conversation_id"]] = hit["snip"]
            rows = []
            for cid, snip in list(best.items())[:limit]:
                conv = conn.execute(
                    """
                    SELECT c.id, c.title, c.tags, c.updated_at,
                           (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS n_messages
                    FROM conversations c WHERE c.id = ?
                    """,
                    (cid,),
                ).fetchone()
                if conv:
                    rows.append(dict(conv, snippet=snip))
        else:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.tags, c.updated_at,
                       count(m.id) AS n_messages, NULL AS snippet
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            dict(row, tags=json.loads(row["tags"]))
            for row in (dict(r) for r in rows)
        ]

    def get_conversation(self, conversation_id):
        conn = self._conn()
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conv is None:
            raise ValueError(f"unknown conversation {conversation_id}")
        rows = conn.execute(
            "SELECT id, parent_id, role, content, meta, frames_file, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        messages = [
            {
                "id": row["id"],
                "parent_id": row["parent_id"],
                "role": row["role"],
                "content": row["content"],
                "meta": json.loads(row["meta"]) if row["meta"] else None,
                "has_frames": row["frames_file"] is not None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return {
            "id": conv["id"],
            "title": conv["title"],
            "tags": json.loads(conv["tags"]),
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": messages,
        }

    def add_message(self, conversation_id, parent_id, role, content, meta=None, *, return_version=False):
        try:
            conn = self._conn()
        except BaseException as exc:
            return self._mutation_outcome("not_committed", "add_message", exc=exc)
        try:
            now = _now()
            meta_sql = json.dumps(meta, ensure_ascii=False) if meta else None
        except BaseException as exc:
            return self._mutation_outcome("not_committed", "add_message", exc=exc)
        try:
            message_id = int(conn.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM messages"
            ).fetchone()[0])
        except BaseException as exc:
            return self._mutation_outcome("not_committed", "add_message", exc=exc)
        intended = (message_id, conversation_id, parent_id, role, content, meta_sql, None, now, 0)
        value = (message_id, 0)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO messages (id, conversation_id, parent_id, role, content, meta, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    conversation_id,
                    parent_id,
                    role,
                    content,
                    meta_sql,
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()
            return self._mutation_outcome(
                "committed", "add_message", entity_id=message_id,
                observed_version=0, value=value,
            )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            keys = ("id", "conversation_id", "parent_id", "role", "content", "meta",
                    "frames_file", "created_at", "version")
            def classify(row):
                if row is None:
                    return "not_committed", None, None, None
                observed = row["version"]
                if tuple(row[key] for key in keys) == intended:
                    return "committed", observed, value, None
                return "ambiguous", observed, None, "message ID contains conflicting durable state"
            return self._reconcile_mutation(
                "add_message", message_id,
                "SELECT id, conversation_id, parent_id, role, content, meta, frames_file, created_at, version "
                "FROM messages WHERE id = ?", (message_id,), classify,
                operational_exc=exc,
            )

    def get_message(self, message_id):
        conn = self._conn()
        row = conn.execute(
            "SELECT id, conversation_id, parent_id, role, content, meta, frames_file, version "
            "FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown message {message_id}")
        return dict(row)

    def update_message(self, message_id, content, meta=None, *, clear_frames=False, expected_version=None):
        """Rewrite a message and increment its durable row version.

        Assistant edits may clear derived frame/provenance artifacts atomically;
        stale continuations then fail their versioned CAS instead of overwriting
        an acknowledged edit.
        """
        try:
            conn = self._conn()
        except BaseException as exc:
            return self._mutation_outcome(
                "ambiguous", "update_message", entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        try:
            row = conn.execute(
                "SELECT conversation_id, content, meta, frames_file, version FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        except BaseException as exc:
            return self._mutation_outcome("ambiguous", "update_message", entity_id=message_id,
                                          expected_version=expected_version, exc=exc)
        if row is None:
            return self._mutation_outcome("superseded", "update_message", entity_id=message_id,
                                          expected_version=expected_version)
        try:
            meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
        except BaseException as exc:
            return self._mutation_outcome(
                "not_committed", "update_message", entity_id=message_id,
                expected_version=expected_version, observed_version=row["version"], exc=exc,
            )
        frames_file = None if clear_frames else row["frames_file"]
        baseline = row["version"] if expected_version is None else expected_version
        original = (row["content"], row["meta"], row["frames_file"], row["version"])
        intended = (content, meta_json, frames_file, baseline + 1)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (content, meta_json, frames_file, message_id, baseline),
            )
            if cur.rowcount != 1:
                self._abort_transaction_or_discard(conn)
                self._discard_conn(conn)
                def classify_stale(current):
                    if current is None:
                        return "superseded", None, None, None
                    observed = current["version"]
                    return "stale", observed, None, "message changed before edit"
                return self._reconcile_mutation(
                    "update_message", message_id,
                    "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                    (message_id,), classify_stale, expected_version=baseline,
                )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), row["conversation_id"]),
            )
            conn.commit()
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            def classify(current):
                if current is None:
                    return "superseded", None, None, None
                observed = current["version"]
                actual = tuple(current[key] for key in ("content", "meta", "frames_file", "version"))
                if actual == intended:
                    return "committed", observed, None, None
                if actual == original:
                    return "not_committed", observed, None, None
                if observed != baseline:
                    return "stale", observed, None, "message contains a newer durable version"
                return "ambiguous", observed, None, "message durable state is inconsistent"
            outcome = self._reconcile_mutation(
                "update_message", message_id,
                "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                (message_id,), classify, expected_version=baseline, operational_exc=exc,
            )
            if outcome.state != "committed":
                return outcome
        if clear_frames and row["frames_file"]:
            self._unlink_best_effort(FRAMES_DIR / row["frames_file"])
        return self._mutation_outcome(
            "committed", "update_message", entity_id=message_id,
            expected_version=baseline, observed_version=baseline + 1,
        )

    def update_message_if_unchanged(self, message_id, expected_version, content, meta=None):
        return self.update_message_and_frames_if_unchanged(
            message_id, expected_version, content, meta, frames=None
        )

    def _write_unique_frame_candidate(self, message_id, expected_version, frame_blob):
        frame_file = f"{message_id}-v{int(expected_version) + 1}-{uuid.uuid4().hex}.msgpack"
        tmp = FRAMES_DIR / f".{frame_file}.tmp-{uuid.uuid4().hex}"
        final = FRAMES_DIR / frame_file
        try:
            with tmp.open("wb") as fh:
                fh.write(frame_blob)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError as exc:
                    if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
                        raise
            tmp.replace(final)
            self._fsync_dir_best_effort(FRAMES_DIR)
            return frame_file, final
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                log.warning("failed to clean temporary frame candidate %s", tmp, exc_info=True)
            try:
                final.unlink(missing_ok=True)
            except Exception:
                log.warning("failed to clean frame candidate %s", final, exc_info=True)
            raise FrameStorageError(str(exc)) from exc

    def update_message_and_frames_if_unchanged(self, message_id, expected_version, content, meta=None, *, frames=None, layers=None, k=0, clear_frames=False, frame_descriptor=None, pin_publication_state=None):
        mutation = "update_message_and_frames_if_unchanged"
        frame_file = None
        final_path = None
        try:
            if frames:
                frame_blob = self._pack_frames_blob(
                    frames, layers or [], k, frame_descriptor,
                    pin_publication_state=pin_publication_state,
                )
                frame_file, final_path = self._write_unique_frame_candidate(
                    message_id, expected_version, frame_blob
                )
                self._validate_frame_candidate(final_path, message_id)
        except BaseException as exc:
            self._unlink_best_effort(final_path)
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        try:
            conn = self._conn()
        except BaseException as exc:
            return self._mutation_outcome(
                "ambiguous", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc, value=frame_file,
            )
        old_file = None
        original = intended = None
        target_frame = None
        try:
            meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
        except BaseException as exc:
            self._unlink_candidate_if_unreferenced(frame_file, final_path)
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT conversation_id, content, meta, frames_file, version "
                "FROM messages WHERE id = ?", (message_id,),
            ).fetchone()
            if row is None:
                self._rollback_or_discard(conn)
                self._unlink_candidate_if_unreferenced(frame_file, final_path)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version,
                )
            old_file = row["frames_file"]
            original = tuple(row[key] for key in ("content", "meta", "frames_file", "version"))
            if row["version"] != expected_version:
                self._rollback_or_discard(conn)
                self._unlink_candidate_if_unreferenced(frame_file, final_path)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version, observed_version=row["version"],
                    message="message changed before continuation",
                )
            target_frame = None if clear_frames else (
                frame_file if frame_file is not None else old_file
            )
            intended = (content, meta_json, target_frame, expected_version + 1)
            cur = conn.execute(
                "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (content, meta_json, target_frame, message_id, expected_version),
            )
            if cur.rowcount != 1:
                self._rollback_or_discard(conn)
                self._discard_conn(conn)
                def classify_miss(current):
                    if current is None:
                        return "superseded", None, None, None
                    observed = current["version"]
                    actual = tuple(current[key] for key in ("content", "meta", "frames_file", "version"))
                    if actual == intended:
                        return "committed", observed, None, None
                    if actual == original:
                        return "not_committed", observed, None, None
                    return "superseded", observed, None, "message changed before continuation"
                outcome = self._reconcile_mutation(
                    mutation, message_id,
                    "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                    (message_id,), classify_miss, expected_version=expected_version,
                    fallback_value=frame_file,
                    fallback_observed_version=row["version"],
                )
                if outcome.state != "ambiguous":
                    self._unlink_candidate_if_unreferenced(frame_file, final_path)
                return outcome
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), row["conversation_id"]),
            )
            conn.commit()
            outcome = self._mutation_outcome(
                "committed", mutation, entity_id=message_id,
                expected_version=expected_version, observed_version=expected_version + 1,
                value=frame_file,
            )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            def classify(current):
                if current is None:
                    return "superseded", None, None, None
                observed = current["version"]
                actual = tuple(current[key] for key in ("content", "meta", "frames_file", "version"))
                if intended is not None and actual == intended:
                    return "committed", observed, frame_file, None
                if original is not None and actual == original:
                    return "not_committed", observed, None, None
                if observed != expected_version:
                    return "superseded", observed, None, "message contains a newer durable version"
                return "ambiguous", observed, frame_file, "message durable state is inconsistent"
            outcome = self._reconcile_mutation(
                mutation, message_id,
                "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                (message_id,), classify, expected_version=expected_version,
                operational_exc=exc,
                fallback_value=frame_file,
                fallback_observed_version=(original[3] if original is not None else None),
            )
            if outcome.state != "ambiguous":
                self._unlink_candidate_if_unreferenced(frame_file, final_path)

        if outcome.state == "committed" and old_file and old_file != target_frame:
            self._unlink_best_effort(FRAMES_DIR / old_file)
        return outcome

    def path_to_root(self, message_id):
        conn = self._conn()
        path = []
        current = message_id
        while current is not None:
            row = conn.execute(
                "SELECT id, parent_id, role, content FROM messages WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            path.append({"role": row["role"], "content": row["content"]})
            current = row["parent_id"]
        path.reverse()
        return path

    @staticmethod
    def _pack_frames_blob(frames, layers, k, descriptor=None, *, pin_publication_state=None):
        vocab = {}
        packed = []
        run_ids = {frame.get("generation_run_id") for frame in frames}
        gen_ids = {frame.get("gen") for frame in frames}
        numeric_gen_ids = {value for value in gen_ids if value is not None}
        archive_version = 2 if not frames or run_ids == {None} else 3
        if archive_version == 3 and (
            len(run_ids) != 1
            or next(iter(run_ids)) is None
            or not isinstance(next(iter(run_ids)), str)
            or len(next(iter(run_ids))) != 32
            or any(ch not in "0123456789abcdef" for ch in next(iter(run_ids)))
        ):
            raise FrameStorageError("schema-v3 frames require one valid generation run id")
        if archive_version == 3:
            pin_publication_state = pin_publication_state or "published"
            if pin_publication_state not in {"pending", "published", "unavailable"}:
                raise FrameStorageError("invalid pin publication state")
            if pin_publication_state in {"pending", "published"} and (
                len(numeric_gen_ids) != 1
                or isinstance(next(iter(numeric_gen_ids)), bool)
                or not isinstance(next(iter(numeric_gen_ids)), int)
                or next(iter(numeric_gen_ids)) < 0
            ):
                raise FrameStorageError("pinnable schema-v3 frames require one valid generation id")
        else:
            pin_publication_state = "unavailable"
        for frame in frames:
            vocab[frame["token_id"]] = frame["tok"]
            entry = {
                "pos": frame["pos"],
                "phase": frame["phase"],
                "token_id": frame["token_id"],
                "gen": frame.get("gen"),
                "generation_run_id": frame.get("generation_run_id"),
                "layers": {},
            }
            for layer, d in frame["layers"].items():
                for tid, s in zip(d["ids"], d["strs"]):
                    vocab[tid] = s
                for tid, s in zip(d["m_ids"], d["m_strs"]):
                    vocab[tid] = s
                entry["layers"][layer] = {
                    "ids": np.asarray(d["ids"], np.int32).tobytes(),
                    "p": np.asarray(d["p"], np.float16).tobytes(),
                    "m_ids": np.asarray(d["m_ids"], np.int32).tobytes(),
                    "m_p": np.asarray(d["m_p"], np.float16).tobytes(),
                    "m_rank": np.asarray(d["m_rank"], np.int32).tobytes(),
                }
            packed.append(entry)
        return msgpack.packb({
            "version": archive_version,
            "k": k,
            "descriptor": dict(descriptor) if descriptor is not None else None,
            "pin_publication_state": pin_publication_state,
            "pin_gen_id": (
                next(iter(numeric_gen_ids))
                if archive_version == 3 and pin_publication_state in {"pending", "published"} else None
            ),
            "pin_generation_run_id": (
                next(iter(run_ids))
                if archive_version == 3 and pin_publication_state in {"pending", "published"} else None
            ),
            # Immutable Store-internal causality for idempotent publication
            # compensation. These are never surfaced as usable pin identity.
            "pin_publication_attempt_gen_id": (
                next(iter(numeric_gen_ids))
                if archive_version == 3 and len(numeric_gen_ids) == 1 else None
            ),
            "pin_publication_attempt_run_id": (
                next(iter(run_ids)) if archive_version == 3 else None
            ),
            "layers": [int(l) for l in layers],
            "frames": packed,
            "vocab": {str(t): s for t, s in vocab.items()},
        })

    def finalize_continuation_pin_publication(
        self, message_id, *, expected_version, generation_run_id, gen_id, state,
    ):
        """Resolve pending frame pin state without changing assistant version/content."""
        mutation = "finalize_continuation_pin_publication"
        if state not in {"published", "unavailable"}:
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=expected_version, message="invalid publication state",
            )
        conn = None
        old_file = new_file = None
        final_path = None
        intended_meta = None
        try:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT meta, frames_file, version FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if row is None or row["frames_file"] is None:
                self._rollback_or_discard(conn)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version,
                )
            if row["version"] != expected_version:
                self._rollback_or_discard(conn)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version, observed_version=row["version"],
                )
            old_file = row["frames_file"]
            data = msgpack.unpackb((FRAMES_DIR / old_file).read_bytes(), strict_map_key=False)
            current_pin_state = data.get("pin_publication_state")
            attempt_gen_id = data.get("pin_publication_attempt_gen_id")
            attempt_run_id = data.get("pin_publication_attempt_run_id")
            exact_attempt = (
                attempt_run_id == generation_run_id
                and (gen_id is None or attempt_gen_id == gen_id)
            )
            allowed_transition = (
                current_pin_state == "pending"
                or current_pin_state == state
                or state == "unavailable" and current_pin_state == "published"
            )
            allowed_source = (
                data.get("version") == 3
                and exact_attempt
                and allowed_transition
            )
            if (
                not allowed_source
            ):
                self._rollback_or_discard(conn)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version, observed_version=row["version"],
                    message="continuation pin segment changed before publication resolved",
                )
            data["pin_publication_state"] = state
            if state != "published":
                data["pin_gen_id"] = None
                data["pin_generation_run_id"] = None
            meta = json.loads(row["meta"]) if row["meta"] else {}
            meta["pin_publication_state"] = state
            if state == "published":
                meta["pin_gen_id"] = gen_id
                meta["pin_generation_run_id"] = generation_run_id
            else:
                meta.pop("pin_gen_id", None)
                meta.pop("pin_generation_run_id", None)
            continuations = meta.get("continuations")
            if isinstance(continuations, list) and continuations:
                segment = continuations[-1]
                if isinstance(segment, dict):
                    segment["pin_publication_state"] = state
                    if state == "published":
                        segment["pin_gen_id"] = gen_id
                        segment["pin_generation_run_id"] = generation_run_id
                    else:
                        segment.pop("pin_gen_id", None)
                        segment.pop("pin_generation_run_id", None)
            intended_meta = json.dumps(meta, ensure_ascii=False)
            blob = msgpack.packb(data)
            new_file, final_path = self._write_unique_frame_candidate(
                message_id, expected_version, blob
            )
            self._validate_frame_candidate(final_path, message_id)
            cur = conn.execute(
                "UPDATE messages SET meta = ?, frames_file = ? "
                "WHERE id = ? AND version = ? AND frames_file = ?",
                (intended_meta, new_file, message_id, expected_version, old_file),
            )
            if cur.rowcount != 1:
                self._rollback_or_discard(conn)
                self._unlink_candidate_if_unreferenced(new_file, final_path)
                return self._mutation_outcome(
                    "superseded", mutation, entity_id=message_id,
                    expected_version=expected_version,
                )
            conn.commit()
            outcome = self._mutation_outcome(
                "committed", mutation, entity_id=message_id,
                expected_version=expected_version, observed_version=expected_version,
                value=state,
            )
        except BaseException as exc:
            if conn is not None:
                self._abort_transaction_or_discard(conn)
                self._discard_conn(conn)
            def classify(current):
                if current is None:
                    return "superseded", None, None, None
                observed = current["version"]
                if intended_meta is None or new_file is None or old_file is None:
                    return "ambiguous", observed, None, "pin publication state could not be read"
                if (
                    current["meta"] == intended_meta
                    and current["frames_file"] == new_file
                    and observed == expected_version
                ):
                    return "committed", observed, state, None
                if observed != expected_version or current["frames_file"] != old_file:
                    return "superseded", observed, None, "message pin state changed"
                return "not_committed", observed, None, None
            outcome = self._reconcile_mutation(
                mutation, message_id,
                "SELECT meta, frames_file, version FROM messages WHERE id = ?",
                (message_id,), classify, expected_version=expected_version,
                operational_exc=exc, fallback_observed_version=expected_version,
            )
            if outcome.state != "ambiguous":
                self._unlink_candidate_if_unreferenced(new_file, final_path)
        if outcome.state == "committed" and old_file and old_file != new_file:
            self._unlink_best_effort(FRAMES_DIR / old_file)
        return outcome

    def mark_frame_publication_failed(self, message_id, *, expected_version, failure_meta):
        mutation = "mark_frame_publication_failed"
        try:
            meta_sql = json.dumps(failure_meta, ensure_ascii=False)
        except BaseException as exc:
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        intended_version = expected_version + 1
        try:
            with self._independent_read_connection() as read:
                initial = read.execute(
                    "SELECT meta, frames_file, version FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
        except BaseException as exc:
            return self._mutation_outcome(
                "ambiguous", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        if initial is None:
            return self._mutation_outcome(
                "superseded", mutation, entity_id=message_id,
                expected_version=expected_version,
            )
        if initial["version"] != expected_version:
            return self._mutation_outcome(
                "superseded", mutation, entity_id=message_id,
                expected_version=expected_version, observed_version=initial["version"],
                message="frame publication state was replaced",
            )
        original = tuple(initial[key] for key in ("meta", "frames_file", "version"))
        intended = (meta_sql, None, intended_version)

        def classify(row):
            if row is None:
                return "superseded", None, None, None
            observed = row["version"]
            actual = tuple(row[key] for key in ("meta", "frames_file", "version"))
            if actual == intended:
                return "committed", observed, None, None
            if actual == original:
                return "not_committed", observed, None, None
            return "superseded", observed, None, "frame publication state was replaced"

        last_outcome = None
        for attempt in range(2):
            conn = None
            try:
                conn = self._conn()
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    "UPDATE messages SET meta = ?, frames_file = NULL, version = version + 1 "
                    "WHERE id = ? AND version = ?",
                    (meta_sql, message_id, expected_version),
                )
                if cur.rowcount != 1:
                    self._abort_transaction_or_discard(conn)
                    self._discard_conn(conn)
                    return self._reconcile_mutation(
                        mutation, message_id,
                        "SELECT meta, frames_file, version FROM messages WHERE id = ?",
                        (message_id,), classify, expected_version=expected_version,
                    )
                conn.commit()
                return self._mutation_outcome(
                    "committed", mutation, entity_id=message_id,
                    expected_version=expected_version, observed_version=intended_version,
                )
            except BaseException as exc:
                if conn is not None:
                    self._abort_transaction_or_discard(conn)
                    self._discard_conn(conn)
                last_outcome = self._reconcile_mutation(
                    mutation, message_id,
                    "SELECT meta, frames_file, version FROM messages WHERE id = ?",
                    (message_id,), classify, expected_version=expected_version,
                    operational_exc=exc,
                )
                if last_outcome.state != "not_committed" or attempt == 1:
                    return last_outcome
        return last_outcome

    def save_frames(self, message_id, frames, layers, k, *, expected_version=None, complete_meta=None, frame_descriptor=None):
        mutation = "save_frames"
        try:
            conn = self._conn()
        except BaseException as exc:
            return self._mutation_outcome(
                "ambiguous", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        try:
            row = conn.execute(
                "SELECT meta, version, frames_file FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        except BaseException as exc:
            return self._mutation_outcome(
                "ambiguous", mutation, entity_id=message_id,
                expected_version=expected_version, exc=exc,
            )
        if row is None:
            return self._mutation_outcome(
                "superseded", mutation, entity_id=message_id,
                expected_version=expected_version,
            )
        baseline = row["version"] if expected_version is None else expected_version
        old_file = row["frames_file"]
        original = (row["frames_file"], row["meta"], row["version"])
        if row["version"] != baseline:
            return self._mutation_outcome(
                "stale", mutation, entity_id=message_id,
                expected_version=baseline, observed_version=row["version"],
                message="message changed before frames attached",
            )
        try:
            filename, final_path = self._write_unique_frame_candidate(
                message_id, baseline,
                self._pack_frames_blob(frames, layers, k, frame_descriptor),
            )
            self._validate_frame_candidate(final_path, message_id)
        except BaseException as exc:
            if "final_path" in locals():
                self._unlink_best_effort(final_path)
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=baseline, observed_version=row["version"], exc=exc,
            )
        try:
            meta_sql = (
                json.dumps(complete_meta, ensure_ascii=False)
                if complete_meta is not None else row["meta"]
            )
        except BaseException as exc:
            self._unlink_candidate_if_unreferenced(filename, final_path)
            return self._mutation_outcome(
                "not_committed", mutation, entity_id=message_id,
                expected_version=baseline, observed_version=row["version"], exc=exc,
            )
        intended = (filename, meta_sql, baseline + 1)

        def classify(current):
            if current is None:
                return "superseded", None, None, None
            observed = current["version"]
            actual = tuple(current[key] for key in ("frames_file", "meta", "version"))
            if actual == intended:
                return "committed", observed, filename, None
            if actual == original:
                return "not_committed", observed, None, None
            if observed != baseline:
                return "stale", observed, None, "message changed before frames attached"
            return "ambiguous", observed, filename, "frame pointer state is inconsistent"

        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE messages SET frames_file = ?, meta = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (filename, meta_sql, message_id, baseline),
            )
            if cur.rowcount != 1:
                self._abort_transaction_or_discard(conn)
                self._discard_conn(conn)
                outcome = self._reconcile_mutation(
                    mutation, message_id,
                    "SELECT frames_file, meta, version FROM messages WHERE id = ?",
                    (message_id,), classify, expected_version=baseline,
                    fallback_value=filename,
                    fallback_observed_version=row["version"],
                )
            else:
                conn.commit()
                outcome = self._mutation_outcome(
                    "committed", mutation, entity_id=message_id,
                    expected_version=baseline, observed_version=baseline + 1,
                    value=filename,
                )
        except BaseException as exc:
            self._abort_transaction_or_discard(conn)
            self._discard_conn(conn)
            outcome = self._reconcile_mutation(
                mutation, message_id,
                "SELECT frames_file, meta, version FROM messages WHERE id = ?",
                (message_id,), classify, expected_version=baseline,
                operational_exc=exc,
                fallback_value=filename,
                fallback_observed_version=row["version"],
            )

        if outcome.state == "committed":
            if old_file and old_file != filename:
                self._unlink_best_effort(FRAMES_DIR / old_file)
        elif outcome.state != "ambiguous":
            self._unlink_candidate_if_unreferenced(filename, final_path)
        return outcome

    def _validate_frame_candidate(self, final_path, message_id):
        try:
            data = msgpack.unpackb(final_path.read_bytes(), strict_map_key=False)
            self._decode_frames_data(data, message_id)
        except FrameStorageError:
            raise
        except Exception as exc:
            raise FrameStorageError(f"candidate frame archive for message {message_id} is invalid: {exc}") from exc

    def _decode_frames_data(self, data, message_id):
        try:
            if not isinstance(data, dict) or data.get("version") not in (1, 2, 3):
                raise ValueError("unsupported frame archive version")
            archive_version = data["version"]
            pin_publication_state = (
                data.get("pin_publication_state", "unavailable")
                if archive_version >= 3 else "unavailable"
            )
            if pin_publication_state not in {"pending", "published", "unavailable"}:
                raise ValueError("invalid pin publication state")
            pin_gen_id = data.get("pin_gen_id") if pin_publication_state == "published" else None
            pin_generation_run_id = (
                data.get("pin_generation_run_id") if pin_publication_state == "published" else None
            )
            descriptor = data.get("descriptor") if archive_version >= 2 else None
            if descriptor is not None and not isinstance(descriptor, dict):
                raise ValueError("invalid frame archive descriptor")
            if isinstance(data.get("k"), bool) or not isinstance(data.get("k"), int) or data["k"] < 0:
                raise ValueError("invalid frame archive k")
            if (
                not isinstance(data.get("layers"), list)
                or any(isinstance(l, bool) or not isinstance(l, int) or l < 0 for l in data["layers"])
            ):
                raise ValueError("invalid frame archive layers")
            if len(set(data["layers"])) != len(data["layers"]):
                raise ValueError("duplicate frame archive layers")
            vocab = data["vocab"]
            if not isinstance(vocab, dict) or not isinstance(data.get("frames"), list):
                raise ValueError("invalid frame archive structure")
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in vocab.items()):
                raise ValueError("invalid frame archive vocabulary")
            frames = []
            last_pos = -1
            archive_run_id = None
            for entry in data["frames"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("layers"), dict):
                    raise ValueError("invalid frame entry")
                if entry.get("phase") not in SUPPORTED_FRAME_PHASES:
                    raise ValueError("invalid frame phase")
                if (
                    isinstance(entry.get("pos"), bool)
                    or not isinstance(entry.get("pos"), int)
                    or entry["pos"] < 0
                    or entry["pos"] < last_pos
                ):
                    raise ValueError("invalid frame position")
                last_pos = entry["pos"]
                if isinstance(entry.get("token_id"), bool) or not isinstance(entry.get("token_id"), int) or entry["token_id"] < 0:
                    raise ValueError("invalid frame token id")
                normalized_layers = {}
                for raw_layer, layer_data in entry["layers"].items():
                    layer_id = _normalize_frame_layer_key(raw_layer)
                    if layer_id in normalized_layers:
                        raise ValueError("duplicate frame layer key")
                    normalized_layers[layer_id] = layer_data
                if set(normalized_layers) != set(data["layers"]):
                    raise ValueError("frame layers do not match archive layers")
                frame_gen = entry.get("gen") if archive_version >= 2 else data.get("gen")
                if frame_gen is not None and (
                    isinstance(frame_gen, bool) or not isinstance(frame_gen, int) or frame_gen < 0
                ):
                    raise ValueError("invalid frame generation index")
                generation_run_id = entry.get("generation_run_id") if archive_version >= 3 else None
                if archive_version >= 3 and (
                    not isinstance(generation_run_id, str)
                    or len(generation_run_id) != 32
                    or any(ch not in "0123456789abcdef" for ch in generation_run_id)
                ):
                    raise ValueError("invalid frame generation run id")
                if archive_version >= 3:
                    if archive_run_id is None:
                        archive_run_id = generation_run_id
                    elif generation_run_id != archive_run_id:
                        raise ValueError("inconsistent frame generation run id")
                frame = {
                    "type": "frame",
                    "phase": entry["phase"],
                    "pos": entry["pos"],
                    "token_id": entry["token_id"],
                    "tok": vocab.get(str(entry["token_id"]), ""),
                    "gen": frame_gen if pin_publication_state == "published" else None,
                    "generation_run_id": generation_run_id if pin_publication_state == "published" else None,
                    "layers": {},
                }
                for layer, d in normalized_layers.items():
                    if not isinstance(d, dict):
                        raise ValueError("invalid layer entry")
                    ids_arr = np.frombuffer(d["ids"], np.int32)
                    p_arr = np.frombuffer(d["p"], np.float16)
                    m_ids_arr = np.frombuffer(d["m_ids"], np.int32)
                    m_p_arr = np.frombuffer(d["m_p"], np.float16)
                    m_rank_arr = np.frombuffer(d["m_rank"], np.int32)
                    if len(ids_arr) != len(p_arr) or len(m_ids_arr) != len(m_p_arr) or len(m_ids_arr) != len(m_rank_arr):
                        raise ValueError("frame vector length mismatch")
                    if len(ids_arr) != data["k"] or len(m_ids_arr) != data["k"]:
                        raise ValueError("frame vector length does not match k")
                    if not np.all(np.isfinite(p_arr.astype(np.float32))) or not np.all(np.isfinite(m_p_arr.astype(np.float32))):
                        raise ValueError("frame probabilities must be finite")
                    ids = ids_arr.tolist()
                    m_ids = m_ids_arr.tolist()
                    frame["layers"][layer] = {
                        "ids": ids,
                        "p": [round(float(v), 5) for v in p_arr],
                        "strs": [vocab.get(str(t), "") for t in ids],
                        "m_ids": m_ids,
                        "m_p": [round(float(v), 5) for v in m_p_arr],
                        "m_rank": m_rank_arr.tolist(),
                        "m_strs": [vocab.get(str(t), "") for t in m_ids],
                    }
                frames.append(frame)
            if pin_publication_state == "published":
                if (
                    isinstance(pin_gen_id, bool) or not isinstance(pin_gen_id, int) or pin_gen_id < 0
                    or pin_generation_run_id != archive_run_id
                ):
                    raise ValueError("published frame archive pin identity is invalid")
            return {
                "version": archive_version,
                "k": data["k"],
                "layers": [int(l) for l in data["layers"]],
                "descriptor": dict(descriptor) if descriptor is not None else None,
                "pin_publication_state": pin_publication_state,
                "pin_gen_id": pin_gen_id,
                "pin_generation_run_id": pin_generation_run_id,
                "frames": frames,
            }
        except Exception as exc:
            raise FrameStorageError(f"corrupt frame archive for message {message_id}: {exc}") from exc


    def load_frames(self, message_id):
        conn = self._conn()
        last_missing = None
        for _attempt in range(2):
            row = conn.execute(
                "SELECT frames_file, version FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            if row is None or row["frames_file"] is None:
                raise FramesNotAttached(f"no frames for message {message_id}")
            frame_file, version = row["frames_file"], row["version"]
            try:
                data = msgpack.unpackb((FRAMES_DIR / frame_file).read_bytes(), strict_map_key=False)
                break
            except FileNotFoundError as exc:
                fresh = conn.execute(
                    "SELECT frames_file, version FROM messages WHERE id = ?", (message_id,)
                ).fetchone()
                if fresh is not None and (
                    fresh["frames_file"] != frame_file or fresh["version"] != version
                ):
                    last_missing = exc
                    continue
                raise FrameFileMissing(f"frame archive missing for message {message_id}") from exc
            except Exception as exc:
                raise FrameStorageError(f"could not read frame archive for message {message_id}: {exc}") from exc
        else:
            raise FramePointerTurnover(f"frame archive changed while reading message {message_id}") from last_missing
        return self._decode_frames_data(data, message_id)

    def export(self, conversation_id, fmt="json", include_frames=False):
        conv = self.get_conversation(conversation_id)
        if include_frames:
            for message in conv["messages"]:
                if message["has_frames"]:
                    try:
                        message["frames"] = self.load_frames(message["id"])
                    except FramesNotAttached:
                        pass
        if fmt == "json":
            return json.dumps(conv, ensure_ascii=False, indent=1), "application/json"
        lines = [f"# {conv['title']}", ""]
        if conv["tags"]:
            lines.append(f"tags: {', '.join(conv['tags'])}")
            lines.append("")
        for message in conv["messages"]:
            meta = message.get("meta") or {}
            head = f"**{message['role']}** (#{message['id']}"
            if message["parent_id"] is not None:
                head += f" ← #{message['parent_id']}"
            head += ")"
            if meta.get("model_id"):
                head += f" — {meta['model_id']} · {meta.get('quant') or meta.get('dtype')}"
            lines.append(head)
            lines.append("")
            lines.append(message["content"])
            lines.append("")
            if include_frames and message.get("frames"):
                lines.append(f"> {len(message['frames']['frames'])} lens frames (layers {message['frames']['layers']})")
                lines.append("")
        return "\n".join(lines), "text/markdown"
