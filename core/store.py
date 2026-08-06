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


class StaleMessageUpdate(RuntimeError):
    """A versioned message update lost its durable CAS race."""


class FrameStorageError(RuntimeError):
    """Frame archive publication or loading failed coherently."""


class FramesNotAttached(ValueError):
    """No frame archive is currently attached to the message."""


class FramePointerTurnover(RuntimeError):
    """The committed frame pointer changed too often while reading."""


class FrameFileMissing(FrameStorageError):
    """The committed frame archive is missing from disk."""


class StorageMutationError(RuntimeError):
    """A durable store mutation could not be classified as successful."""


class StorageCommitAmbiguous(StorageMutationError):
    """A commit result could not be reconciled through an independent read."""


class StorageMutationSuperseded(StorageMutationError):
    """A versioned mutation was superseded by a newer durable row."""


MAX_FRAME_ERROR_CHARS = 512


@dataclass(frozen=True)
class StorageMutationOutcome:
    state: str
    mutation: str
    message_id: int | None = None
    expected_version: int | None = None
    observed_version: int | None = None
    error_kind: str | None = None
    error_message: str | None = None

    @property
    def committed(self):
        return self.state == "committed"

    @property
    def status(self):
        return self.state

    def __bool__(self):
        return self.committed


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
    updated_at TEXT NOT NULL
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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        conn = self._conn()
        conn.executescript(SCHEMA)
        self._ensure_message_version(conn)
        conn.commit()

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
        if getattr(self._local, "conn", None) is conn or conn is None:
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

    def _mutation_outcome(self, state, mutation, *, message_id=None, expected_version=None,
                          observed_version=None, exc=None, message=None):
        return StorageMutationOutcome(
            state=state,
            mutation=mutation,
            message_id=message_id,
            expected_version=expected_version,
            observed_version=observed_version,
            error_kind=exc.__class__.__name__ if exc is not None else None,
            error_message=self._bounded_error(exc) if exc is not None else (
                str(message)[:MAX_FRAME_ERROR_CHARS] if message is not None else None
            ),
        )

    def create_conversation(self, title, tags=None):
        conn = self._conn()
        now = _now()
        tags_sql = json.dumps(tags or [])
        conversation_id = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM conversations")
            conversation_id = int(cur.fetchone()[0])
            conn.execute(
                "INSERT INTO conversations (id, title, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, title, tags_sql, now, now),
            )
            conn.commit()
            return conversation_id
        except Exception as exc:
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    row = read.execute(
                        "SELECT id, title, tags, created_at, updated_at FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone() if conversation_id is not None else None
                if row is not None and (
                    row["title"] == title
                    and row["tags"] == tags_sql
                    and row["created_at"] == now
                    and row["updated_at"] == now
                ):
                    return conversation_id
            except Exception as reconcile_exc:
                raise StorageCommitAmbiguous(
                    f"could not reconcile conversation create: {self._bounded_error(reconcile_exc)}"
                ) from None
            raise StorageCommitAmbiguous(
                f"conversation create did not commit: {self._bounded_error(exc)}"
            ) from None

    def update_conversation(self, conversation_id, title=None, tags=None):
        conn = self._conn()
        new_title = None
        new_tags = None
        now = _now()
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT title, tags FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if current is None:
                self._abort_transaction_or_discard(conn)
                raise ValueError(f"unknown conversation {conversation_id}")
            new_title = current["title"] if title is None else title
            new_tags = current["tags"] if tags is None else json.dumps(tags)
            cur = conn.execute(
                "UPDATE conversations SET title = ?, tags = ?, updated_at = ? WHERE id = ?",
                (new_title, new_tags, now, conversation_id),
            )
            if cur.rowcount != 1:
                self._abort_transaction_or_discard(conn)
                raise ValueError(f"unknown conversation {conversation_id}")
            conn.commit()
            return self._mutation_outcome("committed", "update_conversation", message_id=conversation_id)
        except Exception as exc:
            if isinstance(exc, StaleMessageUpdate):
                raise
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    row = read.execute(
                        "SELECT title, tags, updated_at FROM conversations WHERE id = ?",
                        (conversation_id,),
                    ).fetchone()
                if row is not None and row["title"] == new_title and row["tags"] == new_tags and row["updated_at"] == now:
                    return self._mutation_outcome("committed", "update_conversation", message_id=conversation_id)
                if row is None:
                    return self._mutation_outcome("superseded", "update_conversation", message_id=conversation_id, exc=exc)
            except Exception as reconcile_exc:
                raise StorageCommitAmbiguous(
                    f"could not reconcile conversation update {conversation_id}: {self._bounded_error(reconcile_exc)}"
                ) from None
            raise

    def delete_conversation(self, conversation_id):
        conn = self._conn()
        rows = []
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT frames_file FROM messages WHERE conversation_id = ? AND frames_file IS NOT NULL",
                (conversation_id,),
            ).fetchall()
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
        except Exception as exc:
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    row = read.execute(
                        "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
                    ).fetchone()
                if row is None:
                    for frame_row in rows:
                        self._unlink_best_effort(FRAMES_DIR / frame_row["frames_file"])
                    return self._mutation_outcome("committed", "delete_conversation", message_id=conversation_id)
            except Exception as reconcile_exc:
                raise StorageCommitAmbiguous(
                    f"could not reconcile conversation delete {conversation_id}: {self._bounded_error(reconcile_exc)}"
                ) from None
            raise StorageCommitAmbiguous(
                f"conversation delete {conversation_id} outcome ambiguous: {self._bounded_error(exc)}"
            ) from None
        for row in rows:
            self._unlink_best_effort(FRAMES_DIR / row["frames_file"])
        return self._mutation_outcome("committed", "delete_conversation", message_id=conversation_id)

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
        conn = self._conn()
        message_id = None
        now = _now()
        meta_sql = json.dumps(meta, ensure_ascii=False) if meta else None
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM messages")
            message_id = int(cur.fetchone()[0])
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
        except Exception as exc:
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    if message_id is None:
                        raise
                    row = read.execute(
                        "SELECT id, conversation_id, parent_id, role, content, meta, frames_file, version "
                        "FROM messages WHERE id = ?",
                        (message_id,),
                    ).fetchone()
                    if row is not None and (
                        row["conversation_id"] == conversation_id
                        and row["parent_id"] == parent_id
                        and row["role"] == role
                        and row["content"] == content
                        and row["meta"] == meta_sql
                        and row["frames_file"] is None
                        and row["version"] == 0
                    ):
                        return (row["id"], row["version"]) if return_version else row["id"]
                    if row is not None:
                        raise StorageCommitAmbiguous(f"message insert {message_id} committed with mismatched state")
            except StorageCommitAmbiguous:
                raise
            except Exception:
                raise
            raise
        return (message_id, 0) if return_version else message_id

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
        conn = self._conn()
        row = conn.execute(
            "SELECT conversation_id, frames_file, version FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown message {message_id}")
        meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
        frames_file = None if clear_frames else row["frames_file"]
        baseline = row["version"] if expected_version is None else expected_version
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (content, meta_json, frames_file, message_id, baseline),
            )
            if cur.rowcount != 1:
                self._abort_transaction_or_discard(conn)
                raise StaleMessageUpdate(f"message {message_id} changed before edit")
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), row["conversation_id"]),
            )
            conn.commit()
        except StaleMessageUpdate:
            raise
        except Exception as exc:
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    current = read.execute(
                        "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                        (message_id,),
                    ).fetchone()
                    if current is not None and (
                        current["content"] == content
                        and current["meta"] == meta_json
                        and current["frames_file"] == frames_file
                        and current["version"] == baseline + 1
                    ):
                        pass
                    else:
                        raise
            except Exception:
                raise
        if clear_frames and row["frames_file"]:
            self._unlink_best_effort(FRAMES_DIR / row["frames_file"])

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

    def update_message_and_frames_if_unchanged(self, message_id, expected_version, content, meta=None, *, frames=None, layers=None, k=0, clear_frames=False, frame_descriptor=None):
        frame_file = None
        final_path = None
        if frames:
            frame_blob = self._pack_frames_blob(frames, layers or [], k, frame_descriptor)
            frame_file, final_path = self._write_unique_frame_candidate(message_id, expected_version, frame_blob)
            try:
                self._validate_frame_candidate(final_path, message_id)
            except Exception:
                self._unlink_best_effort(final_path)
                raise
        conn = self._conn()
        old_file = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT conversation_id, frames_file FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                self._rollback_or_discard(conn)
                raise ValueError(f"unknown message {message_id}")
            old_file = row["frames_file"]
            target_frame = None if clear_frames else (frame_file if frame_file is not None else old_file)
            meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
            cur = conn.execute(
                "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (content, meta_json, target_frame, message_id, expected_version),
            )
            if cur.rowcount != 1:
                self._rollback_or_discard(conn)
                if final_path is not None:
                    self._unlink_best_effort(final_path)
                return False
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), row["conversation_id"]),
            )
            conn.commit()
        except Exception as exc:
            self._abort_transaction_or_discard(conn)
            intended_meta = json.dumps(meta, ensure_ascii=False) if meta is not None else None
            try:
                with self._independent_read_connection() as read:
                    current = read.execute(
                        "SELECT content, meta, frames_file, version FROM messages WHERE id = ?",
                        (message_id,),
                    ).fetchone()
                    if current is not None and (
                        current["content"] == content
                        and current["meta"] == intended_meta
                        and current["frames_file"] == target_frame
                        and current["version"] == expected_version + 1
                    ):
                        if frame_file and old_file and old_file != frame_file:
                            self._unlink_best_effort(FRAMES_DIR / old_file)
                        return True
                    if current is not None and current["version"] != expected_version:
                        if final_path is not None:
                            self._unlink_candidate_if_unreferenced(frame_file, final_path)
                        return False
            except Exception as reconcile_exc:
                if final_path is not None and not self._candidate_referenced(frame_file):
                    log.warning("frame mutation reconciliation failed; preserving candidate if referenced", exc_info=True)
                raise StorageCommitAmbiguous(
                    f"could not reconcile message {message_id} mutation: {self._bounded_error(reconcile_exc)}"
                ) from reconcile_exc
            if final_path is not None:
                self._unlink_candidate_if_unreferenced(frame_file, final_path)
            raise
        if (frame_file or clear_frames) and old_file and old_file != frame_file:
            self._unlink_best_effort(FRAMES_DIR / old_file)
        return True

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
    def _pack_frames_blob(frames, layers, k, descriptor=None):
        vocab = {}
        packed = []
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
            "version": 3,
            "k": k,
            "descriptor": dict(descriptor) if descriptor is not None else None,
            "layers": [int(l) for l in layers],
            "frames": packed,
            "vocab": {str(t): s for t, s in vocab.items()},
        })

    def mark_frame_publication_failed(self, message_id, *, expected_version, failure_meta):
        conn = self._conn()
        meta_sql = json.dumps(failure_meta, ensure_ascii=False)
        intended_version = expected_version + 1
        try:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE messages SET meta = ?, frames_file = NULL, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (meta_sql, message_id, expected_version),
            )
            if cur.rowcount != 1:
                self._rollback_or_discard(conn)
                return False
            conn.commit()
            return True
        except Exception:
            self._abort_transaction_or_discard(conn)
            try:
                with self._independent_read_connection() as read:
                    row = read.execute(
                        "SELECT meta, frames_file, version FROM messages WHERE id = ?",
                        (message_id,),
                    ).fetchone()
                    if row is not None and (
                        row["meta"] == meta_sql
                        and row["frames_file"] is None
                        and row["version"] == intended_version
                    ):
                        return True
                    if row is not None and row["version"] != expected_version:
                        return False
                    if row is not None and row["version"] == expected_version:
                        retry_conn = self._conn()
                        try:
                            retry_conn.execute("BEGIN IMMEDIATE")
                            cur = retry_conn.execute(
                                "UPDATE messages SET meta = ?, frames_file = NULL, version = version + 1 "
                                "WHERE id = ? AND version = ?",
                                (meta_sql, message_id, expected_version),
                            )
                            if cur.rowcount == 1:
                                retry_conn.commit()
                                return True
                            self._abort_transaction_or_discard(retry_conn)
                            return False
                        except Exception:
                            self._abort_transaction_or_discard(retry_conn)
                            raise
            except Exception as reconcile_exc:
                raise StorageCommitAmbiguous(
                    f"could not reconcile frame-failure terminalization for message {message_id}: "
                    f"{self._bounded_error(reconcile_exc)}"
                ) from reconcile_exc
            raise StorageCommitAmbiguous(f"message {message_id} frame-failure terminalization is ambiguous")

    def save_frames(self, message_id, frames, layers, k, *, expected_version=None, complete_meta=None, frame_descriptor=None):
        conn = self._conn()
        row = conn.execute(
            "SELECT version, frames_file FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown message {message_id}")
        baseline = row["version"] if expected_version is None else expected_version
        filename, final_path = self._write_unique_frame_candidate(
            message_id, baseline, self._pack_frames_blob(frames, layers, k, frame_descriptor)
        )
        try:
            self._validate_frame_candidate(final_path, message_id)
        except Exception:
            self._unlink_best_effort(final_path)
            raise
        old_file = row["frames_file"]
        committed = False
        try:
            conn.execute("BEGIN IMMEDIATE")
            meta_sql = json.dumps(complete_meta, ensure_ascii=False) if complete_meta is not None else None
            if complete_meta is not None:
                cur = conn.execute(
                    "UPDATE messages SET frames_file = ?, meta = ?, version = version + 1 WHERE id = ? AND version = ?",
                    (filename, meta_sql, message_id, baseline),
                )
            else:
                cur = conn.execute(
                    "UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ? AND version = ?",
                    (filename, message_id, baseline),
                )
            if cur.rowcount != 1:
                self._rollback_or_discard(conn)
                self._unlink_best_effort(final_path)
                raise StaleMessageUpdate(f"message {message_id} changed before frames attached")
            conn.commit()
            committed = True
        except Exception as exc:
            if isinstance(exc, StaleMessageUpdate):
                self._abort_transaction_or_discard(conn)
                self._unlink_candidate_if_unreferenced(filename, final_path)
                raise
            try:
                self._abort_transaction_or_discard(conn)
                with self._independent_read_connection() as read_conn:
                    current = read_conn.execute(
                        "SELECT frames_file, meta, version FROM messages WHERE id = ?",
                        (message_id,),
                    ).fetchone()
                meta_matches = complete_meta is None or (
                    current is not None and current["meta"] == json.dumps(complete_meta, ensure_ascii=False)
                )
                if current is not None and current["frames_file"] == filename and current["version"] == baseline + 1 and meta_matches:
                    if old_file and old_file != filename:
                        self._unlink_best_effort(FRAMES_DIR / old_file)
                    return filename
                self._unlink_candidate_if_unreferenced(filename, final_path)
            except Exception as reconcile_exc:
                if isinstance(reconcile_exc, StorageMutationError):
                    raise
                if not self._candidate_referenced(filename):
                    self._unlink_best_effort(final_path)
                raise StorageCommitAmbiguous(
                    f"could not reconcile frame save for message {message_id}: {self._bounded_error(reconcile_exc)}"
                ) from reconcile_exc
            raise
        if old_file and old_file != filename:
            self._unlink_best_effort(FRAMES_DIR / old_file)
        return filename

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
                if generation_run_id is not None and (
                    not isinstance(generation_run_id, str)
                    or len(generation_run_id) != 32
                    or any(ch not in "0123456789abcdef" for ch in generation_run_id)
                ):
                    raise ValueError("invalid frame generation run id")
                frame = {
                    "type": "frame",
                    "phase": entry["phase"],
                    "pos": entry["pos"],
                    "token_id": entry["token_id"],
                    "tok": vocab.get(str(entry["token_id"]), ""),
                    "gen": frame_gen,
                    "generation_run_id": generation_run_id,
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
            return {
                "version": archive_version,
                "k": data["k"],
                "layers": [int(l) for l in data["layers"]],
                "descriptor": dict(descriptor) if descriptor is not None else None,
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
