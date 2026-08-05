import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

import msgpack
import numpy as np

import config

DB_PATH = config.DATA_DIR / "jlens.db"
FRAMES_DIR = config.DATA_DIR / "frames"

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

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def create_conversation(self, title, tags=None):
        conn = self._conn()
        now = _now()
        cur = conn.execute(
            "INSERT INTO conversations (title, tags, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, json.dumps(tags or []), now, now),
        )
        conn.commit()
        return cur.lastrowid

    def update_conversation(self, conversation_id, title=None, tags=None):
        conn = self._conn()
        if title is not None:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), conversation_id),
            )
        if tags is not None:
            conn.execute(
                "UPDATE conversations SET tags = ?, updated_at = ? WHERE id = ?",
                (json.dumps(tags), _now(), conversation_id),
            )
        conn.commit()

    def delete_conversation(self, conversation_id):
        conn = self._conn()
        rows = conn.execute(
            "SELECT frames_file FROM messages WHERE conversation_id = ? AND frames_file IS NOT NULL",
            (conversation_id,),
        ).fetchall()
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        for row in rows:
            (FRAMES_DIR / row["frames_file"]).unlink(missing_ok=True)

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

    def add_message(self, conversation_id, parent_id, role, content, meta=None):
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, parent_id, role, content, meta, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                parent_id,
                role,
                content,
                json.dumps(meta, ensure_ascii=False) if meta else None,
                _now(),
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )
        conn.commit()
        return cur.lastrowid

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

    def update_message(self, message_id, content, meta=None, *, clear_frames=False):
        """Rewrite a message and increment its durable row version.

        Assistant edits may clear derived frame/provenance artifacts atomically;
        stale continuations then fail their versioned CAS instead of overwriting
        an acknowledged edit.
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT conversation_id, frames_file FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown message {message_id}")
        meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
        frames_file = None if clear_frames else row["frames_file"]
        conn.execute(
            "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 WHERE id = ?",
            (content, meta_json, frames_file, message_id),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), row["conversation_id"]),
        )
        conn.commit()
        if clear_frames and row["frames_file"]:
            (FRAMES_DIR / row["frames_file"]).unlink(missing_ok=True)

    def update_message_if_unchanged(self, message_id, expected_version, content, meta=None):
        return self.update_message_and_frames_if_unchanged(
            message_id, expected_version, content, meta, frames=None
        )

    def _write_unique_frame_candidate(self, message_id, expected_version, frame_blob):
        frame_file = f"{message_id}-v{int(expected_version) + 1}-{uuid.uuid4().hex}.msgpack"
        tmp = FRAMES_DIR / f".{frame_file}.tmp-{uuid.uuid4().hex}"
        final = FRAMES_DIR / frame_file
        with tmp.open("wb") as fh:
            fh.write(frame_blob)
            fh.flush()
            try:
                import os
                os.fsync(fh.fileno())
            except OSError:
                pass
        tmp.replace(final)
        return frame_file, final

    def update_message_and_frames_if_unchanged(self, message_id, expected_version, content, meta=None, *, frames=None, layers=None, k=0):
        frame_file = None
        final_path = None
        if frames:
            frame_blob = self._pack_frames_blob(frames, layers or [], k)
            frame_file, final_path = self._write_unique_frame_candidate(message_id, expected_version, frame_blob)
        conn = self._conn()
        old_file = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT conversation_id, frames_file FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise ValueError(f"unknown message {message_id}")
            old_file = row["frames_file"]
            target_frame = frame_file if frame_file is not None else old_file
            meta_json = json.dumps(meta, ensure_ascii=False) if meta is not None else None
            cur = conn.execute(
                "UPDATE messages SET content = ?, meta = ?, frames_file = ?, version = version + 1 "
                "WHERE id = ? AND version = ?",
                (content, meta_json, target_frame, message_id, expected_version),
            )
            if cur.rowcount != 1:
                conn.rollback()
                if final_path is not None:
                    final_path.unlink(missing_ok=True)
                return False
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (_now(), row["conversation_id"]),
            )
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            if final_path is not None:
                final_path.unlink(missing_ok=True)
            raise
        if frame_file and old_file and old_file != frame_file:
            (FRAMES_DIR / old_file).unlink(missing_ok=True)
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
    def _pack_frames_blob(frames, layers, k):
        vocab = {}
        packed = []
        for frame in frames:
            vocab[frame["token_id"]] = frame["tok"]
            entry = {
                "pos": frame["pos"],
                "phase": frame["phase"],
                "token_id": frame["token_id"],
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
            "version": 1,
            "k": k,
            "gen": frames[-1].get("gen") if frames else None,
            "layers": [int(l) for l in layers],
            "frames": packed,
            "vocab": {str(t): s for t, s in vocab.items()},
        })

    def save_frames(self, message_id, frames, layers, k):
        conn = self._conn()
        row = conn.execute(
            "SELECT version, frames_file FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown message {message_id}")
        filename, final_path = self._write_unique_frame_candidate(
            message_id, row["version"], self._pack_frames_blob(frames, layers, k)
        )
        try:
            conn.execute(
                "UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ? AND version = ?",
                (filename, message_id, row["version"]),
            )
            conn.commit()
        except Exception:
            final_path.unlink(missing_ok=True)
            raise
        if row["frames_file"] and row["frames_file"] != filename:
            (FRAMES_DIR / row["frames_file"]).unlink(missing_ok=True)
        return filename

    def load_frames(self, message_id):
        conn = self._conn()
        row = conn.execute(
            "SELECT frames_file FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if row is None or row["frames_file"] is None:
            raise ValueError(f"no frames for message {message_id}")
        data = msgpack.unpackb((FRAMES_DIR / row["frames_file"]).read_bytes(), strict_map_key=False)
        vocab = data["vocab"]
        frames = []
        for entry in data["frames"]:
            frame = {
                "type": "frame",
                "phase": entry["phase"],
                "pos": entry["pos"],
                "token_id": entry["token_id"],
                "tok": vocab.get(str(entry["token_id"]), ""),
                "gen": data.get("gen"),
                "layers": {},
            }
            for layer, d in entry["layers"].items():
                ids = np.frombuffer(d["ids"], np.int32).tolist()
                m_ids = np.frombuffer(d["m_ids"], np.int32).tolist()
                frame["layers"][layer] = {
                    "ids": ids,
                    "p": [round(float(v), 5) for v in np.frombuffer(d["p"], np.float16)],
                    "strs": [vocab.get(str(t), "") for t in ids],
                    "m_ids": m_ids,
                    "m_p": [round(float(v), 5) for v in np.frombuffer(d["m_p"], np.float16)],
                    "m_rank": np.frombuffer(d["m_rank"], np.int32).tolist(),
                    "m_strs": [vocab.get(str(t), "") for t in m_ids],
                }
            frames.append(frame)
        return {"k": data["k"], "layers": data["layers"], "frames": frames}

    def export(self, conversation_id, fmt="json", include_frames=False):
        conv = self.get_conversation(conversation_id)
        if include_frames:
            for message in conv["messages"]:
                if message["has_frames"]:
                    try:
                        message["frames"] = self.load_frames(message["id"])
                    except ValueError:
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
