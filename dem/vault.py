"""Encrypted SQLite vault. Text payloads are Fernet-encrypted at rest.

Raw capture text has a TTL; approved records remain for reproducibility.
Everything is local — there is no network code in this module.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .crypto import Keyring
from .history_sources import HISTORY_SOURCES, SOURCE_PLACEHOLDERS
from .models import CaptureEvent, now_utc


class Vault:
    def __init__(self, vault_dir: Path, keys: Keyring):
        vault_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.keys = keys
        self.path = vault_dir / "vault.db"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self._schema()

    def _schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS captures (
              id TEXT PRIMARY KEY, source TEXT, surface TEXT, role TEXT,
              conversation_id TEXT, turn_index INTEGER, model_hint TEXT,
              timestamp TEXT, raw_hash TEXT, consent TEXT,
              text_enc BLOB, redacted_enc BLOB, redactions TEXT,
              quarantine_reason TEXT, metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_captures_consent ON captures(consent);
            CREATE INDEX IF NOT EXISTS idx_captures_conv ON captures(conversation_id);
            CREATE TABLE IF NOT EXISTS history_imports (turn_key TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS disabled_chats (
              conversation_id TEXT PRIMARY KEY, source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
              id TEXT PRIMARY KEY, source_tier TEXT, data_enc BLOB,
              raw_hash TEXT, created TEXT
            );
            CREATE TABLE IF NOT EXISTS consent_events (
              id TEXT PRIMARY KEY, ts TEXT, source TEXT, decision TEXT,
              policy_version TEXT, receipt_enc BLOB
            );
            CREATE TABLE IF NOT EXISTS audit (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event TEXT,
              detail_hash TEXT, policy_version TEXT
            );
            CREATE TABLE IF NOT EXISTS publish_log (
              id TEXT PRIMARY KEY, ts TEXT, repo TEXT, commit_sha TEXT,
              manifest_sha TEXT, count INTEGER
            );
            """
        )
        self.db.commit()

    # ---- captures -------------------------------------------------------
    def add_capture(self, event: CaptureEvent) -> None:
        with self.db:
            self._insert_capture(event)
        self.log_audit("capture", event.id, "consent-v3")

    def history_imported(self, turn_key: str) -> bool:
        return self.db.execute("SELECT 1 FROM history_imports WHERE turn_key=?", (turn_key,)).fetchone() is not None

    def add_history_pair(self, turn_key: str, prompt: CaptureEvent, response: CaptureEvent) -> str:
        """Claim and store both halves atomically when the conversation is enabled."""
        with self.db:
            inserted = self.db.execute(
                "INSERT OR IGNORE INTO history_imports "
                "SELECT ? WHERE NOT EXISTS (SELECT 1 FROM disabled_chats WHERE conversation_id=?)",
                (turn_key, prompt.conversation_id),
            )
            if not inserted.rowcount:
                if self.history_imported(turn_key):
                    return "already_imported"
                return "disabled_skipped"
            self._insert_capture(prompt)
            self._insert_capture(response)
        self.log_audit("history:import", turn_key, "consent-v3")
        return "imported"

    def disabled_history_chats(self, source: str) -> frozenset[str]:
        rows = self.db.execute("SELECT conversation_id FROM disabled_chats WHERE source=?", (source,))
        return frozenset(row["conversation_id"] for row in rows)

    def set_history_chat_enabled(self, conversation_id: str, enabled: bool) -> bool:
        with self.db:
            if enabled:
                result = self.db.execute(
                    "DELETE FROM disabled_chats WHERE conversation_id=?", (conversation_id,),
                )
            else:
                result = self.db.execute(
                    "INSERT OR IGNORE INTO disabled_chats "
                    "SELECT conversation_id, source FROM captures "
                    f"WHERE conversation_id=? AND source IN ({SOURCE_PLACEHOLDERS}) LIMIT 1",
                    (conversation_id, *HISTORY_SOURCES),
                )
        if result.rowcount:
            self.log_audit("chat:on" if enabled else "chat:off", conversation_id, "consent-v3")
        return bool(result.rowcount)

    def history_chats(self) -> list[dict]:
        rows = self.db.execute(
            f"""
            WITH sessions AS (
                SELECT conversation_id, source, COUNT(*) AS messages,
                       SUM(consent='pending' AND quarantine_reason IS NULL) AS pending,
                       SUM(quarantine_reason IS NOT NULL AND consent!='rejected') AS quarantined,
                       MAX(rowid) AS latest
                FROM captures WHERE source IN ({SOURCE_PLACEHOLDERS})
                GROUP BY conversation_id, source
                UNION ALL
                SELECT conversation_id, source, 0, 0, 0, 0 FROM disabled_chats AS d
                WHERE NOT EXISTS (
                    SELECT 1 FROM captures AS c WHERE c.conversation_id=d.conversation_id
                )
            )
            SELECT sessions.*, d.conversation_id IS NULL AS enabled, (
                SELECT redacted_enc FROM captures AS c
                WHERE c.conversation_id=sessions.conversation_id AND c.role='prompt'
                  AND c.quarantine_reason IS NULL AND c.consent!='rejected'
                ORDER BY c.turn_index, c.rowid LIMIT 1
            ) AS preview
            FROM sessions LEFT JOIN disabled_chats AS d USING (conversation_id, source)
            ORDER BY latest DESC, conversation_id
            """,
            HISTORY_SOURCES,
        )
        chats = []
        for row in rows:
            chat = dict(row)
            preview = chat.pop("preview")
            chat["title"] = " ".join(self.keys.decrypt(preview).split())[:140] if preview else ""
            chats.append(chat)
        return chats

    def history_messages(self, conversation_id: str) -> list[CaptureEvent]:
        rows = self.db.execute(
            f"SELECT * FROM captures WHERE conversation_id=? AND source IN ({SOURCE_PLACEHOLDERS}) "
            "ORDER BY turn_index, rowid",
            (conversation_id, *HISTORY_SOURCES),
        )
        return [self._row_to_event(row) for row in rows]

    def latest_decision(self, source: str, version: str) -> str | None:
        row = self.db.execute(
            "SELECT decision, policy_version FROM consent_events WHERE source=? ORDER BY rowid DESC LIMIT 1",
            (source,),
        ).fetchone()
        return row["decision"] if row and row["policy_version"] == version else None

    def _insert_capture(self, event: CaptureEvent) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO captures
               (id, source, surface, role, conversation_id, turn_index, model_hint,
                timestamp, raw_hash, consent, text_enc, redacted_enc, redactions,
                quarantine_reason, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event.id, event.source, event.surface, event.role,
                event.conversation_id, event.turn_index, event.model_hint,
                event.timestamp, event.raw_hash, event.consent,
                self.keys.encrypt(event.text),
                self.keys.encrypt(event.redacted_text or ""),
                json.dumps(event.redactions), event.quarantine_reason,
                json.dumps(event.metadata),
            ),
        )

    def _row_to_event(self, row: sqlite3.Row) -> CaptureEvent:
        return CaptureEvent(
            id=row["id"], source=row["source"], surface=row["surface"],
            role=row["role"], conversation_id=row["conversation_id"],
            turn_index=row["turn_index"], model_hint=row["model_hint"],
            timestamp=row["timestamp"], raw_hash=row["raw_hash"],
            consent=row["consent"],
            text=self.keys.decrypt(row["text_enc"]),
            redacted_text=self.keys.decrypt(row["redacted_enc"]) if row["redacted_enc"] else None,
            redactions=json.loads(row["redactions"] or "[]"),
            quarantine_reason=row["quarantine_reason"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_captures(self, consent: str | None = None, limit: int = 500,
                      *, include_history: bool = True) -> list[CaptureEvent]:
        history_filter = "" if include_history else f"AND source NOT IN ({SOURCE_PLACEHOLDERS})"
        history_args = () if include_history else HISTORY_SOURCES
        if consent:
            rows = self.db.execute(
                f"SELECT * FROM captures WHERE consent=? {history_filter} ORDER BY timestamp LIMIT ?",
                (consent, *history_args, limit),
            ).fetchall()
        else:
            rows = self.db.execute(
                f"SELECT * FROM captures WHERE 1=1 {history_filter} ORDER BY timestamp LIMIT ?",
                (*history_args, limit),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def approved_captures(self) -> list[CaptureEvent]:
        rows = self.db.execute(
            "SELECT * FROM captures WHERE consent IN ('approved','redacted') "
            "AND quarantine_reason IS NULL AND NOT EXISTS ("
            "SELECT 1 FROM disabled_chats WHERE disabled_chats.conversation_id=captures.conversation_id"
            ") ORDER BY timestamp, rowid"
        )
        return [self._row_to_event(row) for row in rows]

    def get_capture(self, capture_id: str) -> CaptureEvent | None:
        row = self.db.execute("SELECT * FROM captures WHERE id=?", (capture_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def update_consent(self, capture_id: str, decision: str) -> None:
        self.db.execute("UPDATE captures SET consent=? WHERE id=?", (decision, capture_id))
        self.db.commit()
        self.log_audit(f"consent:{decision}", capture_id, "consent-v3")

    def set_redaction(self, capture_id: str, text: str, redactions: list, reason: str | None) -> None:
        self.db.execute(
            "UPDATE captures SET redacted_enc=?, redactions=?, quarantine_reason=? WHERE id=?",
            (self.keys.encrypt(text), json.dumps(redactions), reason, capture_id),
        )
        self.db.commit()

    def edit_capture(self, capture_id: str, text: str) -> None:
        """Reviewer-edited final text replaces the redacted snapshot."""
        self.db.execute(
            "UPDATE captures SET redacted_enc=?, consent='redacted' WHERE id=?",
            (self.keys.encrypt(text), capture_id),
        )
        self.db.commit()
        self.log_audit("review:edit", capture_id, "consent-v3")

    def ttl_cleanup(self, days: int) -> int:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        cur = self.db.execute(
            "DELETE FROM captures WHERE consent='pending' AND timestamp < ?", (cutoff,)
        )
        self.db.commit()
        if cur.rowcount:
            self.log_audit("retention:purge", str(cur.rowcount), "consent-v3")
        return cur.rowcount

    # ---- records --------------------------------------------------------
    def add_record(self, record) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO records (id, source_tier, data_enc, raw_hash, created) "
            "VALUES (?,?,?,?,?)",
            (record.id, record.source_tier, self.keys.encrypt(json.dumps(record.to_dict())),
             record.raw_hash, now_utc()),
        )
        self.db.commit()

    def list_records(self, source_tier: str | None = None) -> list:
        from .models import DatasetRecord
        if source_tier:
            rows = self.db.execute(
                "SELECT data_enc FROM records WHERE source_tier=?", (source_tier,)
            ).fetchall()
        else:
            rows = self.db.execute("SELECT data_enc FROM records").fetchall()
        return [DatasetRecord.from_dict(json.loads(self.keys.decrypt(r["data_enc"]))) for r in rows]

    # ---- audit / receipts / logs ---------------------------------------
    def log_audit(self, event: str, detail: str, policy_version: str) -> None:
        digest = self.keys.tag(f"{event}:{detail}")
        self.db.execute(
            "INSERT INTO audit (ts, event, detail_hash, policy_version) VALUES (?,?,?,?)",
            (now_utc(), event, digest, policy_version),
        )
        self.db.commit()

    def add_consent_event(self, source: str, decision: str, receipt: str) -> None:
        from .models import ulid
        self.db.execute(
            "INSERT INTO consent_events VALUES (?,?,?,?,?,?)",
            (ulid(), now_utc(), source, decision, "consent-v3", self.keys.encrypt(receipt)),
        )
        self.db.commit()

    def add_publish(self, repo: str, commit_sha: str, manifest_sha: str, count: int) -> None:
        from .models import ulid
        self.db.execute(
            "INSERT INTO publish_log VALUES (?,?,?,?,?,?)",
            (ulid(), now_utc(), repo, commit_sha, manifest_sha, count),
        )
        self.db.commit()

    def stats(self) -> dict:
        def count(table: str, where: str = "") -> int:
            sql = f"SELECT COUNT(*) AS n FROM {table}" + (f" WHERE {where}" if where else "")
            return self.db.execute(sql).fetchone()["n"]
        return {
            "captures_total": count("captures"),
            "captures_pending": count("captures", "consent='pending'"),
            "captures_approved": count("captures", "consent IN ('approved','redacted')"),
            "captures_quarantined": count("captures", "quarantine_reason IS NOT NULL"),
            "records_clean": count("records", "source_tier='clean'"),
            "records_restricted": count("records", "source_tier='restricted'"),
            "publishes": count("publish_log"),
        }

    def purge(self) -> None:
        """Emergency wipe of all captured data and logs (keys remain until re-init)."""
        for table in ("captures", "records", "consent_events", "audit", "publish_log", "history_imports",
                      "disabled_chats"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()
