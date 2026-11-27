import json
import sqlite3
import zipfile

from dem.adapters.history import import_history
from dem.history_sources import SOURCES
from dem.pipeline.formatter import build_records
from tests.test_all import TempEnv

PROMPT = "Can you explain the sorting order?"
ANSWER = "The items are sorted in ascending order according to the selected sort key."


def messages():
    return [{"role": "user", "content": PROMPT},
            {"role": "assistant", "content": [{"type": "thinking", "text": "hidden"},
                                              {"type": "text", "text": ANSWER}]}]


class TestHarnessImports(TempEnv):
    def setUp(self):
        super().setUp()
        self.config.set("capture", "commercial_surfaces_default", "attest")
        self.vault.add_consent_event("onboard", "approved", "{}")
        self.vault.add_consent_event("commercial", "attested", "{}")

    def test_harness_formats_round_trip_into_review_without_reasoning(self):
        extension = [{"type": "say", "say": "task", "text": PROMPT},
                     {"type": "say", "say": "reasoning", "text": "hidden"},
                     {"type": "say", "say": "completion_result", "text": ANSWER}]
        wire = [{"type": "message", "message": msg} for msg in messages()]
        gemini = [{"type": "user", "content": [{"text": PROMPT}]},
                  {"type": "gemini", "content": [{"thought": True, "text": "hidden"}, {"text": ANSWER}]}]
        cases = {
            "aider": ("md", f"# aider chat started at noon\n\n#### {PROMPT}\n> tool output\n\n{ANSWER}\n"),
            "amp": ("json", {"id": "thread", "messages": messages()}),
            "antigravity": ("jsonl", [
                {"type": "USER_INPUT", "content": f"<USER_REQUEST>{PROMPT}</USER_REQUEST><ADDITIONAL_METADATA>hidden"},
                {"type": "PLANNER_RESPONSE", "content": "interim answer", "tool_calls": ["tool"]},
                {"type": "PLANNER_RESPONSE", "content": ANSWER, "thinking": "hidden"},
            ]),
            "cline": ("json", extension),
            "continue": ("json", {"sessionId": "one", "history": wire}),
            "copilot": ("jsonl", [{"type": "user.message", "data": {"content": PROMPT}},
                                  {"type": "assistant.message", "data": {"content": ANSWER}}]),
            "cursor": ("jsonl", [{"role": msg["role"], "message": msg} for msg in messages()]),
            "droid": ("jsonl", [{"type": "session_start", "id": "one"}, *wire]),
            "gemini-cli": ("json", {"sessionId": "one", "messages": gemini}),
            "kilo-code": ("json", extension),
            "kimi-cli": ("jsonl", messages()),
            "kiro": ("jsonl", [{"kind": "Prompt", "data": {"content": PROMPT}},
                               {"kind": "AssistantMessage", "data": {"content": ANSWER}}]),
            "mistral-vibe": ("jsonl", messages()),
            "opencode": ("json", {"info": {"id": "one"}, "messages": [
                {"info": {"role": msg["role"]}, "parts": [{"type": "text", "text": text}]}
                for msg, text in zip(messages(), [PROMPT, ANSWER])
            ]}),
            "openhands": ("json", [
                {"action": "message", "args": {"content": PROMPT}, "source": "user"},
                {"action": "finish", "args": {"final_thought": ANSWER}, "source": "agent"},
            ]),
            "pi": ("jsonl", [{"type": "session", "id": "one"},
                            {**wire[0], "id": "a", "parentId": None},
                            {"type": "message", "message": {"role": "user", "content": "abandoned"},
                             "id": "b", "parentId": "a"},
                            {**wire[1], "id": "c", "parentId": "a"}]),
            "qwen-code": ("jsonl", [
                {"type": "user", "sessionId": "one", "message": {"parts": [{"text": PROMPT}]}},
                {"type": "assistant", "message": {"parts": [{"functionCall": {"name": "tool"}}]}},
                {"type": "assistant", "message": {"parts": [{"text": ANSWER}]}},
            ]),
            "roo-code": ("json", extension),
        }
        self.assertEqual(set(cases) | {"claude-code", "codex", "goose", "crush"}, set(SOURCES))
        for source, (extension, data) in cases.items():
            with self.subTest(source=source):
                path = self.root / f"{source}.{extension}"
                if extension == "jsonl":
                    data = "\n".join(map(json.dumps, data))
                elif extension == "json":
                    data = json.dumps(data)
                path.write_text(data, encoding="utf-8")
                result = import_history(self.config, self.vault, source, path)
                self.assertEqual(result["imported"], 1, result)
                chat = next(chat for chat in self.vault.history_chats() if chat["source"] == source)
                captures = self.vault.history_messages(chat["conversation_id"])
                self.assertEqual([event.text for event in captures], [PROMPT, ANSWER])
                self.assertEqual(import_history(self.config, self.vault, source, path)["already_imported"], 1)

    def test_databases_and_zip_exports_are_read_into_chat_controls(self):
        path = self.root / "goose.db"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE messages(id INTEGER, session_id TEXT, role TEXT,"
                       " content_json TEXT, created_timestamp INTEGER, metadata_json TEXT)")
            for i, msg in enumerate(messages(), 1):
                db.execute("INSERT INTO messages VALUES (?, 'one', ?, ?, ?, '{}')",
                           (i, msg["role"], json.dumps(msg["content"]), i))
        self.assertEqual(import_history(self.config, self.vault, "goose", path)["imported"], 1)
        path = self.root / "crush.db"
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE sessions(id TEXT, parent_session_id TEXT);"
                             "INSERT INTO sessions VALUES ('one', NULL);"
                             "CREATE TABLE messages(session_id TEXT, role TEXT, parts TEXT,"
                             " model TEXT, created_at INTEGER);")
            for i, (role, text) in enumerate(zip(("user", "assistant"), (PROMPT, ANSWER)), 1):
                db.execute("INSERT INTO messages VALUES ('one', ?, ?, 'model', ?)",
                           (role, json.dumps([{"type": "text", "data": {"text": text}}]), i))
        self.assertEqual(import_history(self.config, self.vault, "crush", path)["imported"], 1)

        path = self.root / "opencode.db"
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE session_v2(id TEXT, parent_id TEXT);"
                             "INSERT INTO session_v2 VALUES ('one', NULL);"
                             "CREATE TABLE session_message(id TEXT, session_id TEXT, seq INTEGER, data TEXT, type TEXT);"
                             "CREATE TABLE session_part(session_id TEXT, message_id TEXT, ordinal INTEGER, data TEXT, type TEXT);")
            db.execute("INSERT INTO session_message VALUES ('a', 'one', 1, ?, 'user')", (json.dumps({"text": PROMPT}),))
            db.execute("INSERT INTO session_message VALUES ('b', 'one', 2, '{}', 'assistant')")
            db.execute("INSERT INTO session_part VALUES ('one', 'b', 1, ?, 'text')",
                       (json.dumps({"type": "text", "text": ANSWER}),))
        self.assertEqual(import_history(self.config, self.vault, "opencode", path)["imported"], 1)

        root = self.config.data_dir / "history" / "kiro"
        root.mkdir(parents=True)
        with zipfile.ZipFile(root / "export.zip", "w") as archive:
            archive.writestr("session-1/messages.jsonl", "\n".join(map(json.dumps, messages())))
            archive.writestr("session-1/sub-executions/child/messages.jsonl", "\n".join(map(json.dumps, messages())))
        self.assertEqual(import_history(self.config, self.vault, "kiro")["imported"], 1)
        chat = next(chat for chat in self.vault.history_chats() if chat["source"] == "kiro")
        for message in self.vault.history_messages(chat["conversation_id"]):
            self.vault.update_consent(message.id, "approved")
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.source_tier, "restricted")
        self.vault.set_history_chat_enabled(chat["conversation_id"], False)
        self.assertEqual(build_records(self.vault, self.config), [])
