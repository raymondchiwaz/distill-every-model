import json
from pathlib import Path
from unittest.mock import patch

from dem.adapters.history import import_history
from dem.pipeline.formatter import build_records
from dem.pipeline.release import build_release
from dem.publisher import publish
from tests.test_all import TempEnv


class TestHistory(TempEnv):
    def setUp(self):
        super().setUp()
        self.transcript = self.root / "chat.jsonl"
        self.config.set("capture", "commercial_surfaces_default", "attest")
        self.vault.add_consent_event("onboard", "approved", "{}")
        self.vault.add_consent_event("commercial", "attested", "{}")

    def append(self, *rows):
        with self.transcript.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def run_import(self, source):
        return import_history(self.config, self.vault, source, self.transcript)

    def test_claude_pair_review_reimport_and_append(self):
        def row(role, content, **flags):
            return {"type": role, "sessionId": "session", "message": {"role": role, "content": content}, **flags}

        self.append(
            row("user", "Explain sorting to alice@example.org"),
            row("assistant", [{"type": "thinking", "thinking": "internal"}]),
            row("assistant", [{"type": "text", "text": "Working"}, {"type": "tool_use", "id": "x"}]),
            row("user", [{"type": "tool_result", "content": "shell output"}]),
            row("assistant", [{"type": "text", "text": "Sorting puts the items of a sequence into a chosen order."}]),
        )
        self.assertEqual(self.run_import("claude-code")["imported"], 1)
        prompt, answer = self.vault.list_captures()
        self.assertEqual(build_records(self.vault, self.config), [])
        self.assertNotIn("alice", prompt.redacted_text)
        self.assertNotIn("Explain", json.dumps(answer.metadata))
        self.vault.update_consent(answer.id, "approved")
        self.assertEqual(build_records(self.vault, self.config), [])
        self.vault.edit_capture(prompt.id, "Explain sorting?")
        self.append(row("user", "Another question?"), row("assistant", "Another answer"))
        stats = self.run_import("claude-code")
        self.assertEqual((stats["imported"], stats["already_imported"]), (1, 1))
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.prompt, "Explain sorting?")
        self.assertEqual(record.source_tier, "restricted")
        result = build_release([record], self.dirs["releases"], self.config)
        self.assertEqual(result["counts"]["restricted"], 1)
        with patch.dict("os.environ", {"HF_TOKEN": "fixture"}):
            with self.assertRaisesRegex(SystemExit, "restricted"):
                publish(Path(result["release_dir"]), self.config, self.vault, self.dirs["keys"], result["version"])

    def test_codex_user_facing_events_and_mirrored_wire_messages(self):
        def event(kind, **fields):
            return {"type": "event_msg", "payload": {"type": kind, **fields}}

        def wire(role, text, **fields):
            kind = "input_text" if role == "user" else "output_text"
            return {"type": "response_item", "payload": {"type": "message", "role": role,
                    "content": [{"type": kind, "text": text}], **fields}}

        self.append(
            {"type": "session_meta", "payload": {"id": "abc"}},
            {"type": "turn_context", "payload": {"model": "example-model"}},
            event("item_completed", item={"type": "UserMessage", "content": [{"type": "text", "text": "My question?"}]}),
            wire("user", "injected workspace context"), wire("assistant", "working", phase="commentary"),
            wire("assistant", "First final answer", phase="final_answer"),
            event("agent_message", message="First final answer"),
            event("user_message", message="Second question?"),
            event("item_completed", item={"type": "AgentMessage", "phase": "final_answer",
                  "content": [{"type": "Text", "text": "Second final answer"}]}),
        )
        self.assertEqual(self.run_import("codex")["imported"], 2)
        captures = self.vault.list_captures()
        for capture in captures:
            self.vault.update_consent(capture.id, "approved")
        records = build_records(self.vault, self.config)
        self.assertEqual([(r.prompt, r.response) for r in records],
                         [("My question?", "First final answer"), ("Second question?", "Second final answer")])
        self.assertEqual(records[0].metadata["conversation_id"], records[1].metadata["conversation_id"])
        self.assertEqual(records[0].metadata["model"], "example-model")

    def test_legacy_codex_and_quarantine(self):
        self.transcript.write_text("partial line\n")
        for role, kind, text in [("user", "input_text", "NDA confidential record"),
                                 ("assistant", "output_text", "An answer")]:
            self.append({"type": "response_item", "payload": {"type": "message", "role": role,
                        "content": [{"type": kind, "text": text}]}})
        stats = self.run_import("codex")
        self.assertEqual((stats["imported"], stats["malformed_lines"]), (1, 1))
        for event in self.vault.list_captures():
            self.vault.update_consent(event.id, "approved")
        self.assertEqual(build_records(self.vault, self.config), [])

    def test_watch_checks_consent_and_reads_appended_turns(self):
        seen = {}
        def scan():
            return import_history(self.config, self.vault, "codex", self.transcript, seen)
        def append_turn(text):
            self.append({"type": "event_msg", "payload": {"type": "user_message", "message": text}},
                        {"type": "event_msg", "payload": {"type": "agent_message", "message": "Answer"}})
        append_turn("First?")
        self.assertEqual(scan()["imported"], 1)
        with patch.object(Path, "open", side_effect=AssertionError("unchanged transcript opened")):
            self.assertEqual(scan()["imported"], 0)
        append_turn("Second?")
        self.assertEqual(scan()["imported"], 1)
        self.vault.add_consent_event("onboard", "rejected", "{}")
        with self.assertRaisesRegex(ValueError, "onboard"):
            scan()

    def test_chat_switch_survives_reopen_and_controls_import_and_release(self):
        def append_turn(prompt):
            self.append({"type": "user", "message": {"content": prompt}},
                        {"type": "assistant", "message": {"content": "An example answer."}})

        seen = {}
        append_turn("First question?")
        import_history(self.config, self.vault, "claude-code", self.transcript, seen)
        chat, = self.vault.history_chats()
        self.assertEqual(chat["title"], "First question?")
        self.assertTrue(chat["enabled"])
        chat_id = chat["conversation_id"]
        for capture in self.vault.history_messages(chat_id):
            self.vault.update_consent(capture.id, "approved")
        self.assertEqual(len(build_records(self.vault, self.config)), 1)

        self.assertTrue(self.vault.set_history_chat_enabled(chat_id, False))
        self.vault.close()
        self.vault = type(self.vault)(self.dirs["vault"], self.keys)
        self.assertFalse(self.vault.history_chats()[0]["enabled"])
        self.assertEqual(build_records(self.vault, self.config), [])
        append_turn("New question while the chat is off?")
        stats = import_history(self.config, self.vault, "claude-code", self.transcript, seen)
        self.assertEqual(stats["imported"], 0)
        self.assertEqual(len(self.vault.history_messages(chat_id)), 2)

        self.assertTrue(self.vault.set_history_chat_enabled(chat_id, True))
        stats = import_history(self.config, self.vault, "claude-code", self.transcript, seen)
        self.assertEqual(stats["imported"], 1)
        self.assertEqual(len(self.vault.history_messages(chat_id)), 4)
        self.assertEqual(len(build_records(self.vault, self.config)), 1)

    def test_consent_checked_before_reading(self):
        self.config.set("capture", "capture_prompts", False)
        with self.assertRaisesRegex(ValueError, "capture_prompts"):
            self.run_import("codex")
        self.config.set("capture", "capture_prompts", True)
        self.vault.add_consent_event("commercial", "rejected", "{}")
        with self.assertRaisesRegex(ValueError, "attest"):
            self.run_import("claude-code")
        self.vault.add_consent_event("onboard", "rejected", "{}")
        with self.assertRaisesRegex(ValueError, "onboard"):
            self.run_import("codex")
