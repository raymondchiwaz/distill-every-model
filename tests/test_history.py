import json
from pathlib import Path
from unittest.mock import patch

from dem.adapters.history import _prepare_activity, import_history
from dem.pipeline.formatter import build_records
from dem.pipeline.evidence import collect_task_evidence
from dem.pipeline.release import build_release
from dem.publisher import PublishError, publish
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
            row("assistant", [{"type": "text", "text": "Working"},
                              {"type": "tool_use", "id": "x", "name": "Read", "input": {"path": "/private"}}]),
            row("user", [{"type": "tool_result", "tool_use_id": "x", "content": "File contents", "is_error": False}]),
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
        self.assertEqual(record.activity, [{"type": "progress", "text": "Working"},
                                           {"type": "tool_call", "text": "Read", "input": '{\n  "path": "/private"\n}',
                                            "output": "File contents", "is_error": "false"}])
        result = build_release([record], self.dirs["releases"], self.config)
        self.assertEqual(result["counts"]["restricted"], 1)
        with patch.dict("os.environ", {"HF_TOKEN": "fixture"}):
            with self.assertRaisesRegex(PublishError, "Restricted"):
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

    def test_codex_attributes_tools_to_the_model_at_each_call(self):
        def row(kind, **payload):
            return {"type": "response_item", "payload": {"type": kind, **payload}}

        self.append(
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Check the model switch?"}},
            {"type": "turn_context", "payload": {"model": "model-first"}},
            row("custom_tool_call", name="read", call_id="one", input="Read the example."),
            {"type": "turn_context", "payload": {"model": "model-second"}},
            row("custom_tool_call", name="check", call_id="two", input="Check the example."),
            row("custom_tool_call_output", call_id="one", output="Read complete."),
            row("custom_tool_call_output", call_id="two", output="Check complete."),
            row("message", role="assistant", phase="final_answer",
                content=[{"type": "output_text", "text": "Both tools returned a result."}]),
        )
        self.assertEqual(self.run_import("codex")["imported"], 1)
        chat_id = self.vault.history_chats()[0]["conversation_id"]
        events = self.vault.history_messages(chat_id)
        self.assertEqual(events[1].model_hint, "model-second")
        self.assertEqual([step["model"] for step in events[1].activity], ["model-first", "model-second"])
        self.assertTrue(self.vault.approve_history_chat(chat_id, self.vault.history_review_snapshot(events)[0]))
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.metadata["model"], "model-second")
        self.assertEqual(record.activity[0]["model"], "model-first")

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

    def test_reimport_adds_redacted_activity_and_requires_review(self):
        def rollout(kind, **payload):
            return {"type": "response_item", "payload": {"type": kind, **payload}}

        self.append(
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Explain sorting to me?"}},
            rollout("reasoning", summary=[{"type": "summary_text", "text": "Check the examples from alice@example.org."}],
                    content=[{"type": "text", "text": "raw private reasoning"}], encrypted_content="encoded tail"),
            rollout("function_call", call_id="read-1", name="read_file", arguments='{"path": "example.py"}'),
            rollout("function_call_output", call_id="read-1", output="The example prints a greeting."),
            rollout("message", role="assistant", phase="final_answer",
                    content=[{"type": "output_text", "text": "Sorting arranges the items of a sequence in a chosen order."}]),
        )
        with patch("dem.adapters.history._prepare_activity", return_value=[]):
            self.assertEqual(self.run_import("codex")["imported"], 1)
        chat, = self.vault.history_chats()
        chat_id = chat["conversation_id"]
        for event in self.vault.history_messages(chat_id):
            self.vault.update_consent(event.id, "approved")
        self.assertEqual(len(build_records(self.vault, self.config)), 1)

        self.assertEqual(self.run_import("codex")["activity_updated"], 1)
        self.assertEqual(build_records(self.vault, self.config), [])
        events = self.vault.history_messages(chat_id)
        self.assertEqual([event.consent for event in events], ["approved", "pending"])
        self.assertEqual(events[1].activity, [{"type": "rationale", "text": "Check the examples from [EMAIL_1]."},
                                              {"type": "tool_call", "text": "read_file",
                                               "input": '{\n  "path": "example.py"\n}',
                                               "output": "The example prints a greeting."}])
        trace = self.vault.db.execute("SELECT data_enc FROM history_activity").fetchone()[0]
        self.assertNotIn(b"read_file", trace)
        snapshot, _ = self.vault.history_review_snapshot(events)
        self.assertFalse(self.vault.approve_history_chat(chat_id, "stale"))
        self.assertTrue(self.vault.approve_history_chat(chat_id, snapshot))
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.redaction_summary, [{"field": "activity", "category": "EMAIL", "count": 1}])
        self.assertEqual(self.run_import("codex")["activity_updated"], 0)

    def test_codex_tool_details_match_call_ids_and_keep_held_output_out_of_records(self):
        def rollout(kind, **payload):
            return {"type": "response_item", "payload": {"type": kind, **payload}}

        self.append(
            {"type": "event_msg", "payload": {"type": "user_message", "message": "Test the tool calls?"}},
            rollout("custom_tool_call", name="js", call_id="first", input="console.log('first');"),
            rollout("custom_tool_call", name="js", call_id="second", input="console.log('second');"),
            rollout("custom_tool_call_output", call_id="second", output='{"authorization":"Bearer abc"}'),
            rollout("custom_tool_call_output", call_id="first", output=[{"type": "input_text", "text": "First result"}]),
            rollout("message", role="assistant", phase="final_answer",
                    content=[{"type": "output_text", "text": "The two calls printed different values in order."}]),
        )
        with patch("dem.adapters.history._prepare_activity",
                   side_effect=lambda activity, config, **kwargs: _prepare_activity(activity, config)):
            self.assertEqual(self.run_import("codex")["imported"], 1)
        chat, = self.vault.history_chats()
        self.assertTrue(self.vault.approve_history_chat(chat["conversation_id"],
                                                      self.vault.history_review_snapshot(
                                                          self.vault.history_messages(chat["conversation_id"]))[0]))
        self.assertEqual(self.run_import("codex")["activity_updated"], 1)
        events = self.vault.history_messages(chat["conversation_id"])
        first, second = events[1].activity
        self.assertEqual((first["input"], first["output"]), ("console.log('first');", "First result"))
        self.assertEqual(second["input"], "console.log('second');")
        self.assertIn("[SECRET_1]", second["output"])
        self.assertNotIn("Bearer abc", second["output"])
        self.assertIn("output_hold", second)
        self.assertEqual(events[1].consent, "pending")
        self.assertTrue(self.vault.approve_history_chat(chat["conversation_id"],
                                                      self.vault.history_review_snapshot(events)[0]))
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.activity[0]["output"], "First result")
        self.assertNotIn("output", record.activity[1])
        self.assertEqual(self.run_import("codex")["activity_updated"], 0)

    def test_reimport_restores_code_in_saved_activity(self):
        self.config.set("capture", "capture_code", False)
        progress = "I’ll check this example.\n```python\nprint('hi')\n```"
        self.append(
            {"type": "user", "message": {"content": "Check the example?"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": progress},
                                                           {"type": "tool_use", "name": "read"}]}},
            {"type": "assistant", "message": {"content": "The program prints a short greeting."}},
        )
        self.assertEqual(self.run_import("claude-code")["imported"], 1)
        chat, = self.vault.history_chats()
        _, answer = self.vault.history_messages(chat["conversation_id"])
        self.assertIn("[CODE_BLOCK_REMOVED]", answer.activity[0]["text"])
        self.vault.update_consent(answer.id, "approved")
        self.config.set("capture", "capture_code", True)
        self.assertEqual(self.run_import("claude-code")["activity_updated"], 1)
        _, restored = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(restored.activity[0]["text"], progress)
        self.assertEqual(restored.consent, "pending")
        self.assertEqual(self.run_import("claude-code")["activity_updated"], 0)

    def test_claude_backfills_tool_results_and_exports_contributor_outcome_with_evidence(self):
        def row(role, content):
            return {"type": role, "sessionId": "review", "message": {"content": content}}

        self.append(
            row("user", "Can you test the sorting example?"),
            row("assistant", [{"type": "tool_use", "id": "write-1", "name": "Write",
                               "input": {"file_path": "/workspace/test_sort.py", "content": "assert sorted([2, 1]) == [1, 2]"}},
                              {"type": "tool_use", "id": "bash-1", "name": "Bash", "input": {"command": "pytest -q"}}]),
            row("user", [{"type": "tool_result", "tool_use_id": "bash-1", "is_error": True, "content": "failed"},
                         {"type": "tool_result", "tool_use_id": "write-1", "is_error": False, "content": "File written"}]),
            row("assistant", "The sorting example was written, and the test run reported a failure."),
        )
        with patch("dem.adapters.history._prepare_activity",
                   side_effect=lambda activity, config, **kwargs: _prepare_activity(activity, config)):
            self.assertEqual(self.run_import("claude-code")["imported"], 1)
        chat_id = self.vault.history_chats()[0]["conversation_id"]
        original = self.vault.history_messages(chat_id)
        self.assertEqual(self.run_import("claude-code")["activity_updated"], 1)
        events = self.vault.history_messages(chat_id)
        self.assertEqual([event.id for event in original], [event.id for event in events])
        self.assertEqual([step["output"] for step in events[1].activity], ["File written", "failed"])
        snapshot, _ = self.vault.history_review_snapshot(events)
        target = events[1].id
        self.assertFalse(self.vault.approve_history_chat(chat_id, snapshot, outcome="solved", outcome_target="stale"))
        self.assertEqual(build_records(self.vault, self.config), [])
        self.assertTrue(self.vault.approve_history_chat(chat_id, snapshot, outcome="partial", outcome_target=target))
        record, = build_records(self.vault, self.config)
        self.assertEqual(record.task_outcome, {"status": "partial", "source": "contributor"})
        self.assertEqual(record.evidence, {"files_changed": ["test_sort.py"], "tool_errors": 1,
                                          "checks": [{"name": "pytest", "status": "failed", "source": "recorded_tool"}]})
        self.assertTrue(self.vault.approve_history_chat(chat_id, "", outcome="unspecified", outcome_target=target))
        unchanged, = build_records(self.vault, self.config)
        self.assertIsNone(unchanged.task_outcome)
        self.assertEqual(unchanged.evidence, record.evidence)

    def test_codex_evidence_uses_execution_header_and_simple_checks(self):
        output = "Chunk ID: abc\nProcess exited with code 0\nOutput:\nProcess exited with code 1"
        steps = [
            {"type": "tool_call", "text": "exec_command", "input": '{"cmd":"python -m unittest -q"}', "output": output},
            {"type": "tool_call", "text": "exec_command", "input": '{"cmd":"pytest; echo done"}', "output": output},
            {"type": "tool_call", "text": "exec_command", "input": '{"cmd":"npm test"}',
             "output": "Process exited with code 0"},
            {"type": "tool_call", "text": "apply_patch", "input": "*** Add File: tests/test_new.py\n+pass\n",
             "output": "Success. Updated the following files:\nA tests/test_new.py"},
            {"type": "tool_call", "text": "js", "input": 'const r = await tools.exec_command({cmd: "ruff check ."}); text(r);',
             "output": 'Result\n{"chunk_id":"abc","exit_code":1,"wall_time_seconds":0.1,"output":"lint failed"}'},
        ]
        self.assertEqual(collect_task_evidence(steps), {"files_changed": ["tests/test_new.py"], "tool_errors": 1,
                                                       "checks": [{"name": "unittest", "status": "passed",
                                                                   "source": "recorded_tool"},
                                                                  {"name": "ruff check", "status": "failed",
                                                                   "source": "recorded_tool"}]})

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
