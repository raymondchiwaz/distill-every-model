import json
import sqlite3
import zipfile
from unittest.mock import patch

from zstandard import ZstdCompressor

from dem.adapters.history import import_history
from dem.history_sources import SOURCES, history_files
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

    def test_deepseek_native_generations_and_official_zip(self):
        def event(kind, **data):
            return {"type": kind, "data": data}

        def text(value):
            return {"type": "text", "text": value}

        def assistant(*content, model="deepseek-v4-flash", **extra):
            return event("assistant/message", message={"source": {"kind": "model", "model": model},
                                                      "content": content}, **extra)

        def log(*rows):
            return b"".join(json.dumps(row).encode() + b"\n" for row in rows)

        header = {"type": "session", "id": "outer", "version": 4}
        start = event("turn/start", turn=1)
        end = event("turn/end", turn=1, reason={"kind": "completed"})
        user = event("user/message", source={"kind": "user"}, content=[text(PROMPT)])
        args = '{"path":"sort.py"}'
        call = {"type": "tool-call", "id": "call-1", "name": "read_example", "arguments": args}
        dispatch = event("tool/call", callId="call-1", name="read_example", arguments=args)
        result = event("tool/result", message={"role": "tool", "toolCallId": "call-1",
                                              "content": [text("File contents")]})
        body = log(start, user, event("user/message", source={"kind": "plugin"},
                                     content=[text("Injected instructions")]),
                   assistant({"type": "reasoning", "text": "Hidden thought"}, call),
                   dispatch, result, assistant(text(ANSWER), model="deepseek-v4-pro"),
                   {**assistant(text("Compacted reply")), "surfaceOp": {"op": "replace"}}, end)
        folder = self.root / "sessions" / "--project--" / "outer"
        folder.mkdir(parents=True)
        current = folder / "session.v4.jsonl.zstd"
        compressor = ZstdCompressor()
        current.write_bytes(compressor.compress(log(header)) + compressor.compress(body))
        legacy_result = event("tool/result", message={"role": "user",
                              "source": {"kind": "tool", "callId": "call-1"}, "content": [
                                  {"type": "tool-result", "toolCallId": "call-1",
                                   "content": [text("File contents")], "isError": False}]})
        older = folder / "session.v3.jsonl"
        older.write_bytes(log({**header, "version": 3}, start, user, assistant(call),
                              dispatch, legacy_result, assistant(text(ANSWER), model="deepseek-v4-pro"), end))
        with patch.dict("os.environ", {"DSH_HOME": str(self.root)}):
            self.assertEqual(history_files("deepseek"), [current])
            self.assertEqual(import_history(self.config, self.vault, "deepseek")["imported"], 1)
        self.assertEqual(import_history(self.config, self.vault, "deepseek", older)["already_imported"], 1)
        outer, = self.vault.history_chats()
        prompt, answer = self.vault.history_messages(outer["conversation_id"])
        self.assertEqual((prompt.text, answer.text, answer.model_hint), (PROMPT, ANSWER, "deepseek-v4-pro"))
        step, = [step for step in answer.activity if step["type"] == "tool_call"]
        self.assertEqual((step["text"], step["output"], step["model"]), ("read_example", "File contents",
                                                                      "deepseek-v4-flash"))
        self.assertIn("sort.py", step["input"])
        self.assertNotIn("Hidden thought", str(answer.activity))

        exported = self.root / "deepseek.zip"
        with zipfile.ZipFile(exported, "w") as archive:
            archive.writestr("session.v4.jsonl", log(header) + body)
            archive.writestr("subagents/child/session.v4.jsonl", log(
                {**header, "id": "isolated", "origin": "subagent", "isSeeded": True},
                start, user, dispatch, end, event("session/end-seed", inherited=True),
                start, dispatch, result, assistant(text("Interrupted reply"), interrupted=True), end))
            archive.writestr("files/attachment/session.v4.jsonl", "ignored")
        stats = import_history(self.config, self.vault, "deepseek", exported)
        self.assertEqual((stats["imported"], stats["already_imported"], stats["activity_only"]), (0, 1, 1))
        child = next(chat for chat in self.vault.history_chats() if chat["activity_origin"] == "subagent")
        activity, = self.vault.history_messages(child["conversation_id"])
        self.assertEqual((activity.role, activity.redacted_text), ("activity", ""))
        self.assertEqual([step["output"] for step in activity.activity], ["File contents"])

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
        self.assertEqual(set(cases) | {"claude-code", "codex", "crush", "deepseek", "goose"}, set(SOURCES))
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
            db.execute("INSERT INTO session_message VALUES ('b', 'one', 2, ?, 'assistant')",
                       (json.dumps({"model": {"id": "tool-model"}}),))
            db.execute("INSERT INTO session_part VALUES ('one', 'b', 1, ?, 'text')",
                       (json.dumps({"type": "text", "text": "I’ll check the sort key."}),))
            db.execute("INSERT INTO session_part VALUES ('one', 'b', 2, ?, 'tool')",
                       (json.dumps({"type": "tool", "name": "read",
                                    "state": {"input": {"path": "sort.py"}, "status": "completed",
                                              "content": [{"type": "text", "text": "Saved result"}]}}),))
            db.execute("INSERT INTO session_message VALUES ('c', 'one', 3, ?, 'assistant')",
                       (json.dumps({"model": {"id": "answer-model"}}),))
            db.execute("INSERT INTO session_part VALUES ('one', 'c', 1, ?, 'text')",
                       (json.dumps({"type": "text", "text": ANSWER}),))
        with patch("dem.adapters.history._prepare_activity", return_value=[]):
            self.assertEqual(import_history(self.config, self.vault, "opencode", path)["imported"], 1)
        chat = next(chat for chat in self.vault.history_chats() if chat["source"] == "opencode")
        ids = {chat["conversation_id"]}
        initial = self.vault.history_messages(chat["conversation_id"])
        result = import_history(self.config, self.vault, "opencode", path, conversation_ids=ids)
        self.assertEqual((result["imported"], result["activity_updated"]), (0, 1))
        prompt, answer = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual([event.id for event in initial], [prompt.id, answer.id])
        self.assertEqual(answer.model_hint, "answer-model")
        self.assertEqual(answer.activity, [
            {"type": "progress", "text": "I’ll check the sort key.", "model": "tool-model"},
            {"type": "tool_call", "text": "read", "model": "tool-model", "input": '{\n  "path": "sort.py"\n}',
             "output": "Saved result", "is_error": "false"},
        ])

        self.vault.set_redaction(answer.id, answer.redacted_text, [], "held for review")
        with self.vault.db:
            self.vault.db.execute("DELETE FROM history_activity WHERE capture_id=?", (answer.id,))
        self.vault.update_consent(prompt.id, "approved")
        self.assertEqual(import_history(self.config, self.vault, "opencode", path, conversation_ids=ids)["activity_updated"], 1)
        self.assertEqual(self.vault.history_messages(chat["conversation_id"])[1].activity, answer.activity)
        self.assertEqual(build_records(self.vault, self.config), [])

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

    def test_opencode_inline_tools_and_model_switch_rows(self):
        path = self.root / "opencode.db"
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE session_v2(id TEXT, parent_id TEXT);"
                             "INSERT INTO session_v2 VALUES ('inline', NULL);"
                             "CREATE TABLE session_message(id TEXT, session_id TEXT, seq INTEGER, data TEXT, type TEXT);"
                             "CREATE TABLE session_part(session_id TEXT, message_id TEXT, ordinal INTEGER, data TEXT, type TEXT);")
            rows = [
                ("user", {"text": PROMPT}),
                ("assistant", {"model": {"id": "model-first"}, "finish": "tool-calls", "content": [
                    {"type": "tool", "name": "read", "state": {"input": {"path": "example.py"},
                     "content": [{"type": "text", "text": "File contents"}], "status": "completed"}},
                ]}),
                ("model-switched", {"model": {"id": "model-second"}, "previous": {"id": "model-first"}}),
                ("assistant", {"model": {"id": "model-second"}, "content": [
                    {"type": "text", "text": "This reply mentions a confidential patient record."},
                ]}),
            ]
            for position, (kind, data) in enumerate(rows, 1):
                db.execute("INSERT INTO session_message VALUES (?, 'inline', ?, ?, ?)",
                           (str(position), position, json.dumps(data), kind))
        self.assertEqual(import_history(self.config, self.vault, "opencode", path)["imported"], 1)
        chat, = self.vault.history_chats()
        prompt, answer = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(answer.model_hint, "model-second")
        self.assertIsNotNone(answer.quarantine_reason)
        self.assertEqual([(step["type"], step["model"]) for step in answer.activity],
                         [("tool_call", "model-first"), ("model_change", "model-second")])
        self.assertEqual(answer.activity[0]["output"], "File contents")
        self.vault.update_consent(prompt.id, "approved")
        self.assertEqual(build_records(self.vault, self.config), [])

    def test_saved_tool_activity_round_trips_for_each_supported_harness(self):
        def wire(role, text):
            return {"role": role, "content": text}

        prompt = wire("user", PROMPT)
        reply = wire("assistant", ANSWER)
        request = {"id": "first", "type": "tool_use", "name": "read_example", "input": {"path": "sort.py"}}
        result = {"type": "tool_result", "tool_use_id": "first", "content": "File contents", "is_error": False}
        blocks = [prompt, wire("assistant", [request]), wire("user", [result]), reply]
        function = {"id": "first", "type": "function", "function": {
            "name": "read_example", "arguments": '{"path":"sort.py"}'}}
        openai = [prompt, {**wire("assistant", ""), "tool_calls": [function]},
                  {**wire("tool", "File contents"), "tool_call_id": "first"}, reply]
        gemini = [dict(type="user", content=PROMPT),
                  dict(type="gemini", content=ANSWER, toolCalls=[dict(
                      id="first", name="read_example", args={"path": "sort.py"}, status="success",
                      result=[{"type": "text", "text": "File contents"}])])]
        extension = [{"type": "say", "say": "task", "text": PROMPT},
                     {"type": "ask", "ask": "command", "text": "read_example sort.py"},
                     {"type": "say", "say": "command_output", "text": "File contents"},
                     {"type": "say", "say": "completion_result", "text": ANSWER}]
        codex = [
            {"type": "event_msg", "payload": {"type": "user_message", "message": PROMPT}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "read_example",
                                                 "call_id": "first", "arguments": '{"path":"sort.py"}'}},
            {"type": "response_item", "payload": {"type": "function_call_output",
                                                 "call_id": "first", "output": "File contents"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": ANSWER}},
        ]
        copilot = [
            {"type": "user.message", "data": {"content": PROMPT}},
            {"type": "assistant.message", "data": {"toolRequests": [
                {"toolCallId": "first", "name": "read_example", "arguments": {"path": "sort.py"}}]}},
            {"type": "tool.execution_start", "data": {
                "toolCallId": "first", "toolName": "read_example", "arguments": {"path": "sort.py"}}},
            {"type": "tool.execution_complete", "data": {
                "toolCallId": "first", "success": True, "result": {"content": "File contents"}}},
            {"type": "assistant.message", "data": {"content": ANSWER}},
        ]
        antigravity = [{"type": "USER_INPUT", "content": PROMPT},
                       {"type": "PLANNER_RESPONSE", **openai[1]},
                       {"type": "TOOL_RESULT", **openai[2]},
                       {"type": "PLANNER_RESPONSE", "content": ANSWER}]
        qwen = [{"type": "user", "message": {"parts": [{"text": PROMPT}]}},
                {"type": "assistant", "message": {"parts": [{"functionCall": {
                    "name": "read_example", "args": {"path": "sort.py"}}}]}},
                {"type": "user", "message": {"parts": [{"functionResponse": {
                    "name": "read_example", "response": {"body": "File contents"}}}]}},
                {"type": "assistant", "message": {"parts": [{"text": ANSWER}]}}]
        kiro = [{"kind": "Prompt", "data": {"content": PROMPT}},
                {"kind": "AssistantMessage", "data": {"content": [
                    {"kind": "toolUse", "data": {"name": "read_example", "toolUseId": "first",
                                                 "input": {"path": "sort.py"}}}]}},
                {"kind": "ToolResult", "data": {"toolUseId": "first", "content": "File contents"}},
                {"kind": "AssistantMessage", "data": {"content": [{"kind": "text", "data": {"text": ANSWER}}]}}]
        openhands = [{"kind": "MessageEvent", "source": "user", "llm_message": prompt},
                     {"kind": "ActionEvent", "tool_name": "read_example", "tool_call_id": "first",
                      "action": {"path": "sort.py"}},
                     {"kind": "ObservationEvent", "tool_name": "read_example", "tool_call_id": "first",
                      "observation": {"content": "File contents", "is_error": False}},
                     {"kind": "MessageEvent", "source": "agent", "llm_message": reply}]
        opencode = {"info": {"id": "sample-opencode"}, "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": PROMPT}]},
            {"info": {"role": "assistant"}, "parts": [
                {"type": "tool", "tool": "read_example", "callID": "first", "state": {
                    "status": "completed", "input": {"path": "sort.py"}, "output": "File contents"}}]},
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": ANSWER}]},
        ]}
        cases = {
            "aider": ("md", f"# aider chat started at noon\n\n#### {PROMPT}\n> File contents\n\n{ANSWER}\n"),
            "amp": ("json", {"id": "amp-example", "messages": blocks}),
            "antigravity": ("jsonl", antigravity),
            "claude-code": ("jsonl", [{"type": row["role"], "message": row} for row in blocks]),
            "cline": ("json", extension),
            "codex": ("jsonl", codex),
            "continue": ("json", {"history": [{"message": row} for row in openai]}),
            "copilot": ("jsonl", copilot),
            "cursor": ("jsonl", [{"role": row["role"], "message": row} for row in blocks]),
            "droid": ("jsonl", [{"type": "message", "message": row} for row in blocks]),
            "gemini-cli": ("json", {"messages": gemini}),
            "kilo-code": ("json", extension),
            "kimi-cli": ("jsonl", openai),
            "kiro": ("jsonl", kiro),
            "mistral-vibe": ("jsonl", openai),
            "opencode": ("json", opencode),
            "openhands": ("json", openhands),
            "pi": ("jsonl", [{"type": "message", "message": row} for row in [
                prompt, wire("assistant", [{"type": "toolCall", "id": "first", "name": "read_example",
                                           "arguments": {"path": "sort.py"}}]),
                {**wire("toolResult", "File contents"), "toolCallId": "first", "isError": False}, reply,
            ]]),
            "qwen-code": ("jsonl", qwen),
            "roo-code": ("json", extension),
        }
        goose = self.root / "goose-tools.db"
        with sqlite3.connect(goose) as db:
            db.execute("CREATE TABLE messages(id INTEGER, session_id TEXT, role TEXT,"
                       " content_json TEXT, created_timestamp INTEGER, metadata_json TEXT)")
            content = [
                ("user", PROMPT, True),
                ("assistant", [{"type": "toolRequest", "id": "first", "toolCall": {"status": "success",
                 "value": {"name": "read_example", "arguments": {"path": "sort.py"}}}}], True),
                ("user", [{"type": "toolResponse", "id": "first", "toolResult": {"status": "success",
                 "value": {"content": [{"type": "text", "text": "File contents"}], "isError": False}}}], False),
                ("assistant", ANSWER, True),
            ]
            for index, (role, parts, visible) in enumerate(content, 1):
                db.execute("INSERT INTO messages VALUES (?, 'one', ?, ?, ?, ?)",
                           (index, role, json.dumps(parts), index, json.dumps({"userVisible": visible})))
        crush = self.root / "crush-tools.db"
        with sqlite3.connect(crush) as db:
            db.executescript("CREATE TABLE sessions(id TEXT, parent_session_id TEXT);"
                             "INSERT INTO sessions VALUES ('one', NULL);"
                             "CREATE TABLE messages(session_id TEXT, role TEXT, parts TEXT,"
                             " model TEXT, created_at INTEGER);")
            content = [
                ("user", {"type": "text", "data": {"text": PROMPT}}),
                ("assistant", {"type": "tool_call", "data": {"tool_call_id": "first", "tool_name": "read_example",
                                                            "input": '{"path":"sort.py"}'}}),
                ("tool", {"type": "tool_result", "data": {"tool_call_id": "first", "name": "read_example",
                                                         "content": "File contents", "is_error": False}}),
                ("assistant", {"type": "text", "data": {"text": ANSWER}}),
            ]
            for index, (role, part) in enumerate(content, 1):
                db.execute("INSERT INTO messages VALUES ('one', ?, ?, 'model', ?)", (role, json.dumps([part]), index))
        cases.update({"goose": ("db", goose), "crush": ("db", crush)})
        self.assertEqual(set(cases) | {"deepseek"}, set(SOURCES))

        for source, (suffix, data) in cases.items():
            with self.subTest(source=source):
                if suffix == "db":
                    path = data
                else:
                    path = self.root / f"tool-fixture-{source}.{suffix}"
                    if suffix == "json":
                        data = json.dumps(data)
                    if suffix == "jsonl":
                        data = "\n".join(map(json.dumps, data))
                    path.write_text(data, encoding="utf-8")
                stats = import_history(self.config, self.vault, source, path)
                self.assertEqual(stats["imported"], 1, stats)
                chat = next(chat for chat in self.vault.history_chats() if chat["source"] == source)
                _, response = self.vault.history_messages(chat["conversation_id"])
                step, = [step for step in response.activity if step["type"] == "tool_call"]
                self.assertIn("File contents", step["output"])
                if source == "aider":
                    self.assertEqual(step["text"], "Aider tool output")
                    self.assertNotIn("input", step)
                else:
                    self.assertEqual(step["text"], "command" if source in ("cline", "kilo-code", "roo-code")
                                     else "read_example")
                    self.assertIn("sort.py", step["input"])
                self.assertEqual(import_history(self.config, self.vault, source, path)["activity_updated"], 0)

    def test_unfinished_calls_stay_local_and_are_promoted_when_a_reply_arrives(self):
        path = self.root / "unfinished.jsonl"
        call = {"type": "toolCall", "id": "running-1", "name": "read_example", "arguments": {"path": "sort.py"}}
        rows = [{"type": "message", "message": message} for message in [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": [call, {**call, "id": "running-2"}]},
            {"role": "toolResult", "toolCallId": "running-2", "toolName": "read_example",
             "content": "File contents", "isError": True},
        ]]
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        stats = import_history(self.config, self.vault, "pi", path)
        self.assertEqual((stats["imported"], stats["activity_only"]), (0, 1))
        chat, = self.vault.history_chats()
        pending, = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(pending.role, "activity")
        self.assertEqual(len(pending.activity), 2)
        self.assertNotIn("output", pending.activity[0])
        self.assertEqual(pending.activity[1]["is_error"], "true")
        self.vault.update_consent(pending.id, "approved")
        self.assertEqual(self.vault.get_capture(pending.id).consent, "local")
        self.assertEqual(import_history(self.config, self.vault, "pi", path)["activity_only"], 0)
        self.assertEqual(build_records(self.vault, self.config), [])

        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + json.dumps({"type": "message", "message": {"role": "assistant", "content": ANSWER}}))
        self.assertEqual(import_history(self.config, self.vault, "pi", path,
                                        conversation_ids={chat["conversation_id"]})["imported"], 1)
        prompt, answer = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(answer.activity, pending.activity)
        self.assertEqual(prompt.text, PROMPT)
        self.assertIsNone(self.vault.get_capture(pending.id))

    def test_child_history_records_tools_without_importing_child_chat_text(self):
        base = self.root / "subagents"
        base.mkdir()
        path = base / "agent-example.jsonl"
        rows = [
            ("user", "Internal delegated prompt"),
            ("assistant", [{"type": "tool_use", "name": "read_example", "id": "child-read",
                            "input": {"path": "sort.py"}}]),
            ("user", [{"type": "tool_result", "tool_use_id": "child-read", "content": "File contents"}]),
            ("assistant", "Internal delegated reply"),
        ]
        path.write_text("\n".join(json.dumps({"type": role, "isSidechain": True, "agentId": "child-agent",
                                             "sessionId": "parent", "message": {"content": content}})
                                  for role, content in rows), encoding="utf-8")
        stats = import_history(self.config, self.vault, "claude-code", base)
        self.assertEqual((stats["imported"], stats["activity_only"]), (0, 1), stats)
        chat, = self.vault.history_chats()
        event, = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(chat["activity_origin"], "subagent")
        self.assertEqual(event.redacted_text, "")
        self.assertEqual(event.activity[0]["output"], "File contents")
        self.assertEqual(build_records(self.vault, self.config), [])

    def test_codex_event_mirrors_and_late_results_keep_original_turn(self):
        path = self.root / "mirror.jsonl"

        def event(kind, **args):
            return {"type": "event_msg", "payload": {"type": kind, **args}}

        def wire(kind, **args):
            return {"type": "response_item", "payload": {"type": kind, **args}}

        rows = [
            event("user_message", message=PROMPT),
            event("exec_command_begin", call_id="shell-1", command=["echo", "hello"]),
            wire("function_call", call_id="shell-1", name="exec_command", arguments='{"cmd":"echo hello"}'),
            event("agent_message", message=ANSWER),
            event("user_message", message="A follow-up question"),
            event("exec_command_end", call_id="shell-1", exit_code=0, aggregated_output="hello"),
            event("agent_message", message="A follow-up response"),
        ]
        path.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        result = import_history(self.config, self.vault, "codex", path)
        self.assertEqual(result["imported"], 2, result)
        chat, = self.vault.history_chats()
        _, first, _, second = self.vault.history_messages(chat["conversation_id"])
        tool, = first.activity
        self.assertEqual((tool["text"], tool["output"], tool["exit_code"]), ("exec_command", "hello", "0"))
        self.assertEqual(second.activity, [])

    def test_partial_reimport_keeps_distinct_repeated_calls(self):
        path = self.root / "partial.jsonl"

        def write(*calls):
            rows = [{"role": "user", "content": PROMPT}]
            for label in calls:
                rows.extend([
                    {"role": "assistant", "content": [{"type": "toolCall", "id": label,
                      "name": "read_example", "arguments": {"path": f"{label}.py"}}]},
                    {"role": "toolResult", "toolCallId": label, "toolName": "read_example",
                     "content": f"Contents of {label}.py", "isError": False},
                ])
            rows.append({"role": "assistant", "content": ANSWER})
            path.write_text("\n".join(json.dumps({"type": "message", "message": row}) for row in rows),
                            encoding="utf-8")

        write("first", "second")
        self.assertEqual(import_history(self.config, self.vault, "pi", path)["imported"], 1)
        write("second")
        self.assertEqual(import_history(self.config, self.vault, "pi", path)["activity_updated"], 0)
        chat, = self.vault.history_chats()
        _, response = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual([step["output"] for step in response.activity], ["Contents of first.py", "Contents of second.py"])

    def test_antigravity_backfills_indexed_results_from_native_transcript(self):
        logs = self.root / "antigravity-example" / ".system_generated" / "logs"
        logs.mkdir(parents=True)
        transcript = logs / "transcript.jsonl"
        full = logs / "transcript_full.jsonl"

        def row(index, kind, status="DONE", **data):
            return {"type": kind, "source": "USER_EXPLICIT" if kind == "USER_INPUT" else "MODEL",
                    "step_index": index, "status": status, **data}

        calls = [{"name": "manage_task", "args": {"action": "list"}},
                 {"name": "read_file", "args": {"path": "shortened"}},
                 {"name": "manage_task", "args": {"action": "wait"}},
                 {"name": "manage_task", "args": {"action": "check"}}]
        prompt = row(1, "USER_INPUT", content=f"<USER_REQUEST>{PROMPT}</USER_REQUEST>")
        planner = row(2, "PLANNER_RESPONSE", tool_calls=calls, truncated_fields=["tool_calls"])
        final = row(7, "PLANNER_RESPONSE", content=ANSWER)
        shortened = row(4, "GENERIC", content="Shortened", truncated_fields=["content"])
        rows = [prompt, planner, row(5, "GENERIC", "RUNNING", content="Still running"),
                row(3, "GENERIC", content="Task listed"), shortened,
                row(5, "GENERIC", "INVALID", content="Wait failed", error="Test error"), final,
                row(6, "GENERIC", content="Unrelated later result")]

        def write(path, records):
            path.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")

        write(transcript, rows)
        with patch("dem.adapters.history._prepare_activity",
                   side_effect=lambda activity, config, **kwargs: [
                       {"type": kind, "text": text} for kind, text in activity if kind == "tool_call"
                   ]):
            self.assertEqual(import_history(self.config, self.vault, "antigravity", transcript)["imported"], 1)
        chat, = self.vault.history_chats()
        earlier = self.vault.history_messages(chat["conversation_id"])
        self.assertTrue(self.vault.approve_history_chat(chat["conversation_id"],
                                                       self.vault.history_review_snapshot(earlier)[0]))

        full_calls = calls.copy()
        full_calls[1] = {"name": "read_file", "args": {"path": "example.py"}}
        write(full, [prompt, row(2, "PLANNER_RESPONSE", tool_calls=full_calls), *rows[2:4],
                     row(4, "GENERIC", content="Full file result"), *rows[5:]])
        stats = import_history(self.config, self.vault, "antigravity", logs)
        self.assertEqual((stats["files"], stats["imported"], stats["activity_updated"]), (1, 0, 1))
        _, answer = self.vault.history_messages(chat["conversation_id"])
        self.assertEqual(answer.id, earlier[1].id)
        self.assertEqual(answer.consent, "pending")
        first, second, third, fourth = answer.activity
        self.assertEqual((first["text"], first["output"], first["is_error"]), ("manage_task", "Task listed", "false"))
        self.assertEqual((second["text"], second["output"], second["is_error"]), ("read_file", "Full file result", "false"))
        self.assertIn("example.py", second["input"])
        self.assertEqual(third["is_error"], "true")
        self.assertIn("Test error", third["output"])
        self.assertNotIn("output", fourth)
        self.assertEqual(import_history(self.config, self.vault, "antigravity", logs)["activity_updated"], 0)
