"""Core v0.3 tests. Run: python3 -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from dem.adapters.base import CaptureSink
from dem.config import Config
from dem.crypto import Keyring
from dem.models import sha256_text, ulid
from dem.pipeline.dedup import Deduplicator
from dem.pipeline.formatter import build_records
from dem.pipeline.quality import score, score_all
from dem.pipeline.release import build_release
from dem.privacy import redact
from dem.vault import Vault


class TempEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = Config(path=self.root / "config.toml")
        self.config.set("general", "data_dir", str(self.root))
        self.dirs = self.config.ensure_layout()
        self.keys = Keyring(self.dirs["keys"])
        self.vault = Vault(self.dirs["vault"], self.keys)
        self.config.set("capture", "ollama_proxy", True)
        self.config.set("capture", "capture_prompts", True)

    def tearDown(self):
        self.vault.close()
        self.tmp.cleanup()


class TestPrivacy(TempEnv):
    def test_secret_redacted_and_quarantined(self):
        # Secrets are replaced for reviewer visibility AND the record is
        # quarantined (hard block) even if the user would click approve.
        result = redact("my key is sk-abcdefghijklmnopqrstuvwxyz0123", self.config)
        self.assertIn("SECRET", result.text)
        self.assertNotIn("sk-abcdef", result.text)
        self.assertIsNotNone(result.quarantine_reason)
        self.assertFalse(result.scrubbed)

    def test_email_redacted(self):
        result = redact("email me at alice@example.com please", self.config)
        self.assertIn("[EMAIL_1]", result.text)
        self.assertNotIn("alice@example.com", result.text)

    def test_fail_closed_on_credentials(self):
        result = redact("AWS_KEY=AKIAIOSFODNN7EXAMPLE ok", self.config)
        self.assertIsNotNone(result.quarantine_reason)
        self.assertFalse(result.scrubbed)

    def test_third_party_marker_quarantined(self):
        result = redact("patient diagnosis: NDA confidential record", self.config)
        self.assertIsNotNone(result.quarantine_reason)

    def test_code_block_removed_by_default(self):
        result = redact("answer\n```python\nprint('hi')\n```\ndone", self.config)
        self.assertIn("[CODE_BLOCK_REMOVED]", result.text)


class TestDedup(unittest.TestCase):
    def test_exact_and_near_duplicate(self):
        class R:
            def __init__(self, rid, text):
                self.id = rid
                self.prompt, self.response = text, ""
                self.quality = {"score": 0.8}
        first = R("a", "Explain how photosynthesis works in plants and algae")
        exact = R("b", "Explain how photosynthesis works in plants and algae")
        near = R("c", "Explain how photosynthesis works in plants and algae!")
        result = Deduplicator().run([first, exact, near])
        self.assertEqual(len(result.unique), 1)
        self.assertEqual(len(result.duplicates), 2)


class TestQuality(TempEnv):
    def test_repetitive_junk_low(self):
        class R:
            prompt = "Explain transformers"
            response = "yes " * 300
        value, _ = score(R(), self.config)
        self.assertLess(value, 0.6)

    def test_good_answer_high(self):
        class R:
            prompt = "Explain why the sky is blue"
            response = ("Rayleigh scattering preferentially scatters shorter wavelengths of "
                        "sunlight. Blue light is scattered in all directions more than red "
                        "light, which is why the sky appears blue during the day.")
        value, _ = score(R(), self.config)
        self.assertGreaterEqual(value, 0.6)


class TestPipeline(TempEnv):
    def test_end_to_end_release(self):
        sink = CaptureSink(self.config, self.vault)
        prompt = "What is 2 + 2?"
        response = "2 + 2 equals 4. This is basic arithmetic used in everyday calculations."
        preview = sink.prepare(prompt)
        event = sink.ingest(
            text=response, role="response", source="ollama", surface="localhost:11434",
            conversation_id="conv-1", turn_index=1, model_hint="llama3.1:8b",
            metadata={"prompt_text": preview.text},
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.consent, "pending")
        # Encrypt at rest: plaintext must not appear in the DB file.
        self.assertNotIn(b"2 + 2 equals 4", (self.dirs["vault"] / "vault.db").read_bytes())
        self.vault.update_consent(event.id, "approved")

        records = build_records(self.vault, self.config)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_tier, "clean")
        result = build_release(records, self.dirs["releases"], self.config)
        self.assertEqual(result["counts"]["accepted"], 1)
        release_dir = Path(result["release_dir"])
        self.assertTrue((release_dir / "manifest.json").exists())
        self.assertTrue((release_dir / "qc_report.json").exists())
        self.assertTrue((release_dir / "README.md").exists())
        self.assertTrue((release_dir / "data" / "clean" / "train" / "train-00000-of-00001.jsonl").exists())
        sha = (release_dir / "manifest.sha256").read_text().strip()
        self.assertEqual(len(sha), 64)

    def test_commercial_output_goes_to_restricted(self):
        sink = CaptureSink(self.config, self.vault)
        self.config.set("capture", "api_proxy", True)
        self.config.set("capture", "api_upstream", "https://api.openai.com")
        self.config.set("capture", "commercial_surfaces_default", "attest")
        event = sink.ingest(
            text="Here is a summary of the topic you asked about in detail.",
            role="response", source="api_proxy", surface="api.openai.com",
            conversation_id="conv-2", turn_index=1,
            metadata={"prompt_text": "Summarize the topic", "derived_from": "proprietary-api-output"},
        )
        self.assertIsNotNone(event)
        self.vault.update_consent(event.id, "approved")
        records = build_records(self.vault, self.config)
        self.assertEqual(records[0].source_tier, "restricted")
        accepted, rejected = score_all(records, self.config)
        self.assertEqual(len(accepted), 1)


if __name__ == "__main__":
    unittest.main()
