from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub import DatasetCard
from huggingface_hub.hf_api import RepoUrl

from dem.community import community_repo, prepare_contribution, submit_contribution
from dem.config import Config
from dem.models import CaptureEvent, DatasetRecord
from dem.pipeline.formatter import build_records
from dem.pipeline.release import SHARD_NAME, build_release
from dem.publisher import PublishError, export_hub_bundle, prepare_private_chats, publish, upload_hub_bundle
from tests.test_all import TempEnv


def record(tier, conversation, prompt, response):
    record_id = conversation + "-" + tier
    return DatasetRecord(
        id=record_id, prompt=prompt, response=response,
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        metadata={"conversation_id": conversation, "source": "fixture", "surface": "localhost"},
        quality={}, redactions=[], license="Apache-2.0", rights_attestation="fixture",
        source_tier=tier, derived_from="open-model-output" if tier == "clean" else "proprietary-api-output",
        consent_token="test-consent",
    )


class TestHuggingFace(TempEnv):
    def test_community_destination_defaults_for_existing_vaults_and_can_be_overridden(self):
        default_repo = "Distilleverymodel/community-chats"
        self.assertEqual(community_repo(Config(path=self.config.path)), default_repo)

        self.config.set("publish", "community_repo", "")
        self.config.save()
        self.assertEqual(community_repo(Config(path=self.config.path)), default_repo)

        self.config.set("publish", "community_repo", "other-org/community")
        self.config.save()
        configured = Config(path=self.config.path)
        self.assertEqual(community_repo(configured), "other-org/community")
        self.assertEqual(community_repo(configured, "explicit/dataset"), "explicit/dataset")

    def test_chat_batch_excludes_unselected_unapproved_and_disabled_pairs(self):
        for chat_id, answer_consent in (("selected", "approved"), ("unselected", "approved"),
                                       ("pending", "pending"), ("disabled", "approved")):
            for index, role in enumerate(("prompt", "response")):
                text = f"{chat_id} {role}"
                self.vault.add_capture(CaptureEvent(
                    id=f"{chat_id}-{role}", source="codex", surface="codex", role=role,
                    text=text, redacted_text=text, conversation_id=chat_id, turn_index=index,
                    consent="approved" if role == "prompt" else answer_consent,
                    metadata={"history_turn": chat_id},
                ))
        self.vault.set_history_chat_enabled("disabled", False)

        batch = build_records(self.vault, self.config, conversation_ids=("selected", "pending", "disabled"))
        self.assertEqual([record.prompt for record in batch], ["selected prompt"])
        self.assertEqual(build_records(self.vault, self.config, conversation_ids=()), [])
        self.assertEqual({record.prompt for record in build_records(self.vault, self.config)},
                         {"selected prompt", "unselected prompt"})

    def mixed_release(self):
        clean = record("clean", "conv-1", "Why is the sky blue?",
                       "Rayleigh scattering preferentially scatters shorter wavelengths of sunlight. "
                       "Blue light is scattered in all directions more than red light.")
        restricted = record("restricted", "conv-2", "How do leaves produce sugar?",
                            "Plants use photosynthesis to convert carbon dioxide and water into sugar "
                            "using light energy captured by pigments in their leaves.")
        release = build_release([clean, restricted], self.dirs["releases"], self.config)
        self.assertEqual((release["counts"]["clean"], release["counts"]["restricted"]), (1, 1))
        return Path(release["release_dir"]), release["version"]

    def test_export_only_includes_eligible_rows_and_a_card_for_existing_splits(self):
        release, _ = self.mixed_release()
        (release / "untracked.txt").write_text("This file does not belong in the bundle.", encoding="utf-8")
        with patch("huggingface_hub.HfApi") as api:
            export = export_hub_bundle(release, self.dirs["export"])
        api.assert_not_called()

        folder = Path(export["destination"])
        self.assertEqual({p.name for p in folder.iterdir()}, {"README.md", "train.jsonl"})
        card = DatasetCard.load(folder / "README.md")
        self.assertEqual(card.data.license, "apache-2.0")
        self.assertEqual(card.data.configs, [
            {"config_name": "default", "data_files": [{"split": "train", "path": "train.jsonl"}]},
        ])
        row = json.loads((folder / "train.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["prompt"], "Why is the sky blue?")
        self.assertEqual(row["messages"][1]["content"], row["response"])

    def test_modified_shard_is_rejected_before_accessing_the_hub(self):
        release, version = self.mixed_release()
        shard = release / "data" / "clean" / "train" / SHARD_NAME
        shard.write_bytes(shard.read_bytes().replace(b"the sky", b"a sky"))
        with patch("huggingface_hub.HfApi") as api:
            with self.assertRaisesRegex(PublishError, "release data failed verification"):
                publish(release, self.config, self.vault, self.dirs["keys"], version)
        api.assert_not_called()

    def test_private_upload_reuses_login_and_refuses_public_destination(self):
        release, version = self.mixed_release()
        destination = RepoUrl("https://huggingface.co/datasets/alice/distill-every-model-v0.3.0")
        self.config.set("publish", "repo", "dem/distill-every-model")
        with (patch.dict("os.environ", {}, clear=True),
              patch("huggingface_hub.get_token", return_value="fixture-token"),
              patch("huggingface_hub.HfApi") as factory):
            api = factory.return_value
            api.create_repo.return_value = destination
            api.repo_info.return_value = SimpleNamespace(private=False, sha="parent")
            with self.assertRaisesRegex(PublishError, "already public"):
                publish(release, self.config, self.vault, self.dirs["keys"], version)
            api.upload_folder.assert_not_called()
            self.assertEqual(self.vault.stats()["publishes"], 0)

            api.repo_info.return_value.private = True
            api.upload_folder.return_value = SimpleNamespace(oid="commit")
            result = publish(release, self.config, self.vault, self.dirs["keys"], version)

        factory.assert_called_with(endpoint="https://huggingface.co", token="fixture-token")
        api.create_repo.assert_called_with("distill-every-model-v0.3.0", repo_type="dataset",
                                           private=True, exist_ok=True)
        self.assertEqual(api.upload_folder.call_args.kwargs["parent_commit"], "parent")
        self.assertEqual(result["destination"], str(destination))
        row = self.vault.db.execute("SELECT repo, commit_sha, count FROM publish_log").fetchone()
        self.assertEqual(tuple(row), (destination.repo_id, "commit", 1))

    def test_review_upload_includes_imported_activity_and_enforces_private_destination(self):
        imported = record("restricted", "conv-2", "How do leaves produce sugar?",
                          "Plants use photosynthesis to convert carbon dioxide and water into sugar "
                          "using light energy captured by pigments in their leaves.\n"
                          "```python\nprint('photosynthesis')\n```")
        imported.activity = [{"type": "tool_call", "text": "read_file"}]
        imported.metadata["model"] = "example-model"
        imported.task_outcome = {"status": "solved", "source": "contributor"}
        imported.evidence = {"files_changed": [], "tool_errors": 0,
                             "checks": [{"name": "pytest", "status": "passed", "source": "recorded_tool"}]}
        folder = self.dirs["export"]
        bundle = prepare_private_chats([imported], folder, self.config)
        row = json.loads((folder / "train.jsonl").read_text(encoding="utf-8"))
        self.assertIn("```python\nprint('photosynthesis')\n```", row["response"])
        self.assertEqual(row["messages"][1]["content"], row["response"])
        self.assertEqual(row["activity"], imported.activity)
        self.assertEqual(row["model"], "example-model")
        self.assertEqual(row["task_outcome"], imported.task_outcome)
        self.assertEqual(row["evidence"], imported.evidence)
        self.assertEqual(row["derived_from"], "proprietary-api-output")
        self.assertNotIn("consent_token", row)
        self.assertIsNone(DatasetCard.load(folder / "README.md").data.license)
        with patch("huggingface_hub.HfApi") as factory:
            with self.assertRaisesRegex(PublishError, "only be uploaded to a private dataset"):
                upload_hub_bundle(folder, bundle, self.config, self.dirs["keys"], public=True)
            factory.assert_not_called()

    def test_community_upload_screens_payload_and_opens_a_pr_without_creating_a_repo(self):
        eligible = record("restricted", "conv-1", "Why is the sky blue?",
                          "Rayleigh scattering preferentially scatters shorter wavelengths of sunlight. "
                          "Blue light is scattered in all directions more than red light.")
        eligible.activity = [{"type": "tool_call", "text": "read_file", "input": "examples/sky.md"}]
        held = record("clean", "conv-2", "How do leaves produce sugar?",
                      "Plants use photosynthesis to convert carbon dioxide and water into sugar "
                      "using light energy captured by pigments in their leaves.")
        held.activity = [{"type": "tool_call", "text": "lookup", "output": '{"contact": "alice@example.org"}'}]
        self.config.set("privacy", "pii_redaction", False)
        contribution = prepare_contribution([eligible, held], self.config)
        self.assertEqual((contribution.count, contribution.skipped), (1, 1))
        row = json.loads(contribution.payload)
        self.assertEqual(row["activity"], eligible.activity)
        self.assertEqual(row["derived_from"], "proprietary-api-output")
        self.assertNotIn("consent_token", row)
        self.assertNotIn(b"alice@example.org", contribution.payload)
        self.assertNotEqual(contribution.review_token("org/dataset"), contribution.review_token("org/other"))

        with patch("dem.community.HfApi") as factory:
            api = factory.return_value
            api.repo_info.return_value = SimpleNamespace(private=True, sha="parent")
            with self.assertRaisesRegex(PublishError, "require a public dataset"):
                submit_contribution(contribution, "org/dataset", token="fixture-token")
            api.upload_file.assert_not_called()

            api.repo_info.return_value.private = False
            api.file_exists.return_value = False
            api.upload_file.return_value = SimpleNamespace(
                oid="commit", pr_url="https://huggingface.co/datasets/org/dataset/discussions/3",
            )
            result = submit_contribution(contribution, "org/dataset", token="fixture-token")
            api.create_repo.assert_not_called()
            args = api.upload_file.call_args.kwargs
            self.assertTrue(args["create_pr"])
            self.assertEqual(args["path_or_fileobj"], contribution.payload)
            self.assertEqual(args["path_in_repo"], f"data/submissions/{contribution.checksum}.jsonl")
            self.assertEqual(result["destination"], api.upload_file.return_value.pr_url)

    def test_community_cli_upload_passes_only_the_screened_file(self):
        eligible = record("clean", "conv-1", "Why is the sky blue?",
                          "Rayleigh scattering preferentially scatters shorter wavelengths of sunlight. "
                          "Blue light is scattered in all directions more than red light.")
        contribution = prepare_contribution([eligible], self.config)

        def hf_upload(command, **kwargs):
            self.assertEqual(command[:3], ["hf", "upload", "org/dataset"])
            self.assertEqual(Path(command[3]).read_bytes(), contribution.payload)
            self.assertIn("--create-pr", command)
            self.assertNotIn("fixture-token", command)
            return SimpleNamespace(stdout=f"https://huggingface.co/datasets/org/dataset/blob/refs%2Fpr%2F4/{command[4]}\n")

        with (patch("dem.community.HfApi") as factory,
              patch("dem.community.subprocess.run", side_effect=hf_upload) as cli):
            api = factory.return_value
            api.repo_info.return_value = SimpleNamespace(private=False, sha="parent")
            api.file_exists.return_value = False
            result = submit_contribution(contribution, "org/dataset", token="fixture-token", via_hf_cli=True)
            api.upload_file.assert_not_called()
            cli.assert_called_once()
            self.assertEqual(result["destination"], "https://huggingface.co/datasets/org/dataset/discussions/4")
