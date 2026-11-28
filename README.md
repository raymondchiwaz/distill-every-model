# DEM — Distill Every Model

> **Local-first, explicit-consent capture of LLM outputs into an open training-data commons.**
> Your AI memory. The AI commons.

DEM captures chats from your LLM tools, redacts them on your machine, and helps you share selected chats on Hugging Face. You can review the upload yourself or authorize your AI to review and submit a selected batch.

Capture is opt-in. Each community submission starts with your authorization for the selected chats and destination.

```
Capture → encrypted local vault → redaction → review → Hugging Face
```

---

## Quickstart

On macOS or Linux with Python 3.11+ and venv support, install the released CLI for your user account.

```sh
curl -fsSL https://github.com/raymondchiwaz/distill-every-model/releases/latest/download/install.sh | sh
~/.local/bin/dem review --open
```

Install with npm if you have Node.js, npm, and Python 3.11 or newer with `venv` and `pip` support.

```bash
npm install -g @distilleverymodel/distill-every-model
dem review --open
```

The first `dem` or `demd` command downloads the Python dependencies into a per-user cache. It needs internet access for that setup.

To install from a source checkout instead, run:

```bash
cd "Distill Every Model"
pip install -e .
dem review --open
```

The workspace opens in your browser. You can also use the localhost link printed in the terminal. Click **Choose a chat source**, select your tool, and import its saved sessions or enter a local transcript path. DEM asks for consent on first import. Open a conversation in **Chat history** to review it. Sharing to Hugging Face is optional.

### Capture live Ollama sessions

```bash
dem onboard                      # first-run consent for live capture
dem daemon                       # leave the capture proxy running
```

Point Ollama at the local proxy:

```bash
export OLLAMA_HOST=http://127.0.0.1:11435
ollama run llama3.1 "Why is the sky blue?"
```

Open the review workspace in another terminal:

```bash
dem review
```

Everything lives under `DEM_HOME` (default `~/.local/share/dem`). To use a throwaway directory: `export DEM_HOME=/tmp/dem-test`.

## Contribute to the Hugging Face community dataset

1. In **Chat history**, select chats and click **Approve selected chats**. You can also approve an open chat from its detail page.
2. Click **Export to Hugging Face** in Chat history. It previews approved pairs from your selection, or from all chats shown if nothing is selected. The default public destination is the [DEM community dataset](https://huggingface.co/datasets/Distilleverymodel/community-chats).
3. Click **Send to community**. DEM uses your Hugging Face CLI sign-in, or you can enter your own token with pull request access.

Your contribution becomes public as soon as the pull request opens. The dataset maintainer merges accepted contributions. DEM sends each batch to `data/submissions/` as a separate JSONL file.

To use a different community dataset, set it with `dem config publish community_repo owner/dataset`. The dataset needs to be public with discussions and pull requests enabled. Configure its dataset card to load `data/submissions/*.jsonl` into the training split.

### Have your AI review and submit

Use the [DEM contribution skill](.agents/skills/dem-contribute/SKILL.md) from this project. Sign in once with `hf auth login` using your Hugging Face account. Then give your agent a scoped request like this:

> Use the DEM contribution skill to review last week's OpenCode chat about the test runner and submit it to `owner/dataset` if it is safe to share publicly. I have permission to share it under that dataset's terms.

DEM runs redaction, privacy, and quality checks before showing the exact proposed upload. The agent checks it for sensitive details that the local rules may miss. It can skip questionable chats and send the cleared batch using `hf upload --create-pr` without asking you to review every pair. Use an agent you trust to see the redacted preview.

```bash
dem contribute --list
dem contribute --chat CHAT_ID --repo owner/dataset
dem contribute --chat CHAT_ID --repo owner/dataset --send REVIEW_TOKEN --via-hf-cli
```

### Upload to a private dataset

1. Open a chat, review its prompts, answers, and activity, then click **Approve entire chat**.
2. Expand **Upload to a private dataset**, then click **Upload to Hugging Face**. DEM uses your existing Hugging Face CLI sign-in or asks for a write token.
3. Click **Open on Hugging Face** to view your private dataset.

The review workspace uploads approved pairs, including imported chats, after deduplication and safety checks. A token entered in the workspace is kept in memory for the current DEM session. You can set the dataset name under **Dataset name and sign-in**.

On the review page, **Did this solve the task?** is optional. DEM attaches the contributor's choice to the latest reviewed prompt and response pair. **Task evidence** summarizes file changes, errors, and known checks reported in saved tool results. The dataset records that provenance alongside the contributor's outcome.

Code blocks are kept in the review text and upload after redaction. For older vaults configured to omit code, run `dem config capture capture_code true` to restore code from the saved originals and review the changed messages. Import chats again to restore code in saved activity.

For scripted publishing or a folder for manual upload, use the terminal:

```bash
dem release                      # build a release from approved captures
dem publish --dry-run            # preview locally
dem publish                      # confirm and upload the latest release
dem publish --export             # write a folder for manual upload
```

Use `dem publish --version X --repo owner/name` for scripted uploads. Add `--public` to upload to a public dataset. The CLI upload and export use the clean partition. Uploading to an existing dataset replaces files with matching names, including its dataset card.

---

## Import past coding-agent chats

With DEM installed, run the following to prepare training pairs from your saved CLI sessions.

```bash
dem onboard                           # once per vault
dem attest                            # include outputs from paid models
dem config capture capture_prompts true
dem import-history all                # or choose one source from the table below
dem review                            # manage chat switches and review messages
dem release                           # build local JSONL from approved pairs
```

DEM reads Claude Code project transcripts under `$CLAUDE_CONFIG_DIR/projects` (default `~/.claude/projects`) and Codex rollouts under `$CODEX_HOME/sessions` and `$CODEX_HOME/archived_sessions` (default `~/.codex`). To select one saved session or a copied transcript directory, run `dem import-history codex --path /path/to/session.jsonl` or use `claude-code` in its place. History must be available as local JSONL. Run `dem import-history all --watch` to keep picking up saved turns every ten seconds, or rerun the one-time import after a new chat. The `both` shortcut selects Claude Code and Codex. The summary counts imported pairs, previously imported pairs, unreadable files and malformed lines.

Open the localhost link printed by `dem review`. **Chat history** lists each imported conversation. Turn a chat off to pause importing new turns and exclude it from future releases. Turn it back on to restore its previous review decisions. Use **Import chats** to pick up saved turns. On first use, the interface asks for consent to import local prompts and responses, including outputs from paid models. Open any conversation to review its prompt and answer. Existing release files are snapshots.

In **Chat history**, check individual chats or click **Select all chats**, then click **Approve selected chats**. Select all follows the current title search, source, and status filters. Uploading is a separate action.

Use the **Codex**, **Claude Code**, and **OpenCode** shortcuts to jump directly to those chats. On a source page, the import button opens the importer with that tool selected. If your transcripts are saved elsewhere, use **Import from a file or folder** and enter the local path.

### Supported sources

| Tool | CLI source | History DEM can read |
| --- | --- | --- |
| Aider | `aider` | Current project's `.aider.chat.history.md`, or an explicitly selected history file |
| Amp | `amp` | Local thread JSON where available, or a thread JSON export |
| Claude Code | `claude-code` | Saved project JSONL |
| Cline | `cline` | VS Code task `ui_messages.json` |
| Codex | `codex` | Saved and archived rollouts |
| Continue | `continue` | Local session JSON |
| Crush | `crush` | Current project's `.crush/crush.db`, or an explicitly selected database |
| Cursor | `cursor` | Local agent JSONL and role-labeled text transcripts |
| DeepSeek Harness | `deepseek` | Compressed and plain JSONL in `$DSH_HOME/sessions` (default `~/.dsh/sessions`) and session export ZIPs |
| Factory Droid | `droid` | Local Factory session JSONL |
| Gemini CLI | `gemini-cli` | Saved session JSON and JSONL |
| GitHub Copilot CLI | `copilot` | Local `session-state/*/events.jsonl` |
| Google Antigravity | `antigravity` | App, CLI, and IDE transcript JSONL with tool results and matching full-log payloads when available |
| Goose | `goose` | Local `sessions.db` |
| Kilo Code | `kilo-code` | VS Code task `ui_messages.json` |
| Kimi Code CLI | `kimi-cli` | Local session `context.jsonl` |
| Kiro | `kiro` | Local CLI event history or IDE chat export ZIP containing `messages.jsonl` |
| Mistral Vibe | `mistral-vibe` | Session logs containing `messages.jsonl` |
| OpenCode | `opencode` | Both SQLite schemas and the JSON produced by `opencode export` |
| OpenHands | `openhands` | Local trajectory JSON/JSONL or ZIP of numbered events |
| Pi | `pi` | Local Pi JSONL session |
| Qwen Code | `qwen-code` | Local Qwen chat JSONL |
| Roo Code | `roo-code` | VS Code task `ui_messages.json` |

For a custom location or exported chat, run `dem import-history <source> --path /path/to/file-or-directory`. You can also place exports in `<DEM data dir>/history/<source>/` and click **Import chats**. DEM reads local files and labels each recovered conversation by source. Cloud-only chats must be exported in a supported format before import.

Imported messages enter the encrypted review queue. **Models used** lists recorded models in order. Open a model to jump to the first reply or activity at that point in the chat. **Tool calls** lists each saved invocation in transcript order, including repeated calls. Open an entry to see the redacted input, output, and status where the harness saved them. DEM keeps tool activity from turns without a final reply in the local vault. The detail view marks unavailable input and output explicitly. Coverage depends on what the harness writes to its local transcript. Plain-text logs such as Aider’s may expose output without a tool name. Detected subagent transcripts, including Claude Code, Codex, OpenCode, Crush, and Kiro sessions, appear as separate local tool histories. Use **Check saved chat for activity** to refresh a transcript in its usual location, or use **Import chats** to reimport a custom file. Newly added activity requires review before uploading.

---

## What v0.3 includes

| Layer | Implemented |
| --- | --- |
| Consent | default-deny, versioned `consent-v3`, per-source toggles, opt-in for paid-model outputs, signed receipts |
| Local storage | SQLite + Fernet field encryption, keys in separate `0600` files, WAL, retention TTL, hash-chained audit log |
| Privacy | secret/credential + PII detection, typed placeholders, fail-closed quarantine of secrets/financial IDs/third-party conversations/special-category data, code-block stripping, mandatory review |
| Capture | Ollama/local-model reverse proxy (streaming relay), generic base-URL API proxy, `dem run --` CLI wrapper, manual ingest |
| Review | localhost-only token-gated UI, raw→redacted diff, approve/reject/edit, bulk-approve local captures; captured text rendered as escaped plain text only |
| Pipeline | conversation pairing, exact + MinHash/LSH near-dup, weighted quality scoring with degenerate-output gate, PII re-scan |
| Releases | semver versions, clean vs restricted partitions, train/val/test group splits, manifest + SHA-256, QC report, auto dataset card |
| Publish | local preview, public Hugging Face pull requests, private dataset upload, and release export |
| Safety | kill switch (`dem kill`), emergency wipe (`dem purge`), no telemetry, egress only during explicit publish |

### Not in v0.3 (deliberate)
- **Browser extension / MV3** — high fragility risk; manual "save" path used instead.
- Screen OCR, keylogging, clipboard, browsing history — **categorically excluded**.
- Heavy ML NER redaction (Presidio) — v0.3 uses deterministic/entropy detectors; precision/recall must be measured before 1.0.
- Parquet shards, DPO pairs, k-anonymity/DP aggregate stats, sybil-resistant aggregation, foundation governance — designed, scheduled for v1.0.
- HTTPS MITM — not default; only an advanced opt-in for hardcoded-endpoint apps.

---

## The privacy contract (hard no's)

No silent capture · no silent auto-upload · no capturing others' devices/conversations · no bypassing DLP/MDM/provider access controls · no publishing third-party personal/confidential data, live secrets, PHI, or minors' data · no re-identification · no sale/ad targeting/telemetry · no claiming licenses we don't hold · no implying provider endorsement · no one-click "publish everything."

Local capture and public uploads each require user authorization. You can authorize an AI to review and send a selected batch. Public copies can remain in caches and mirrors after removal.

---

## ChatGPT / Claude / Gemini sources

Enable commercial-source imports with `dem attest`. These rows carry `derived_from: proprietary-api-output`. Community contributions retain that label and require permission to share the selected content publicly under the dataset's terms.

---

## Licensing note

Copyright 2026 DEM contributors. Licensed under the [Apache License, Version 2.0](LICENSE).

The default license is **Apache-2.0**, as requested. Apache-2.0 is a *software* license; it is applied here to data because it is permissive and carries an express patent grant. It is applied **only to records the contributor had the right to license** — third-party content is excluded by the redaction/review pipeline. Contributors who prefer a data-specific license can tag records `ODC-By-1.0` or `CC-BY-4.0`. Downstream users must do their own legal review. **This is not legal advice.**

---

## Commands

| Command | Purpose |
| --- | --- |
| `dem init` | initialize data dir, keys, vault, config |
| `dem onboard` | first-run consent wizard |
| `dem attest` | include outputs from paid models you use |
| `dem config get/set <section> <key> [value]` | inspect/change config |
| `dem daemon` / `demd` | start capture proxies and retention loop |
| `dem capture` / `dem captures --consent pending` | list events |
| `dem ingest --prompt ... --response ...` | manual test capture |
| `dem approve <id>` / `dem approve --all` / `dem reject ...` | review decisions |
| `dem review` | localhost review UI |
| `dem run -- <command>` | wrap a tool through capture proxies |
| `dem release [--version X]` | build deduped, quality-gated release |
| `dem publish [--dry-run \| --export] [--version X] [--repo owner/name] [--public]` | upload, preview, or export a Hugging Face dataset |
| `dem contribute --list` or `dem contribute --chat CHAT_ID` | find imported chats or preview a public contribution |
| `dem status` | status + vault stats |
| `dem kill` | trigger daemon kill switch |
| `dem purge` | emergency local wipe (requires typing PURGE) |

---

## Architecture

```
dem/
  cli.py            command line entrypoint
  daemon.py         long-lived daemon, retention, kill switch
  config.py         safe defaults + config.toml
  models.py         ULIDs, CaptureEvent, DatasetRecord, SHA-256
  crypto.py         key management, Fernet encryption, HMAC tags
  vault.py          encrypted SQLite store + audit log
  consent.py        versioned consent, onboarding, attestations
  privacy.py        redaction + fail-closed quarantine
  review.py         localhost-only review UI
  publisher.py      dry-run + HF publish / local export
  adapters/         ollama.py, api_proxy.py, cli_wrapper.py, base.py
  pipeline/         formatter.py, dedup.py, quality.py, release.py
tests/              unittest suite
```

---

## Roadmap

- **v0.4** — dataset lineage/impact receipts, contributor credits & leaderboards, privacy red-team eval set with published precision/recall, browser extension prototype.
- **v1.0** — Parquet shards + DPO views, k-anonymity & DP aggregate stats, sybil-resistant aggregation, before/after benchmark study, Commons Foundation governance + takedown SLA, external audit.

---

## Testing

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The tests cover redaction/quarantine, dedup, quality gates, encryption-at-rest, clean-vs-restricted partitioning, and an end-to-end release build.
