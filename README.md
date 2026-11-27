# DEM — Distill Every Model

> **Local-first, explicit-consent capture of LLM outputs into an open training-data commons.**
> Your AI memory. The AI commons.

DEM is a local daemon that captures the answers your LLM tools produce, redacts them on your machine, **asks you to approve the exact final text**, and — only if you choose — publishes it to HuggingFace as open training data for better open models.

**Nothing is captured silently. Nothing is uploaded automatically.** Those aren't slogans; they're enforced in code: consent is default-deny, publish requires an explicit reviewed action, and every run makes zero network calls until you publish.

```
Capture → encrypted local vault → redaction → human review → dedup → quality gate
        → versioned release → [explicit publish] → HuggingFace
```

---

## Quickstart

```bash
cd "Distill Every Model"
pip install -e .                 # or: pip install -r requirements.txt
dem init                         # data dir, keys, encrypted vault, default config
dem onboard                      # plain-language consent wizard (type "I CONSENT")
dem daemon                       # start capture proxies (Ctrl-C to stop)
```

Point Ollama at the local proxy:

```bash
export OLLAMA_HOST=http://127.0.0.1:11435
ollama run llama3.1 "Why is the sky blue?"
```

Then review, build, and publish:

```bash
dem review                       # localhost-only UI; approve/reject/edit each item
# or for scripted use: dem approve --all
dem release                      # dedup + quality gate + versioned release
dem publish --dry-run 0.3.0      # full local preview; NO network calls
dem publish --version 0.3.0      # explicit publish (HF token, else local export)
dem status                       # vault stats, consent state, releases
```

Everything lives under `DEM_HOME` (default `~/.local/share/dem`). To use a throwaway directory: `export DEM_HOME=/tmp/dem-test`.

---

## Import past coding-agent chats

From a checkout with DEM installed, run the following to prepare training pairs from your saved CLI sessions.

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
| Factory Droid | `droid` | Local Factory session JSONL |
| Gemini CLI | `gemini-cli` | Saved session JSON and JSONL |
| GitHub Copilot CLI | `copilot` | Local `session-state/*/events.jsonl` |
| Google Antigravity | `antigravity` | Local agent transcript JSONL when present |
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

Imported messages enter the encrypted review queue. A pair becomes eligible for the local restricted dataset once both halves are approved and clear quarantine. Tool calls, tool results and reasoning blocks are skipped. The training files live under `releases/v<version>/data/restricted/` inside the DEM data directory. Releases containing restricted records support local export only.

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
| Publish | dry-run preview (zero network), personal HF repo upload (or local export fallback), idempotent publish log |
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

Two consent switches are kept separate: **capture locally** ≠ **publish publicly**. An ephemeral session can override global toggles. Publication is irreversible; the UI says so, and retraction cannot guarantee erasure from caches/mirrors.

---

## ChatGPT / Claude / Gemini sources

You pay for these outputs, so they're yours to keep and contribute. Enable them with `dem attest`. Those records carry `derived_from: proprietary-api-output`, are tagged `user_owned_paid_output`, and are grouped in the `restricted` partition so the dataset stays clear about where each record came from. Nothing about them is hidden or second-class — the tag is just provenance.

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
| `dem publish --dry-run [X]` / `dem publish --version X` | preview / publish |
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

---

## The honest part

You paid for the outputs you contribute, so they're yours to share. DEM doesn't treat them as second-class data — the `restricted` tag simply records where each record came from, so anyone using the dataset knows which kind of model produced it. The operating rules are straightforward: local-first, consent-based, review-before-publish, and clear provenance on every record. That is what makes it possible for people to actually trust and want to use it.
