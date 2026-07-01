# speech_ai_exp

Voice AI experiments: **staged pipeline (Project 1)** vs **omni voice-to-voice (Project 2)**.

| Path | Role |
|------|------|
| [`server/`](server/) | **Voice chatbot (FastAPI)** — multi-turn mic chat, Project 1 only |
| [`project/`](project/) | **Project 1** — `staged-voice`: ASR → LLM → TTS |
| [`project2/`](project2/) | **Project 2** — Mini-Omni / Moshi voice-to-voice |
| [`compare/`](compare/) | Side-by-side JSON profile comparison |
| [`docs/`](docs/) | Architecture, research, and operator guides |

**Main architecture doc:** [`docs/SERVER_AND_PROJECT1_ARCHITECTURE.md`](docs/SERVER_AND_PROJECT1_ARCHITECTURE.md)

---

## Clone the repository

This project lives inside the **`full_stack_ai_pipelines`** monorepo:

```bash
git clone https://github.com/linjiahu-imachines/full_stack_ai_pipelines.git
cd full_stack_ai_pipelines/speech_ai_exp
```

All commands below assume your shell is in `speech_ai_exp/` (the repo root for this project).

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10–3.12** | **Not 3.13** — Kokoro TTS (`kokoro>=0.9.4`) requires `<3.13`. Tested on 3.12 |
| **Linux** | Ubuntu 24.04 on NVIDIA Thor (ARM64) |
| **`espeak-ng`** | Kokoro phonemization (`sudo apt install -y espeak-ng`) |
| **`ffmpeg`** | Browser WebM → WAV for ASR (`sudo apt install -y ffmpeg`) |
| **Disk / network** | First run downloads HF models (Whisper, Qwen, Kokoro, embeddings) |
| **PostgreSQL 16** (optional) | Commerce DB + hybrid RAG — see [Database setup](#database-setup-optional) |
| **GPU** (optional) | Project 2 Moshi / Mini-Omni; Project 1 runs CPU Whisper + Kokoro by default |

Install system packages once:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip espeak-ng ffmpeg
```

---

## One-time setup (Project 1 + voice server)

Project 1 and the FastAPI server share **one** virtualenv at `project/.venv`.

**Use a dedicated venv with Python 3.10–3.12.** Do not install into Conda `base` if it is Python 3.13 — `kokoro` will not resolve.

```bash
cd project

# Pick one interpreter in the 3.10–3.12 range (examples):
python3.12 -m venv .venv          # system Python 3.12
# conda create -n speech_ai python=3.12 && conda activate speech_ai && python -m venv .venv

source .venv/bin/activate
python --version                  # must show 3.10.x, 3.11.x, or 3.12.x — not 3.13+
pip install -U pip setuptools wheel
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"

cd ../server
pip install -e ".[vector]"
```

| Install target | What it provides |
|----------------|------------------|
| `project` extras | Whisper ASR, Hugging Face LLM, Kokoro TTS, YAML configs |
| `server[vector]` | Chroma + sentence-transformers for hybrid RAG |

---

## Configuration (secrets)

Create `server/.env.local` (git-ignored). The server loads it automatically at startup.

```bash
cd server
nano .env.local
```

Minimum for the **default demo** (commerce agent + web search):

```ini
# PostgreSQL — see docs/DATABASE_SETUP.md
DATABASE_SQL_URL=postgresql+psycopg://horizon:horizon@127.0.0.1:5432/horizon_store

# Tavily web search — https://tavily.com/
TAVILY_API_KEY=tvly-your-key-here
WEB_SEARCH_ENABLED=true

# Optional: gated Hugging Face models (e.g. Gemma3-1B in the UI dropdown)
# HF_TOKEN=hf_...

# Optional: remote LLM simulator (slow; internal network)
# REMOTE_LLM_BASE_URL=http://172.16.1.7:8080
# REMOTE_LLM_API_KEY=abcdefg
# REMOTE_LLM_TIMEOUT_SEC=3600
```

---

## Database setup (optional)

The default preset (`demo_staged_kokoro_agent_dual.yaml`) enables a **commerce database** (PostgreSQL) and **hybrid RAG** (Chroma + BM25). Skip this section for a minimal ASR → LLM → TTS smoke test; you need it to reproduce the **Alex Morgan / Horizon Store** customer-service demo.

**Full guide:** [`docs/DATABASE_SETUP.md`](docs/DATABASE_SETUP.md)

Quick path:

```bash
# 1. Install and start PostgreSQL
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql

# 2. Create DB user and database
sudo -u postgres psql <<'EOF'
CREATE USER horizon WITH PASSWORD 'horizon';
CREATE DATABASE horizon_store OWNER horizon;
GRANT ALL PRIVILEGES ON DATABASE horizon_store TO horizon;
EOF

# 3. Set DATABASE_SQL_URL in server/.env.local (see above)

# 4. Initialize schema, seed data, and vector index
cd server
source ../project/.venv/bin/activate
python scripts/init_databases.py
```

Build or refresh the **BM25 keyword index** after editing files under `server/data/knowledge/`:

```bash
cd server
source ../project/.venv/bin/activate
python scripts/build_knowledge_index.py
```

---

## Run the voice chatbot (main demo)

This is the primary way to reproduce the staged pipeline with the web UI.

```bash
cd server
source ../project/.venv/bin/activate

export STAGED_CONFIG="$(pwd)/../project/configs/demo_staged_kokoro_agent_dual.yaml"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/** in a browser.

### What the default stack does

```
Browser mic/WAV → ASR (Whisper) → Agent (RAG + tools + LLM) → TTS (Kokoro) → reply audio
```

The **LLM model** dropdown offers three backends (no server restart to switch):

| UI label | Backend |
|----------|---------|
| Qwen2.5-1B on Thor (ARM64) | Local HF `Qwen/Qwen2.5-1.5B-Instruct` (lazy load on first turn) |
| Gemma3-1B on Thor (ARM64) | Local HF `google/gemma-3-1b-it` (needs `HF_TOKEN` + HF license) |
| Gemma3-1B on IMI-RISCV Qemu | Remote HTTP API at `172.16.1.7:8080` (internal; very slow) |

**Operator guide (start/stop, tmux, troubleshooting):** [`docs/RUN_VOICE_PIPELINE.md`](docs/RUN_VOICE_PIPELINE.md)

### Verify the server is healthy

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expect `"pipeline_ready": true` and agent fields such as `knowledge_chunks` and (if DB is set up) `database_enabled: true`.

### Microphone from another machine

Browsers block the mic on plain HTTP for LAN IPs. From your laptop:

```bash
ssh -N -L 18080:127.0.0.1:8000 <user>@<server-host>
```

Then open **http://127.0.0.1:18080/** (not the LAN IP). Or use **Upload WAV** on the chat page.

### Reproduce the 4-turn e-commerce demo

With the server running, agent tools **On**, and DB + index initialized:

1. *"Where is my latest order?"*
2. *"Has the laptop from that order been delivered yet?"*
3. *"What's the weather forecast in Seattle today?"* (needs `TAVILY_API_KEY`)
4. *"Given the weather you just told me and my laptop shipment, should I worry about a delay on June eighteenth?"*

Script and expected RAG/tool behavior: [`docs/DEMO_VOICE_QUERIES.md`](docs/DEMO_VOICE_QUERIES.md)

---

## Project 1 CLI (batch, no server)

Run a single turn from the command line (useful for profiling):

```bash
cd project
source .venv/bin/activate

staged-voice-run \
  --config configs/demo_staged_kokoro.yaml \
  --audio data/sample_in/test_voice.wav \
  --profile-json profiles/demo.json
```

---

## Project 2 — voice-to-voice (separate venv)

Project 2 uses its **own** Python environment. Do not reuse `project/.venv`.

```bash
cd project2
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[experiment,sidecar]"
```

For Mini-Omni (recommended Q&A demo), also clone the upstream repo and install extras — see [`project2/README.md`](project2/README.md).

Example run:

```bash
omni-voice-run \
  --backend mini_omni \
  --config configs/example_mini_omni.yaml \
  --audio data/sample_in/test_voice.wav \
  --profile-json profiles/mini_omni_run.json \
  --moshi-device cuda
```

Compare Project 1 vs Project 2 profiles:

```bash
python3 compare/run_compare.py \
  --staged project/profiles/demo.json \
  --omni project2/profiles/mini_omni_run.json \
  --out compare/reports/vs_staged_mini_omni.md
```

---

## Documentation index

| Document | Topic |
|----------|--------|
| [`docs/SERVER_AND_PROJECT1_ARCHITECTURE.md`](docs/SERVER_AND_PROJECT1_ARCHITECTURE.md) | Full system architecture (server + Project 1) |
| [`docs/RUN_VOICE_PIPELINE.md`](docs/RUN_VOICE_PIPELINE.md) | Start/stop server, LLM modes, tmux, troubleshooting |
| [`docs/DATABASE_SETUP.md`](docs/DATABASE_SETUP.md) | PostgreSQL + Chroma hybrid RAG |
| [`docs/DEMO_VOICE_QUERIES.md`](docs/DEMO_VOICE_QUERIES.md) | 4-turn customer-service demo script |
| [`docs/PLAN_AGENTIC_VOICE.md`](docs/PLAN_AGENTIC_VOICE.md) | Agent layer (RAG + tools) |
| [`server/README.md`](server/README.md) | Short server quick start |
| [`project2/README.md`](project2/README.md) | Omni voice-to-voice install and run |

---

## Troubleshooting (common)

| Symptom | Fix |
|---------|-----|
| `No matching distribution found for kokoro>=0.9.4` / pip lists kokoro `0.7.x` only | **Python 3.13+.** Kokoro 0.9.4 requires `>=3.10,<3.13`. Recreate the venv with 3.12: `rm -rf project/.venv && python3.12 -m venv project/.venv`, activate, reinstall. Or `conda create -n speech_ai python=3.12`. |
| Packages install into `miniconda3/lib/python3.13/...` instead of `.venv` | Venv not activated (or created with wrong Python). Run `source project/.venv/bin/activate` and confirm `which python` points to `project/.venv/bin/python`. |
| `uvicorn: command not found` | `source project/.venv/bin/activate` |
| `address already in use` (port 8000) | `pkill -f "uvicorn app.main:app"` then restart |
| `pipeline_ready: false` | Check terminal logs; verify HF model download / GPU memory |
| `knowledge_chunks: 0` | Add files under `server/data/knowledge/`, run `build_knowledge_index.py`, restart |
| `database_enabled: false` | Set `DATABASE_SQL_URL` in `.env.local`, run `init_databases.py`, restart |
| Mic blocked in browser | Use SSH tunnel to `127.0.0.1` or **Upload WAV** |
| First local LLM turn very slow | Normal — weights lazy-load on first use |

More detail: [`docs/RUN_VOICE_PIPELINE.md`](docs/RUN_VOICE_PIPELINE.md) and [`server/README.md`](server/README.md).
