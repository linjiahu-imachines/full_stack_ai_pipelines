# How to run the voice AI pipeline

Operator guide for the **FastAPI voice chatbot** on `imu-thor` (example host `172.16.1.103`).

**Default (recommended):** one server start with **three LLM options** in the web UI dropdown — no restart required:

| UI label | Backend |
|----------|---------|
| Qwen2.5-1B LLM on NVIDIA Thor Machine with ARM64 CPU | Local HF `Qwen/Qwen2.5-1.5B-Instruct` (lazy load) |
| Gemma3-1B LLM on NVIDIA Thor Machine with ARM64 CPU | Local HF `google/gemma-3-1b-it` (lazy load; needs `HF_TOKEN`) |
| Gemma3-1B LLM on Intelligent Machine IMI-RISCV Qemu CPU Emulator | HTTP API at `172.16.1.7:8080` |

Config: [`demo_staged_kokoro_agent_dual.yaml`](../project/configs/demo_staged_kokoro_agent_dual.yaml)

**Last updated:** June 2026

---

## Quick start (dual LLM — UI selector)

```bash
pkill -f "uvicorn app.main:app" 2>/dev/null

cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent_dual.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the chat UI → use the **LLM model** dropdown before recording.

| UI option | Backend |
|-----------|---------|
| Qwen2.5-1B LLM on NVIDIA Thor Machine with ARM64 CPU | Local HF Qwen (lazy load) |
| Gemma3-1B LLM on NVIDIA Thor Machine with ARM64 CPU | Local HF Gemma3-1B (lazy load; `HF_TOKEN` in `.env.local`) |
| Gemma3-1B LLM on Intelligent Machine IMI-RISCV Qemu CPU Emulator | Remote HTTP API |

Verify models are listed:

```bash
curl -s http://127.0.0.1:8000/api/llm-models | python3 -m json.tool
```

---

## Local Gemma3-1B on Thor (Hugging Face)

Runs **in-process** on imu-thor (same as Qwen), much faster than the remote RISC-V simulator.

1. Accept the license: https://huggingface.co/google/gemma-3-1b-it  
2. Create a token: https://huggingface.co/settings/tokens  
3. Add to `server/.env.local`:
   ```bash
   HF_TOKEN=hf_your_token_here
   ```
4. Restart uvicorn. Select **Gemma3-1B LLM on NVIDIA Thor Machine with ARM64 CPU** in the UI.  
   First turn downloads and loads the model (may take several minutes).

Compare **local Gemma3** (fast, Thor ARM64) vs **remote Gemma3** (slow, IMI-RISCV Qemu) for the same model family.

---

## Remote simulator LLM (IMI-RISCV Qemu) — slow but valid

Use this when you need to prove the **Gemma3** path on the **IMI-RISCV Qemu CPU emulator** (`172.16.1.7:8080`). It is typically **10×–50× slower** than the local Qwen model on Thor — multi-minute (or longer) turns are normal with agent + RAG.

### 1. Set timeout in `server/.env.local`

```bash
# Add or update in server/.env.local
REMOTE_LLM_TIMEOUT_SEC=3600
```

| Value | Use |
|-------|-----|
| `3600` (default) | **1 hour** — recommended for simulator demos |
| `7200` | **2 hours** — long agent turns with full RAG + tool loop |

Restart uvicorn after editing `.env.local`.

### 2. UI settings

- **LLM model:** Gemma3-1B LLM on Intelligent Machine IMI-RISCV Qemu CPU Emulator  
- **Agent tools:** On (if testing RAG/tools)  
- Wait for **“Processing turn… remote simulator may take many minutes”** — do not refresh.

### 3. Verify remote API before voice

```bash
curl -s --max-time 120 -X POST http://172.16.1.7:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer abcdefg" \
  -d '{"model":"local-model","stream":false,"messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

### 4. What to expect in server logs

```
INFO staged_voice.llm: Remote LLM request | url=http://172.16.1.7:8080/v1/chat/completions | timeout_s=3600 | ...
INFO staged_voice.llm: LLM call #1 | ... | duration_s=...
INFO staged_voice.llm: LLM turn complete | calls=1 | llm_wall_s=...
```

If you still hit **504 timed out**, raise `REMOTE_LLM_TIMEOUT_SEC` or shorten the turn (fewer prior messages, Tools off for a plain-LLM smoke test).

---

## Legacy: two separate config files

You can still run with a **single** LLM locked at startup (no UI switch):

| | **Mode A — Local LLM** | **Mode B — Remote LLM** |
|---|------------------------|-------------------------|
| **Config file** | `demo_staged_kokoro_agent.yaml` | `demo_staged_kokoro_agent_remote.yaml` |
| **YAML `llm.backend`** | `hf` | `remote` |
| **Default model** | Qwen2.5-1.5B-Instruct (in-process) | `local-model` via HTTP API |
| **Startup time** | Slow (1–3+ min, loads HF weights) | Fast (~10 s, no local LLM load) |
| **ASR / TTS** | Local Whisper + Kokoro | Local Whisper + Kokoro |
| **RAG + agent** | Yes (both modes) | Yes (both modes) |

**Voice flow (both modes):**

```
Browser mic/WAV → ASR (local) → LLM (local OR remote) → TTS (local) → reply audio
```

---

## One-time setup

Run once on the host (or after a fresh clone):

```bash
cd /home/linhu/projects/speech_ai_exp/project
source .venv/bin/activate
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
cd ../server && pip install -e .
sudo apt install -y espeak-ng ffmpeg
```

Build the knowledge index (after adding/editing files under `server/data/knowledge/`):

```bash
cd /home/linhu/projects/speech_ai_exp/server
python scripts/build_knowledge_index.py
```

---

## Stop the server (always do this first)

```bash
pkill -f "uvicorn app.main:app"
```

Confirm port 8000 is free (no output = good):

```bash
ss -tlnp | grep 8000
```

If a process is stuck:

```bash
kill -9 $(ss -tlnp | grep 8000 | grep -oP 'pid=\K[0-9]+')
```

---

## Mode A — Local LLM (Hugging Face Qwen)

### 1. Start

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate

export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent.yaml

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Leave this terminal open. Wait until logs show:

```
Pipeline ready (llm=hf, tts=kokoro, whisper=cpu, agent=True)
Application startup complete.
```

### 2. Verify

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expect:

- `"pipeline_ready": true`
- `"config": ".../demo_staged_kokoro_agent.yaml"`
- `"agent": { "enabled": true, "knowledge_chunks": N, ... }`

### 3. Config reference

File: [`project/configs/demo_staged_kokoro_agent.yaml`](../project/configs/demo_staged_kokoro_agent.yaml)

```yaml
llm:
  backend: hf
  hf_model: Qwen/Qwen2.5-1.5B-Instruct
  hf_device: auto
  hf_max_new_tokens: 256
  temperature: 0.2
```

---

## Mode B — Remote LLM server

The LLM runs on a **separate machine**. This server sends the transcript (and agent context) to that machine’s HTTP chat API, then speaks the reply with local TTS.

### 0. (Optional) Test the remote LLM alone

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
python scripts/chat_completions_stream.py
```

Type a question. If this fails, fix network/firewall on the remote LLM host before starting the voice server.

### 1. Start

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate

export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent_remote.yaml

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for:

```
Pipeline ready (llm=remote, tts=kokoro, whisper=cpu, agent=True)
Application startup complete.
```

### 2. Verify

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Expect:

- `"pipeline_ready": true`
- `"config": ".../demo_staged_kokoro_agent_remote.yaml"`

### 3. Config reference

File: [`project/configs/demo_staged_kokoro_agent_remote.yaml`](../project/configs/demo_staged_kokoro_agent_remote.yaml)

```yaml
llm:
  backend: remote
  remote_base_url: http://172.16.1.7:8080
  remote_model: local-model
  remote_api_key: abcdefg
  max_tokens: 256
  temperature: 0.2
```

### 4. Override remote server without editing YAML

```bash
export REMOTE_LLM_BASE_URL=http://172.16.1.7:8080
export REMOTE_LLM_API_KEY=abcdefg
export REMOTE_LLM_MODEL=local-model
```

---

## Run in tmux (recommended)

Keep the server running after you disconnect:

```bash
tmux new -s voice_ai_server
# run Mode A or Mode B start commands above
# detach: Ctrl+B, then D
# reattach: tmux attach -t voice_ai_server
```

---

## Test in the web browser

### From your Windows laptop (microphone works)

**Terminal A** — SSH port forward (keep open):

```powershell
ssh -N -L 18080:127.0.0.1:8000 linhu@172.16.1.103
```

**Browser:** **http://127.0.0.1:18080/**

Use port **18080** on the laptop to avoid conflicts with other apps on port 8000.

### From the LAN (no mic, or use Upload WAV)

**http://172.16.1.103:8000/**

Browsers block the microphone on plain HTTP for LAN IPs. Use the SSH tunnel above for mic, or **Upload WAV** on the chat page.

### Voice vs text input

The chat UI supports two **input modes** (dropdown at top of page):

| Mode | How to use |
|------|------------|
| **Voice** | **Start listening** → speak → auto-send after pause; or **Upload WAV** |
| **Text** | Type in the text box → **Send** (Enter); optional **Speak reply (TTS)** checkbox |

Text mode skips ASR (faster). TTS is off by default in text mode; enable **Speak reply** to hear the answer.

API:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8000/api/sessions | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn/text" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hi, this is John Smith. Status of order 784921?","tools_enabled":true,"speak_reply":false}'
```

### What to try

Quick one-liners (e-commerce agent):

- *“Where is my latest order?”* — tests RAG
- Full **4-turn** script (RAG + web search + history): [DEMO_VOICE_QUERIES.md](DEMO_VOICE_QUERIES.md#featured-rag--web-search--conversation-history-4-turns)
- DevTools → Network → `turn` request → check `rag_sources` and `reply_text` in JSON

---

## Switch between modes

1. Stop the server: `pkill -f "uvicorn app.main:app"`
2. Change `STAGED_CONFIG` to the other YAML file
3. Start `uvicorn` again

| Mode | `STAGED_CONFIG` |
|------|-----------------|
| Local LLM | `.../demo_staged_kokoro_agent.yaml` |
| Remote LLM | `.../demo_staged_kokoro_agent_remote.yaml` |

Only **one** uvicorn process may listen on port **8000**.

---

## Knowledge base updates

After adding or editing files in `server/data/knowledge/`:

```bash
cd /home/linhu/projects/speech_ai_exp/server
python scripts/build_knowledge_index.py
# restart uvicorn
```

Supported formats: `.md`, `.txt`, `.rst` (not `.docx`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `address already in use` | Server already running | `curl .../health` or `pkill -f "uvicorn app.main:app"` |
| Health never returns / hangs | Stuck turn or crashed process | `kill -9` the uvicorn PID, restart |
| `pipeline_ready: false` | Model load error | Check terminal logs; for local LLM ensure HF cache/network OK |
| Remote mode slow or empty reply | Remote LLM slow or unreachable | Test with `chat_completions_stream.py`; check firewall |
| Mic not working on laptop | HTTP on LAN IP | SSH tunnel → `http://127.0.0.1:18080/` |
| `knowledge_chunks: 0` | Empty knowledge dir | Add files under `data/knowledge/`, rebuild index, restart |

---

## Related docs

| Document | Topic |
|----------|--------|
| [SERVER_AND_PROJECT1_ARCHITECTURE.md](SERVER_AND_PROJECT1_ARCHITECTURE.md) | Full system architecture |
| [PLAN_AGENTIC_VOICE.md](PLAN_AGENTIC_VOICE.md) | RAG + tools agent layer |
| [../server/README.md](../server/README.md) | Short server quick start |
