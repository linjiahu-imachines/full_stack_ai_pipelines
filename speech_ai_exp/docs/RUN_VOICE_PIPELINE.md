# How to run the voice AI pipeline

Operator guide for the **FastAPI voice chatbot** on `imu-thor` (example host `172.16.1.103`).

**Default (recommended):** one server start with **both LLM options** available. Users pick **Local model on imu-thor** or **Remote model on sim IM CPU** from a dropdown in the web UI — no restart required.

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
| Local model on imu-thor | Qwen2.5-1.5B on imu-thor (lazy-loaded on first use) |
| Remote model on sim IM CPU | HTTP API at `172.16.1.7:8080` |

Verify models are listed:

```bash
curl -s http://127.0.0.1:8000/api/llm-models | python3 -m json.tool
```

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

### What to try

- *“What are the support hours?”* — tests RAG from knowledge base
- *“What is Intelligent Machines’ platform vision?”* — tests your company overview doc
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
