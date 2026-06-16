# Voice chat server (Project 1)

Turn-based **voice web chat**: ASR → LLM → TTS via the staged pipeline.

**Architecture (full):** [`../docs/SERVER_AND_PROJECT1_ARCHITECTURE.md`](../docs/SERVER_AND_PROJECT1_ARCHITECTURE.md) · **Run guide (local vs remote LLM):** [`../docs/RUN_VOICE_PIPELINE.md`](../docs/RUN_VOICE_PIPELINE.md) · **Agent (RAG + tools):** [`../docs/PLAN_AGENTIC_VOICE.md`](../docs/PLAN_AGENTIC_VOICE.md)

## Quick start

```bash
cd /home/linhu/projects/speech_ai_exp/project && source .venv/bin/activate
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
cd ../server && pip install -e .
sudo apt install -y espeak-ng ffmpeg

export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent_dual.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### LLM model selection (UI)

The default dual config exposes **Local model on imu-thor** and **Remote model on sim IM CPU** in the web UI dropdown. No restart needed to switch.

Legacy single-LLM configs (`demo_staged_kokoro_agent.yaml`, `demo_staged_kokoro_agent_remote.yaml`) still work but hide the dropdown choice.

Test the remote LLM API alone: `python scripts/chat_completions_stream.py`

Default preset includes **agent** (RAG over `data/knowledge/` + tools). Disable with `AGENT_ENABLED=false`.

### Knowledge base index

After adding or editing files under `data/knowledge/`:

```bash
cd server
python scripts/build_knowledge_index.py
# restart uvicorn
```

The server loads `data/knowledge_index.json` when it is up to date; otherwise it chunks from source files at startup.

Open **http://127.0.0.1:8000/** (LAN example: **http://172.16.1.103:8000/**).

### Microphone blocked on LAN IP?

Browsers only allow the mic on **HTTPS** or **http://127.0.0.1** — not `http://172.16.1.103:8000`.

**Fix (from your laptop):**

```bash
ssh -L 8000:127.0.0.1:8000 linhu@172.16.1.103
```

Then open **http://127.0.0.1:8000/** (not the 172.16 address).

Or on the chat page use **Upload WAV** to test without the mic.

## Health

```bash
curl -s http://127.0.0.1:8000/health
```

## Admin

See **§8 Operations** in the architecture doc, or `deploy/chatctl` for systemd.
