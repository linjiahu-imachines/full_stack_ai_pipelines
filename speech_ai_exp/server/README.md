# Voice chat server (Project 1)

Turn-based **voice web chat**: ASR → LLM → TTS via the staged pipeline.

**Architecture (full):** [`../docs/SERVER_AND_PROJECT1_ARCHITECTURE.md`](../docs/SERVER_AND_PROJECT1_ARCHITECTURE.md)

## Quick start

```bash
cd /home/linhu/projects/speech_ai_exp/project && source .venv/bin/activate
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
cd ../server && pip install -e .
sudo apt install -y espeak-ng ffmpeg

export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/** (LAN example: **http://172.16.1.103:8000/**).

## Health

```bash
curl -s http://127.0.0.1:8000/health
```

## Admin

See **§8 Operations** in the architecture doc, or `deploy/chatctl` for systemd.
