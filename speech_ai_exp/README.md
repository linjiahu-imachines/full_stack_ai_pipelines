# speech_ai_exp

Voice AI experiments: **staged pipeline (Project 1)** vs **omni voice-to-voice (Project 2)**.

| Path | Role |
|------|------|
| [`server/`](server/) | **Voice chatbot (FastAPI)** — multi-turn mic chat, Project 1 only |
| [`project/`](project/) | **Project 1** — `staged-voice`: ASR → LLM → TTS |
| [`project2/`](project2/) | **Project 2** — Mini-Omni / Moshi |
| [`compare/`](compare/) | Side-by-side JSON profiles |
| [`docs/`](docs/) | Architecture and research |

**Main architecture doc (server + Project 1):** [`docs/SERVER_AND_PROJECT1_ARCHITECTURE.md`](docs/SERVER_AND_PROJECT1_ARCHITECTURE.md)

## Voice chatbot (MVP)

```bash
cd project && source .venv/bin/activate
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
cd ../server && pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Default stack: Whisper + **Qwen2.5-1.5B-Instruct** + **Kokoro-82M** TTS (local inference).
