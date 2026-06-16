# Voice chat server & Project 1 architecture

Architecture reference for the **FastAPI voice chatbot** (`server/`) and the **staged pipeline** in **Project 1** (`project/`). This document is the main technical overview for operators and developers.

**Last updated:** June 2026

---

## 1. What this system is

### Staged voice assistant (Project 1)

Project 1 implements a **Lego / staged** voice assistant:

```
User speech (WAV) → ASR → text → [Agent: RAG + tools + LLM] → reply text → TTS → reply speech (WAV)
```

When the **agent layer** is enabled (default chatbot preset), the LLM step is wrapped by `AgentService`: it retrieves knowledge passages, may call tools, then produces a short spoken answer. With agent disabled, the pipeline is plain ASR → LLM → TTS.

Each stage is a **separate component** with explicit text between steps. That is different from **true voice-to-voice** (Project 2), where one model maps audio → audio without separate ASR/LLM/TTS in the hot path.

| Term | Meaning in this repo |
|------|----------------------|
| **Staged pipeline** | Project 1: ASR → LLM → TTS |
| **Voice chatbot (server)** | Web UI + REST API wrapping Project 1 with **multi-turn** sessions |
| **Agent layer** | Optional RAG + tool loop between transcript and TTS (no LangChain) |
| **RAG** | BM25 keyword retrieval over local knowledge files in `server/data/knowledge/` |
| **Voice-to-voice (omni)** | Project 2 only (`project2/`) — not used by this server |

The server delivers **turn-based voice conversation**: the user records one utterance per turn; the assistant replies with synthesized speech. Prior turns are passed to the LLM as **text history** (not as re-transcribed audio).

---

## 2. Repository context

```
speech_ai_exp/
├── project/          # Project 1 — staged-voice package (ASR → LLM → TTS)
├── server/           # FastAPI chatbot (uses Project 1 only)
├── project2/         # Project 2 — omni voice-to-voice (separate; not in server)
├── compare/          # JSON profile comparison (no runtime coupling)
└── docs/             # Architecture, research, plans
```

| Path | Role |
|------|------|
| [`project/`](../project/) | Core pipeline library (`staged-voice`), CLI, demos, YAML configs |
| [`server/`](../server/) | Browser chat, sessions, HTTP API |
| [`project2/`](../project2/) | Mini-Omni / Moshi experiments (benchmark sibling) |

**Design rule:** Do not merge Project 2 into the chat server. P1 remains the reference staged stack for MVP and benchmarking.

---

## 3. System architecture (deployment view)

How the MVP runs on a single host (e.g. `imu-thor` at `172.16.1.103`).

```mermaid
flowchart TB
  subgraph clients["Internal clients"]
    browser["Browser\nmic + speakers"]
  end

  subgraph host["Host — speech_ai_exp server"]
    subgraph http["HTTP :8000"]
      static["static/\nindex.html, app.js"]
      api["FastAPI\napp/main.py"]
    end
    store["SessionStore\ndata/sessions/"]
    svc["PipelineService\nsingleton + lock"]
    subgraph agent["Agent layer (optional)"]
      rag["KnowledgeBase\nBM25 + index JSON"]
      tools["ToolRegistry\nsearch, time, remember"]
      loop["AgentService\nTOOL_CALL / FINAL loop"]
    end
    kb[("data/knowledge/\n.md .txt .rst")]
    idx[("data/knowledge_index.json")]
    subgraph p1["Project 1 — in-process"]
      asr["ASR\nfaster-whisper"]
      llm["LLM\nHF Transformers"]
      tts["TTS\nKokoro-82M"]
    end
    ffmpeg["ffmpeg\nWebM → WAV"]
  end

  browser -->|"HTTP GET /"| static
  browser -->|"POST /api/.../turn"| api
  api --> store
  api --> ffmpeg
  api --> svc
  svc --> asr
  asr --> loop
  kb --> rag
  idx --> rag
  rag --> loop
  loop --> tools
  tools --> loop
  loop --> llm
  llm --> loop
  loop --> tts
  store -->|"session.json + WAVs\n+ agent_memory"| disk[("Disk")]
```

### Network access

| Audience | URL (example) |
|----------|----------------|
| Same machine | http://127.0.0.1:8000/ |
| LAN / VPN | http://172.16.1.103:8000/ |
| Health | http://172.16.1.103:8000/health |

### Runtime characteristics

| Property | Behavior |
|----------|----------|
| **Inference location** | All models run **on the host** (local); first run may download weights from Hugging Face |
| **Concurrency** | **One pipeline turn at a time** (global lock in `PipelineService`) |
| **Startup** | Models loaded once in FastAPI `lifespan` (1–3+ minutes cold start) |
| **Persistence** | Sessions on disk under `server/data/sessions/{id}/`; knowledge index at `server/data/knowledge_index.json` |
| **Auth** | None in v1 (trusted internal LAN MVP) |
| **Agent (default preset)** | RAG + tools enabled via `demo_staged_kokoro_agent.yaml` |

### System dependencies

| Tool | Purpose |
|------|---------|
| Python 3.10+ venv | `project/.venv` — staged-voice + server |
| `espeak-ng` | Kokoro phonemization; eSpeak TTS if configured |
| `ffmpeg` | Browser WebM → mono WAV for ASR |

### Optional process management

| Method | Use |
|--------|-----|
| `tmux new -s voice_ai_server` | Manual long-running `uvicorn` |
| `server/deploy/speech-chat.service` | systemd start/stop (`chatctl`) |

---

## 4. Project 1 architecture (staged pipeline)

### Package: `staged-voice`

| Item | Location |
|------|----------|
| Package root | [`project/src/staged_voice/`](../project/src/staged_voice/) |
| CLI | `staged-voice-run` → [`cli.py`](../project/src/staged_voice/cli.py) |
| Orchestrator | [`pipeline.py`](../project/src/staged_voice/pipeline.py) — `StagedVoicePipeline.run_turn()` |
| Config | [`config.py`](../project/src/staged_voice/config.py) + YAML overlay |

### Stage diagram

```mermaid
flowchart LR
  in["Input WAV\n(new utterance)"]
  asr["ASR\nFasterWhisperASR"]
  txt["Transcript\ntext"]
  agent["AgentService\n(optional)"]
  rag["RAG\nBM25 retrieve"]
  tools["Tools\nsearch, time, remember"]
  llm["LLM\nHFCausalLM or OllamaLLM"]
  reply["Reply text"]
  tts["TTS\nKokoro or eSpeak"]
  out["Output WAV\nreply"]

  hist["Prior turns\nuser/assistant text"] -.-> agent
  kb[("data/knowledge/")] --> rag
  in --> asr --> txt --> agent
  rag --> agent
  agent --> tools
  tools --> agent
  agent --> llm
  llm --> agent
  agent --> reply --> tts --> out
```

When `agent_runner` is **not** wired (agent disabled), the path is `txt → llm → reply` directly.

### Backends (pluggable per stage)

| Stage | Backend ID | Implementation | Default (chatbot YAML) |
|-------|--------------|----------------|-------------------------|
| **ASR** | `faster_whisper` | [`asr_faster_whisper.py`](../project/src/staged_voice/backends/asr_faster_whisper.py) | `small`, CPU, `int8` |
| **LLM** | `hf` | [`llm_hf_causal.py`](../project/src/staged_voice/backends/llm_hf_causal.py) | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **LLM** | `ollama` | [`llm_ollama.py`](../project/src/staged_voice/backends/llm_ollama.py) | Optional (separate Ollama daemon) |
| **TTS** | `kokoro` | [`tts_kokoro.py`](../project/src/staged_voice/backends/tts_kokoro.py) | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), voice `af_heart` |
| **TTS** | `espeak` | [`tts_espeak.py`](../project/src/staged_voice/backends/tts_espeak.py) | Fast baseline (not default for server) |

Factory: [`tts_factory.py`](../project/src/staged_voice/backends/tts_factory.py) selects TTS from `RunConfig`.

### One turn inside `run_turn()`

1. **ASR** — transcribe **only** the new user WAV.
2. **Agent (optional)** — if `agent_runner` is set:
   - Retrieve top-k knowledge chunks (BM25) from the user transcript.
   - Inject passages + tool descriptions into an extended system prompt.
   - Run an LLM **tool loop** (up to `max_tool_steps`): model emits `TOOL_CALL: {...}` or `FINAL: ...`.
   - Return grounded `reply_text` plus agent metadata (`rag_sources`, `tool_calls`).
3. **LLM (no agent)** — build messages: `system_prompt` + `history` + new transcript; stream tokens via `iter_chat_messages()`.
4. **TTS** — synthesize full reply text to `out_wav_path`.
5. **Profile** — return `StageProfile` (timings, transcript, reply text, agent meta, paths).

Hook in [`pipeline.py`](../project/src/staged_voice/pipeline.py): `agent_runner(transcript, history, llm, cfg) → (reply_text, meta)`.

Multi-turn memory for the LLM is **text-only** from earlier turns; old user audio is not re-fed to ASR. Session-scoped **agent memory** (`remember` / `recall` tools) is stored separately in `session.json`.

### Default chatbot presets

| Preset | Agent | Use |
|--------|-------|-----|
| [`demo_staged_kokoro_agent.yaml`](../project/configs/demo_staged_kokoro_agent.yaml) | **On** (default for server) | RAG + tools + Kokoro |
| [`demo_staged_kokoro.yaml`](../project/configs/demo_staged_kokoro.yaml) | Off | Plain multi-turn Q&A |

[`project/configs/demo_staged_kokoro.yaml`](../project/configs/demo_staged_kokoro.yaml) (plain chat, no agent):

```yaml
asr:
  model_size: small
  device: cpu
  compute_type: int8
llm:
  backend: hf
  hf_model: Qwen/Qwen2.5-1.5B-Instruct
tts:
  backend: kokoro
  kokoro_voice: af_heart
```

Agent preset adds an `agent:` block — see [§5 Agent layer](#agent-layer-rag--tools-no-langchain).

Override via environment variable `STAGED_CONFIG`.

**The chatbot does not use Ollama by default** — it uses `llm.backend: hf` (see below). Ollama is an optional swap via YAML.

### LLM backend: Hugging Face (default) vs Ollama (optional)

Project 1 supports two LLM backends. The server uses the same `PipelineService` wiring for both; only the YAML `llm.backend` field changes.

| | **`hf` (default)** | **`ollama` (optional)** |
|---|-------------------|-------------------------|
| **Used by MVP chatbot?** | **Yes** — `demo_staged_kokoro.yaml` | No, unless you point `STAGED_CONFIG` at an Ollama preset |
| **How it runs** | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) loaded **inside** the uvicorn process (`HFCausalLM`) | HTTP streaming chat to a **separate** [Ollama](https://ollama.com/) daemon |
| **Implementation** | [`llm_hf_causal.py`](../project/src/staged_voice/backends/llm_hf_causal.py) | [`llm_ollama.py`](../project/src/staged_voice/backends/llm_ollama.py) |
| **Multi-turn** | Same: `system_prompt` + prior user/assistant text + new transcript | Same message list via Ollama `/api/chat` |
| **Typical pros** | Self-contained server process; no extra service | Easier model swaps (`ollama pull`); can offload GPU RAM from the FastAPI process |
| **Typical cons** | HF weights loaded with ASR/TTS in one Python process | Requires Ollama installed, running, and reachable at `ollama_host` |

```mermaid
flowchart TB
  subgraph default["Default — llm.backend: hf"]
    uv1["uvicorn process"]
    uv1 --> qwen["Qwen2.5-1.5B\nTransformers in-process"]
  end

  subgraph ollama_opt["Optional — llm.backend: ollama"]
    uv2["uvicorn process"]
    daemon["Ollama daemon\n:11434"]
    uv2 -->|"HTTP POST /api/chat"| daemon
    daemon --> model["e.g. llama3.2,\nqwen2.5, mistral…"]
  end
```

#### Confirm which backend is active

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

- Open the `config` path and check `llm.backend`.
- On startup, logs show: `Pipeline ready (llm=hf, …)` or `Pipeline ready (llm=ollama, …)`.

#### Switch the chat server to Ollama

1. **Install and start Ollama** on the same host (or a machine reachable from the server).

   ```bash
   # Example: pull a model once
   ollama pull llama3.2
   ollama serve   # often already running as a system service
   ```

2. **Use an Ollama YAML preset** — start from [`project/configs/example_ollama.yaml`](../project/configs/example_ollama.yaml) or the chatbot-oriented preset below (Kokoro TTS + Ollama LLM).

3. **Point the server at that file and restart:**

   ```bash
   export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_ollama_kokoro.yaml
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   ASR and TTS stages are unchanged; only the LLM stage talks to Ollama.

**Example Ollama + Kokoro preset** ([`demo_staged_ollama_kokoro.yaml`](../project/configs/demo_staged_ollama_kokoro.yaml)):

```yaml
asr:
  model_size: small
  device: cpu
  compute_type: int8

llm:
  backend: ollama
  ollama_host: http://127.0.0.1:11434
  ollama_model: llama3.2          # must match a model you pulled in Ollama
  max_tokens: 256
  temperature: 0.3
  system_prompt: "You are a concise voice assistant in a multi-turn conversation. Answer in one or two short sentences suitable for text-to-speech."

tts:
  backend: kokoro
  kokoro_lang_code: a
  kokoro_voice: af_heart
  kokoro_speed: 1.0
```

| YAML key (Ollama) | Meaning |
|-------------------|---------|
| `ollama_host` | Base URL of the Ollama API (default in code: `http://127.0.0.1:11434`) |
| `ollama_model` | Model name as shown by `ollama list` |
| `max_tokens` | Cap on generated tokens (`num_predict` in Ollama options) |

**CLI without the server** (same Ollama backend):

```bash
staged-voice-run --audio path/to.wav --config configs/example_ollama.yaml \
  --llm-backend ollama --ollama-model llama3.2
```

### Where models execute

| Component | Execution |
|-----------|-----------|
| Whisper (ASR) | In-process (CTranslate2 / faster-whisper) |
| **LLM (`hf`)** | In-process — Hugging Face `AutoModelForCausalLM` (default Qwen2.5-1.5B-Instruct) |
| **LLM (`ollama`)** | **Out-of-process** — Ollama HTTP API on `ollama_host`; not loaded inside uvicorn |
| Kokoro (TTS) | In-process (`kokoro` + PyTorch); uses `espeak-ng` for phonemes |

No cloud inference API is required for the default stack. Ollama is still **local** on your network when `ollama_host` points to your machine or LAN.

### Other Project 1 entry points (no web UI)

| Tool | Port | Purpose |
|------|------|---------|
| `demo_staged_pipeline.py --serve` | 8765 | One-shot 3-stage demo |
| `run_demo_inputs_batch.py` + gallery | 8767 | Batch clip verification |
| `staged-voice-run` | — | CLI / JSON profiles |

---

## 5. Server architecture (FastAPI chatbot)

### Layout

```
server/
├── app/
│   ├── main.py              # Routes, lifespan, static mount
│   ├── config.py            # ServerConfig, STAGED_CONFIG
│   ├── pipeline_service.py  # Load pipeline once; inference lock; agent wiring
│   ├── session_store.py     # Sessions in memory + JSON on disk
│   ├── audio_util.py        # WebM → WAV (ffmpeg)
│   └── agent/               # RAG + tools (no LangChain)
│       ├── service.py       # Agent loop (TOOL_CALL / FINAL)
│       ├── rag.py           # Chunking, BM25 search, JSON index load
│       └── tools.py         # Tool registry
├── scripts/
│   └── build_knowledge_index.py   # Offline index builder
├── data/
│   ├── knowledge/           # Source documents (.md, .txt, .rst)
│   ├── knowledge_index.json # Pre-built chunk index (optional, recommended)
│   └── sessions/
```

### Component responsibilities

| Component | Responsibility |
|-----------|----------------|
| **`main.py`** | HTTP API, multipart upload, mounts UI at `/` |
| **`PipelineService`** | Builds `StagedVoicePipeline` at startup; loads `KnowledgeBase` + **`AgentService`** when agent enabled |
| **`SessionStore`** | CRUD sessions; `history_messages()` for LLM context; **`agent_memory`** for remember/recall |
| **`audio_util`** | Normalize browser audio to mono WAV for Whisper |
| **Browser (`static/`)** | Record WebM, POST turn, show transcripts + audio players |

### Multi-turn conversation model

```mermaid
sequenceDiagram
  participant UI as Browser
  participant API as FastAPI
  participant S as SessionStore
  participant P as StagedVoicePipeline
  participant A as AgentService

  UI->>API: POST /api/sessions (new chat)
  API->>S: create session_id
  loop Each voice turn
    UI->>API: POST /api/sessions/{id}/turn (audio)
    API->>S: history_messages() + agent_memory
    API->>P: run_turn(new_wav, history, agent_runner)
    P->>A: transcript + RAG retrieve + tool loop
    A-->>P: reply_text, rag_sources, tool_calls
    P-->>API: transcript, reply_text, reply_wav, agent meta
    API->>S: append TurnRecord, save session.json
    API-->>UI: JSON + audio URLs
  end
```

**Session disk layout:**

```
data/sessions/{session_id}/
  session.json          # turns + agent_memory {key: value}
  turn_001_user.wav
  turn_001_reply.wav
  turn_002_user.wav
  ...
```

Each `TurnRecord` may include an `agent` object: `rag_sources`, `tool_calls`, `agent_steps`, `agent_s`.

### HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI (static) |
| `GET` | `/health` | `pipeline_ready`, config path, **`agent`** block (chunks, index_loaded) |
| `GET` | `/api/sessions` | List sessions (newest first) |
| `POST` | `/api/sessions` | Create session |
| `GET` | `/api/sessions/{id}` | Full turn history |
| `DELETE` | `/api/sessions/{id}` | Delete session |
| `POST` | `/api/sessions/{id}/turn` | Multipart field **`audio`** → run pipeline |
| `GET` | `/api/sessions/{id}/audio/{filename}` | Play stored WAV |

Turn response includes: `transcript`, `reply_text`, `reply_audio_url`, `timings`, `context_messages`, `turn_count`, `rag_sources`, `tool_calls`, `agent_enabled`.

### Agent layer (RAG + tools, no LangChain)

When `agent.enabled: true` in YAML (default: [`demo_staged_kokoro_agent.yaml`](../project/configs/demo_staged_kokoro_agent.yaml)), the LLM stage is replaced by **`AgentService.run()`** — still using the same in-process HF or Ollama backend, but with retrieval and optional tool calls first.

#### High-level flow

```mermaid
flowchart TB
  t["User transcript"]
  r["Passive RAG\nBM25 top-k on transcript"]
  p["Extended system prompt\nbase + tools + passages"]
  l["LLM generate"]
  tc{"Output line?"}
  fin["FINAL: spoken answer"]
  tool["TOOL_CALL: run tool"]
  tr["Tool result → next LLM turn"]
  out["reply_text → TTS"]

  t --> r --> p --> l --> tc
  tc -->|FINAL| fin --> out
  tc -->|TOOL_CALL| tool --> tr --> l
  tc -->|unparseable| out
```

#### Knowledge base (RAG sources)

| Item | Location / rule |
|------|-----------------|
| **Source files** | `server/data/knowledge/` (`.md`, `.txt`, `.rst` only) |
| **Not indexed** | `.docx`, PDF, etc. — convert to `.txt` or `.md` first |
| **Chunking** | ~600 characters per chunk, paragraph-aware split |
| **Retrieval** | BM25 keyword scoring (no embeddings, no vector DB) |
| **Persisted index** | `server/data/knowledge_index.json` (built offline) |

Example layout:

```
server/data/knowledge/
  Intelligent_Machines_Company_Overview.txt   # primary company doc
  company_overview.md                         # short sample (optional)
  operator_notes.md                           # ops / FAQ hints
server/data/knowledge_index.json              # generated; load at startup
```

#### Index build workflow

Chunking runs **once per index build**, not on every voice turn. At startup the server loads the JSON index if it matches current source file mtimes/sizes; otherwise it re-chunks from disk.

```bash
cd /home/linhu/projects/speech_ai_exp/server
python scripts/build_knowledge_index.py
# restart uvicorn
```

Script options: `--knowledge-dir`, `--output`, `--chunk-chars` (default 600).

Verify after restart:

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
# expect: "knowledge_chunks": N, "index_loaded": true
```

| Health field | Meaning |
|--------------|---------|
| `agent.enabled` | Agent layer active |
| `knowledge_dir` | Absolute path to source documents |
| `knowledge_index` | Path to `knowledge_index.json` |
| `knowledge_chunks` | Number of indexed chunks |
| `index_loaded` | `true` if server loaded pre-built JSON (not live re-chunk) |
| `rag_top_k` | Passages retrieved per query (default 3) |

#### Two RAG modes

| Mode | When | How |
|------|------|-----|
| **Passive inject** | Every turn (`inject_rag: true`) | BM25 on user transcript → top-k chunks appended to system prompt as “Knowledge passages” |
| **Active search** | LLM chooses tool | `search_knowledge_base` tool runs BM25 with a custom `query` argument |

Passive RAG alone is sufficient for many questions. Small models (Qwen2.5-1.5B) often answer from injected passages without emitting `TOOL_CALL` lines.

#### Tool protocol

No OpenAI function-calling API — the model is prompted to output **exactly one line**:

```
TOOL_CALL: {"name": "search_knowledge_base", "arguments": {"query": "support hours"}}
```

or

```
FINAL: Our support hours are nine to five on business days.
```

Loop (max `agent.max_tool_steps`, default 3):

1. LLM generates text.
2. Parse `FINAL:` → done; text becomes TTS input.
3. Parse `TOOL_CALL:` → execute tool, append result as a user message, loop.
4. Neither → use raw output as fallback.

Implementation: [`server/app/agent/service.py`](../server/app/agent/service.py).

#### Built-in tools

| Tool | Purpose | Storage |
|------|---------|---------|
| `search_knowledge_base` | BM25 search with explicit `query` | Reads same index as passive RAG |
| `get_current_time` | Current UTC date/time | Ephemeral |
| `remember` | Store `key` → `value` for this session | `session.json` → `agent_memory` |
| `recall` | Read back a stored key | `session.json` → `agent_memory` |

Register more tools in [`server/app/agent/tools.py`](../server/app/agent/tools.py) via `ToolRegistry.register()`.

#### Agent YAML config

```yaml
agent:
  enabled: true
  knowledge_dir: data/knowledge
  knowledge_index: data/knowledge_index.json
  rag_top_k: 3
  max_tool_steps: 3
  inject_rag: true
```

| Key / env | Default | Purpose |
|-----------|---------|---------|
| `agent.enabled` / `AGENT_ENABLED` | `true` | Enable agent layer |
| `agent.knowledge_dir` / `AGENT_KNOWLEDGE_DIR` | `data/knowledge` | Source document folder |
| `agent.knowledge_index` / `AGENT_KNOWLEDGE_INDEX` | `data/knowledge_index.json` | Pre-built chunk index |
| `agent.rag_top_k` | `3` | Chunks retrieved per search |
| `agent.max_tool_steps` | `3` | Max tool rounds per turn |
| `agent.inject_rag` | `true` | Auto-inject passages into system prompt |

#### Turn API fields (RAG / tools observability)

`POST /api/sessions/{id}/turn` response:

| Field | Meaning |
|-------|---------|
| `rag_sources` | Source filenames retrieved this turn (e.g. `Intelligent_Machines_Company_Overview.txt`) |
| `tool_calls` | List of `{name, arguments, result}` (result truncated to 500 chars) |
| `agent_enabled` | Whether agent layer was active |

Example: ask *“What are the support hours?”* — expect `rag_sources` non-empty and reply grounded in knowledge files, even when `tool_calls` is empty.

Disable agent: `AGENT_ENABLED=false` or use `demo_staged_kokoro.yaml` (no agent block).

Further notes: [PLAN_AGENTIC_VOICE.md](PLAN_AGENTIC_VOICE.md)

### Configuration (server)

| Variable | Default |
|----------|---------|
| `STAGED_CONFIG` | `project/configs/demo_staged_kokoro_agent.yaml` (agent + Kokoro) |
| | Plain chat: `demo_staged_kokoro.yaml` or `AGENT_ENABLED=false` |
| `AGENT_KNOWLEDGE_DIR` | Override knowledge source folder |
| `AGENT_KNOWLEDGE_INDEX` | Override index JSON path |
| `SESSIONS_DIR` | `server/data/sessions` |
| `CHAT_HOST` / `CHAT_PORT` | `0.0.0.0` / `8000` |

---

## 6. End-to-end data flow (one voice turn)

```
Browser mic (WebM)
    → POST /api/sessions/{id}/turn
    → ffmpeg → turn_NNN_user.wav
    → Faster-Whisper → transcript
    → AgentService (when enabled):
         BM25 retrieve from knowledge_index.json / data/knowledge/
         optional tool loop (search_knowledge_base, remember, …)
         LLM (+ prior session text + injected passages) → reply_text
       OR direct LLM when agent disabled
    → Kokoro-82M → turn_NNN_reply.wav
    → JSON response (rag_sources, tool_calls) + GET audio URLs
    → Browser displays text + plays reply (auto-play when allowed)
```

**Latency (typical on edge CPU):** tens of seconds per turn (ASR + LLM + TTS sequential). Second concurrent user waits on the global lock.

---

## 7. Project 1 vs Project 2 (scope boundary)

| | Project 1 + server | Project 2 |
|---|-------------------|-----------|
| Architecture | Staged ASR → LLM → TTS | Omni audio → audio |
| Web MVP | **Yes** (`server/`) | Separate demo (port 8766) |
| Interpretability | Per-stage text + timings | End-to-end; optional sidecar ASR for demos |
| Use in compare | `profiles/*.json` | `profiles/*.json` |

See [`PROJECT2_SEMANTIC_RELEVANCE.md`](PROJECT2_SEMANTIC_RELEVANCE.md) and [`PLAN_PROJECT2_VOICE_TO_VOICE.md`](PLAN_PROJECT2_VOICE_TO_VOICE.md).

---

## 8. Operations quick reference

### Install

```bash
cd /home/linhu/projects/speech_ai_exp/project
source .venv/bin/activate
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
cd ../server && pip install -e .
sudo apt install -y espeak-ng ffmpeg
```

### Run

```bash
cd /home/linhu/projects/speech_ai_exp/server
source ../project/.venv/bin/activate
export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent.yaml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Knowledge index (after editing RAG documents)

```bash
cd /home/linhu/projects/speech_ai_exp/server
python scripts/build_knowledge_index.py
# restart uvicorn
```

### Verify

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
# expect: "pipeline_ready": true, "agent": {"enabled": true, "knowledge_chunks": N, "index_loaded": true}
```

**RAG smoke test (API):**

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8000/api/sessions | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
espeak-ng -w /tmp/rag_test.wav "What are the support hours?"
curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn" -F "audio=@/tmp/rag_test.wav" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('rag_sources:', d.get('rag_sources')); print('reply:', d.get('reply_text')[:200])"
```

### Admin (systemd, if installed)

```bash
server/deploy/chatctl start|stop|status|health
```

**Important:** Only one process may listen on port **8000** (manual `uvicorn` *or* `speech-chat.service`, not both).

---

## 9. Limitations (v1)

| Area | v1 behavior |
|------|-------------|
| Turn mode | Turn-based HTTP (no streaming ASR, no duplex WebSocket) |
| Auth / HTTPS | Not included — internal LAN MVP |
| RAG | BM25 keyword search only — no embeddings / semantic search |
| Tool calling | Prompt-based `TOOL_CALL` / `FINAL` text protocol; small LLMs may skip tools |
| Knowledge reload | Rebuild index + restart server (no hot reload API) |
| History cap | No token budget trim yet — long chats grow LLM prompt |
| Identity | Sessions are browser `localStorage` ids, not per-user accounts |
| Throughput | Single inference lock — queue under load |

---

## 10. Related documents

| Document | Topic |
|----------|--------|
| [`RUN_VOICE_PIPELINE.md`](RUN_VOICE_PIPELINE.md) | **How to run** — local LLM vs remote LLM server |
| [`PLAN_AGENTIC_VOICE.md`](PLAN_AGENTIC_VOICE.md) | RAG + tools agent layer (no LangChain) |
| [`project/configs/example_ollama.yaml`](../project/configs/example_ollama.yaml) | Minimal Ollama + eSpeak CLI preset |
| [`project/configs/demo_staged_ollama_kokoro.yaml`](../project/configs/demo_staged_ollama_kokoro.yaml) | Ollama LLM + Kokoro for the chat server |
| [`PLAN_PROJECT2_VOICE_TO_VOICE.md`](PLAN_PROJECT2_VOICE_TO_VOICE.md) | Omni / Project 2 plan |
| [`PROJECT2_SEMANTIC_RELEVANCE.md`](PROJECT2_SEMANTIC_RELEVANCE.md) | When to use P1 vs P2 |
| [`server/README.md`](../server/README.md) | Operator quick start (if present) |
| [`project/README.md`](../project/README.md) | CLI, demos, batch inputs |

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **Turn** | One user recording → one assistant reply |
| **Session** | Ordered list of turns + stored WAVs |
| **History** | Prior user/assistant **text** passed to the LLM |
| **Stage profile** | JSON timings and metadata from `run_turn()` |
| **Staged** | Explicit ASR, LLM, and TTS steps (Project 1) |
| **Agent layer** | RAG + tool loop wrapping the LLM between ASR and TTS |
| **RAG** | Retrieval-augmented generation — here, BM25 over local markdown/text files |
| **Knowledge index** | Pre-built JSON chunk file (`knowledge_index.json`) for fast startup |
| **Passive RAG** | Auto-inject retrieved passages into the system prompt each turn |
| **Tool call** | LLM-emitted `TOOL_CALL: {"name": ..., "arguments": ...}` line executed by `ToolRegistry` |
| **Ollama backend** | LLM stage calls a local Ollama daemon instead of in-process Hugging Face |
