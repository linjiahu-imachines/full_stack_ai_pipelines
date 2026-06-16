# Agent layer (RAG + tools) — no LangChain

Lightweight agent between **ASR transcript** and **TTS**, implemented in `server/app/agent/`.

## What it does

| Feature | Implementation |
|---------|----------------|
| **RAG** | BM25 keyword retrieval over `server/data/knowledge/`; index at `data/knowledge_index.json` |
| **Tools** | `search_knowledge_base`, `get_current_time`, `remember`, `recall` |
| **Session memory** | `remember` / `recall` store key–value facts in `session.json` |
| **Chat history** | Unchanged — prior turns still passed to the agent LLM loop |

## Enable / disable

**Default preset:** `project/configs/demo_staged_kokoro_agent.yaml`

```bash
export STAGED_CONFIG=/home/linhu/projects/speech_ai_exp/project/configs/demo_staged_kokoro_agent.yaml
# or disable without changing YAML:
export AGENT_ENABLED=false
```

Check `/health` → `"agent": {"enabled": true, "knowledge_chunks": N, ...}`

## Add knowledge

1. Add `.md`, `.txt`, or `.rst` under `server/data/knowledge/` (not `.docx`)
2. Rebuild the index and restart uvicorn:

```bash
cd server && python scripts/build_knowledge_index.py
```

Check `/health` → `"agent": {"enabled": true, "knowledge_chunks": N, "index_loaded": true, ...}`

## Tool protocol

The LLM is prompted to emit:

```
TOOL_CALL: {"name": "search_knowledge_base", "arguments": {"query": "..."}}
```

or

```
FINAL: Short spoken answer for the user.
```

Max tool steps: `agent.max_tool_steps` (default 3).

## Architecture

```
ASR → AgentService (RAG inject + tool loop) → reply text → TTS
         ↑
    data/knowledge/
    session.agent_memory
```

## API fields (per turn)

`POST /api/sessions/{id}/turn` response includes:

- `rag_sources` — document paths used
- `tool_calls` — name, arguments, truncated result
- `agent_enabled` — bool

Turn records in `session.json` include an `agent` object.

## Future (not in v1)

- Remote LLM on 172.16.1.77
- Embedding-based RAG
- LangGraph swap behind same `AgentService` interface

See also: [SERVER_AND_PROJECT1_ARCHITECTURE.md](SERVER_AND_PROJECT1_ARCHITECTURE.md)
