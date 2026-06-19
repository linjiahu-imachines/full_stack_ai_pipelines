# Agent layer (RAG + tools) — no LangChain

Lightweight agent between **ASR transcript** and **TTS**, implemented in `server/app/agent/`.

## What it does

| Feature | Implementation |
|---------|----------------|
| **RAG** | BM25 keyword retrieval over `server/data/knowledge/`; index at `data/knowledge_index.json` |
| **Tools** | `search_knowledge_base`, `search_web` (Tavily), `get_current_time`, `remember`, `recall` |
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

## Web search (Tavily)

Public web search uses the [Tavily](https://tavily.com/) Search API (AI-oriented search, similar to what many agents use).

1. Sign up at https://tavily.com/ and create an API key (`tvly-...`).
2. Enable in YAML (`demo_staged_kokoro_agent_dual.yaml`):

```yaml
agent:
  web_search:
    enabled: true
    provider: tavily
    max_results: 5
    search_depth: basic   # or advanced
```

3. Set the key in `server/.env.local` (recommended — loaded automatically at startup):

```bash
cd /home/linhu/projects/speech_ai_exp/server
cp .env.local.example .env.local
nano .env.local   # paste your tvly-... key
```

Example `server/.env.local`:

```
TAVILY_API_KEY=tvly-your-key-here
WEB_SEARCH_ENABLED=true
```

Or export in the shell instead: `export TAVILY_API_KEY=tvly-...`

4. Restart uvicorn. Check `/health` → `"agent": { ..., "web_search_enabled": true }`.

The LLM can call:

```
TOOL_CALL: {"name": "search_web", "arguments": {"query": "latest news about edge AI"}}
```

Use **`search_knowledge_base`** for Alex's orders, shipping, returns, and store policies; use **`search_web`** for public or current information online.

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

- Embedding-based RAG
- LangGraph swap behind same `AgentService` interface

See also: [SERVER_AND_PROJECT1_ARCHITECTURE.md](SERVER_AND_PROJECT1_ARCHITECTURE.md)
