# Demo voice query samples — e-commerce customer service

Four-turn demo for **Alex Morgan** at **Horizon Store** (fictional). Uses RAG, web search, and session conversation history in one chat.

**Knowledge source:** [`server/data/knowledge/alex_ecommerce_rag_context.txt`](../server/data/knowledge/alex_ecommerce_rag_context.txt)

**Prerequisites:** Server with [`demo_staged_kokoro_agent_dual.yaml`](../project/configs/demo_staged_kokoro_agent_dual.yaml). Index built (`python scripts/build_knowledge_index.py`). See [RUN_VOICE_PIPELINE.md](RUN_VOICE_PIPELINE.md).

---

## Quick reference — say these in order

Start a **new chat**. **Agent tools → On**. **Remote LLM** recommended.

| # | Query |
|---|--------|
| 1 | Where is my latest order? |
| 2 | Has the laptop from that order been delivered yet? |
| 3 | What's the weather forecast in Seattle today? |
| 4 | Given the weather you just told me and my laptop shipment, should I worry about a delay on June eighteenth? |

| Turn | RAG | Web search | Session history |
|------|-----|------------|-----------------|
| 1 | ✓ | — | — |
| 2 | ✓ | — | ✓ (turn 1) |
| 3 | — | ✓ (`search_web`, `auto`) | — |
| 4 | ✓ | — | ✓ (turns 2–3) |

---

## Demo setup

| Setting | Value |
|---------|--------|
| **Agent tools** | On |
| **LLM** | **Gemma3-1B LLM on Intelligent Machine IMI-RISCV Qemu CPU Emulator** (simulator — expect **many minutes** per turn) |
| **Session** | New chat |
| **Web search** | `TAVILY_API_KEY` + `WEB_SEARCH_ENABLED=true` in `server/.env.local` |
| **Browser** | http://127.0.0.1:8000/ · laptop tunnel: `ssh -N -L 18080:127.0.0.1:8000 linhu@172.16.1.103` → http://127.0.0.1:18080/ |

After each turn, check `rag_sources`, `tool_calls`, and `reply_text` in the API response.

---

## Remote simulator (Gemma3 on IMI-RISCV Qemu) — expected speed

**10–20+ minutes per turn is normal** when Agent tools are **On** (RAG passages + tool instructions in the prompt). The emulator runs Gemma3-1B on emulated RISC-V — often **10×–50× slower** than Qwen on Thor.

| Phase | Thor (local) | Remote simulator (typical) |
|-------|----------------|---------------------------|
| ASR + TTS | ~5–10 s | ~5–10 s (same, on Thor) |
| LLM only | ~5–30 s | **~5–20 min** (large agent prompt) |

**What to check in server logs** (confirms it is working, not hung):

```
INFO staged_voice.llm: Remote LLM request | ... | timeout_s=3600 | system_chars=...
INFO staged_voice.llm: LLM call #1 | ... | duration_s=...
INFO staged_voice.llm: LLM turn complete | llm_wall_s=...
```

### Faster remote smoke test (still validates simulator)

Use this before the full 4-turn demo:

| Setting | Value |
|---------|--------|
| LLM | Gemma3 remote |
| **Agent tools** | **Off** |
| Query | *Hello, can you hear me?* |

Plain LLM (no RAG) is much smaller input → often **1–3 minutes** on the simulator instead of 10+.

### Tuning (already applied in dual YAML)

- Remote `max_tokens: 96` — short spoken answers, less decode time  
- `rag_top_k: 3` — fewer RAG chunks in the prompt  
- `REMOTE_LLM_TIMEOUT_SEC=3600` in `.env.local` — allow long waits  

Restart uvicorn after YAML or `.env.local` changes.

---

## Turn-by-turn guide

### Turn 1 — RAG (order status)

**Say:** *Where is my latest order?*

**Expect:**

- `rag_sources`: `alex_ecommerce_rag_context.txt`
- Reply: order **ORD-2026-0612-7842** is **partially shipped** — laptop and dock in transit (UPS), mouse delivered to front desk, sleeve still processing

---

### Turn 2 — Conversation history + RAG

**Say:** *Has the laptop from that order been delivered yet?*

**Expect:**

- `rag_sources`: `alex_ecommerce_rag_context.txt`
- Reply: **No** — NovaBook X15 **in transit**; estimated **June 18, 2026** (UPS, afternoon window)
- Uses turn 1 history (“that order”); no need to repeat the order number

---

### Turn 3 — Web search (Tavily)

**Say:** *What's the weather forecast in Seattle today?*

**Expect:**

- `tool_calls`: `search_web` with `"auto": true`
- Reply: current Seattle weather from Tavily (not in the static RAG file)
- Seattle matches Alex’s default shipping address in the corpus

---

### Turn 4 — Conversation history + RAG

**Say:** *Given the weather you just told me and my laptop shipment, should I worry about a delay on June eighteenth?*

**Expect:**

- `rag_sources`: may include `alex_ecommerce_rag_context.txt`
- Reply ties together turn 2 (laptop / June 18) and turn 3 (weather) with grounded shipment facts
- Must **not** claim the laptop was delivered (forbidden in corpus)
- No invented delivery guarantee — estimate vs. actual delivery only

---

## Extra single-turn queries (RAG only)

| Say this | Expected answer |
|----------|-----------------|
| How much store credit do I have? | **USD 42.75** |
| Can I return my headphones? | **No** — return window ended **May 22, 2026** |
| Where was the mouse delivered? | **Front desk**, signed by **M. Chen**, **June 15, 2026** |
| Was I charged for the cancelled vacuum? | **No** — PayPal authorization voided |

---

## Contrast: Agent tools Off

Repeat turn 1 with **Agent tools → Off**.

- `rag_sources` and `tool_calls` empty
- Plain ASR → LLM → TTS (no RAG, no tools)

---

## API smoke test (curl + espeak-ng)

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:8000/api/sessions | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

espeak-ng -w /tmp/t1.wav "Where is my latest order?"
curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn" \
  -F "audio=@/tmp/t1.wav" -F "tools_enabled=true" -F "llm_model=remote_model_on_remote_sim_im_cpu" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('rag:', d.get('rag_sources')); print('reply:', (d.get('reply_text') or '')[:200])"

espeak-ng -w /tmp/t2.wav "Has the laptop from that order been delivered yet?"
curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn" \
  -F "audio=@/tmp/t2.wav" -F "tools_enabled=true" -F "llm_model=remote_model_on_remote_sim_im_cpu" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('rag:', d.get('rag_sources')); print('reply:', (d.get('reply_text') or '')[:200])"

espeak-ng -w /tmp/t3.wav "What is the weather forecast in Seattle today?"
curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn" \
  -F "audio=@/tmp/t3.wav" -F "tools_enabled=true" -F "llm_model=remote_model_on_remote_sim_im_cpu" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('tools:', d.get('tool_calls')); print('reply:', (d.get('reply_text') or '')[:200])"

espeak-ng -w /tmp/t4.wav "Given the weather you just told me and my laptop shipment, should I worry about a delay on June eighteenth?"
curl -s -X POST "http://127.0.0.1:8000/api/sessions/${SESSION}/turn" \
  -F "audio=@/tmp/t4.wav" -F "tools_enabled=true" -F "llm_model=remote_model_on_remote_sim_im_cpu" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('rag:', d.get('rag_sources')); print('reply:', (d.get('reply_text') or '')[:200])"
```

---

## Related documents

| Document | Topic |
|----------|--------|
| [RUN_VOICE_PIPELINE.md](RUN_VOICE_PIPELINE.md) | Start/stop server, index rebuild |
| [SERVER_AND_PROJECT1_ARCHITECTURE.md](SERVER_AND_PROJECT1_ARCHITECTURE.md) | RAG, tools, API |
| [PLAN_AGENTIC_VOICE.md](PLAN_AGENTIC_VOICE.md) | Agent layer design |
