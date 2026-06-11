# Research: Decagon vs Pipecat

**Purpose:** Brief for internal discussion on realtime AI voice / agent stacks.  
**Compiled:** 2026-05-08  

Sources are vendor and project sites; benchmark claims should be validated in your own pilots.

---

## Decagon

**Site:** https://www.decagon.ai/

### What it is

Decagon markets itself as an **enterprise “AI concierge” platform** for customer experience: a unified product layer across **voice, chat, and email**, with emphasis on non-developers defining behavior and operations teams iterating without heavy engineering lift.

### Positioning

Decagon is a **hosted application platform** (workflows, channels, optimization, analytics), not a low-level orchestration library. Public messaging highlights:

- **Agent Operating Procedures (AOPs)** — workflows described in **natural language** rather than a traditional configuration language, aimed at faster iteration and transparency for business stakeholders.
- **Omnichannel** — a single “intelligence layer” story across voice, chat, and email.
- **Voice** — realtime, brand-customizable voice agents with escalation to humans; materials reference **Decagon Voice 2.0** (latency and naturalness positioning, customization of tone/pacing/pronunciation, cross-channel memory) and **outbound** use cases.
- **Proactive / outbound** — public narrative includes **proactive agents**, **user memory** across sessions/channels, **outbound voice**, and tooling such as an **agent workbench** for debugging and iteration (see blog and resource announcements).
- **Decagon University** — training and certification for CX and technical teams.

### How to use this in architecture discussions

Decagon fits when the goal is **vendor-backed CX agent software**, faster iteration for **non-engineers**, and a **unified channel** strategy. You should still validate **data residency**, **security review**, **telephony integrations**, and **measured latency** in your stack—marketing pages optimize for differentiation, not your SLOs.

### Links

- https://www.decagon.ai/
- https://decagon.ai/product/voice
- Blog (product launches): https://decagon.ai/blog

---

## Pipecat

**Site:** https://pipecat.ai/  
**Repository:** https://github.com/pipecat-ai/pipecat  

### What it is

Pipecat is an **open-source Python framework** for **real-time voice and multimodal** conversational AI. It is **orchestration infrastructure**: transports, streaming audio, and composable integration of STT, LLM, TTS, and related services.

### Architecture (as described in docs and community)

Core concepts:

- **Frames** — streamed units (e.g. audio chunks, partial transcripts, LLM output, synthesized audio).
- **Frame processors** — components for STT, LLM, TTS, transforms, and guards.
- **Pipelines** — composition of processors into an end-to-end realtime path.

A typical described flow: ingest audio → transcribe → LLM → TTS → stream out, tuned for **low end-to-end latency**. Latency numbers on marketing or doc pages should be treated as **illustrative** until you benchmark on your network and model choices.

### Ecosystem

- Closely associated with **[Daily](https://www.daily.co/)** (WebRTC / media); Pipecat is often used with Daily transports and related tooling.
- **RTVI** (Real-Time Voice Interaction) — documented protocol layer between clients and servers for synchronizing transcription, bot output, and related events. https://docs.pipecat.ai
- **Client SDKs** for web, mobile, and embedded-style clients (per project README and site).
- **Pipecat Cloud** — hosted offering to run Pipecat-based agents at scale (scaling, telephony positioning, compliance claims—verify under your procurement and legal process).

### How to use this in architecture discussions

Pipecat is **not** the same category as Decagon’s product. It is the **glue** teams use to build **their own** realtime stack. You choose or swap ASR/LLM/TTS vendors, implement **turn-taking / barge-in**, and place **governance** (policy, retrieval, logging, human handoff) in surrounding systems or custom processors.

### Links

- https://docs.pipecat.ai/
- https://github.com/pipecat-ai/pipecat
- Daily product context: https://docs.daily.co/guides/products/ai-toolkit

---

## Comparison table

| Dimension | Decagon | Pipecat |
|-----------|---------|---------|
| **Category** | Enterprise CX concierge / AI agent platform | Open-source realtime voice/multimodal **orchestration** framework |
| **Primary audience** | CX, operations, enterprise digital | Engineers building agents |
| **Channels** | Voice, chat, email (unified product story) | Whatever you integrate (often WebRTC / voice-first) |
| **Governance** | Productized workflows, analytics, enterprise narrative | You implement (policy engines, backends, guardrails) |
| **Control vs integration burden** | Less custom plumbing; more platform coupling | More control; more you own end-to-end |

### Relationship

Both sit in the **real-time AI voice / agent** space. They are **complementary**, not direct substitutes: Pipecat is a plausible **technical substrate** for in-house builds; Decagon is a **buy** option when a full concierge stack and AOP-style iteration meet the requirements.

---

## Suggested evaluation steps

1. **Pilot scope** — one narrow journey (e.g. authenticated support, FAQ plus a single tool) with **latency**, **interruptibility**, and **audit trail** (transcript plus decision logging) as primary pass/fail metrics.
2. **Governance** — if using Pipecat, design **procedure and policy** outside the raw LLM (retrieval, allowed tools, escalation). If using Decagon, map those requirements to their workflow and observability model.
3. **Outbound voice** — if proactive/outbound matters, allocate time for **telephony regulation**, consent, and recording/storage policy regardless of vendor.
