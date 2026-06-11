# CEO notes: real-time voice AI — original text and analysis

**Source:** Email note from Mohammad (CEO)  
**Compiled:** 2026-05-08

---

## 1. Original notes (verbatim)

1. Real time AI voice (Decagon and Pipecat are both at the forefront of the shift toward advanced, real-time AI conversational agents, though they often serve different parts of the tech stack:  
   Decagon is an enterprise application platform for AI agents, while Pipecat is a framework for orchestration.)

2. Co-pilot agent assistant

3. Voice AI with rules and procedures governance

**Architecture options discussed**

- One single model  
- **Or** 2 models  
  - Audio encoding into text  
  - LLM model to generate text  
  - Text to voice  

**Technical references**

- Nemotron speech (Nvidia) speech streaming  
- Turn detection models (when to speak or interrupt)  
- Voice to voice model (Japanese Research work)  

**Product / UX**

- It can generate Survey and feedback offline  

Thanks  
Mohammad

---

## 2. Analysis

### 2.1 Three strategic pillars

**Real-time conversational voice**  
Decagon is positioned as an enterprise *application platform* for AI agents (routing, escalation, integrations, and often governance-oriented features). Pipecat is an *orchestration* framework: pipes, transports, and sequencing across components such as speech-to-text, the language model, and text-to-speech, including interrupts and bot behavior. They are complementary in many stacks: orchestration sits under or beside a higher-level agent platform unless the organization builds the full platform internally.

**Co-pilot agent assistant**  
This implies assisting a human operator (summarize, draft, suggest next steps) as distinct from a fully autonomous customer-facing voice agent. Copilots differ in requirements for approval flows, auditing, and tool access; clarifying whether the priority is human-in-the-loop assistance, autonomous voice self-service, or both will shape architecture and governance.

**Voice AI with rules and procedures (“governance”)**  
For regulated or brand-sensitive contexts, governance usually means: allowed scripts and flows, tools only through approved connectors, escalation paths, transcript handling, consent, and PII handling. Governance is typically enforced *around* the model (policy engine, approved knowledge bases, deterministic branches), not rely on prompt text alone.

### 2.2 “One model” vs staged pipeline

| Approach | Typical shape | Strengths | Trade-offs |
|----------|----------------|-----------|-------------|
| **Single end-to-end voice model** | Audio in → audio out (“voice-to-voice”) | Can simplify the stack and unify prosody / interaction in research | Harder word-level audit trails; tool use and grounded answers may need extra design |
| **Staged (e.g. ASR → LLM → TTS)** | Audio → text → text → audio | Clear text for compliance; easy to swap vendors; fits retrieval and procedural flows well | Requires careful integration for **latency**, **barge-in**, and **turn-taking** across stages |

The note set already mixes both ideas: streaming speech stack and turn detection (pipeline-friendly) alongside voice-to-voice research (end-to-end). A practical split is often **production = staged + governance**, with **R&D** on tighter integration or voice-to-voice where it materially helps.

### 2.3 Mapping technical bullets

- **Nemotron speech (NVIDIA), streaming** — Aligns with a GPU-centric, low-latency speech input/output stack; pairs with streaming ASR/TTS and chunked LLM inference.
- **Turn detection** — Distinct from “who spoke” in raw ASR; drives when the agent starts or stops speaking, **interrupts**, **endpointing**, and **barge-in** for natural realtime dialog.
- **Voice-to-voice model (Japanese research)** — Interesting for long-term simplification or quality; may not be the first choice when strict auditability and procedural control dominate.
- **Survey and feedback offline** — Suggests **asynchronous** capture after the session (e.g. SMS, email, short voice prompt) so realtime conversation quality is not sacrificed for long in-call surveys.

### 2.4 One-paragraph executive alignment (optional reply)

We are aligning on realtime voice with governance, likely combining a streaming speech path (including turn detection) with procedural and retrieval-backed control. Customer/runtime expectations may match an enterprise agent platform layer, while orchestration may use a framework-style approach (Pipecat-class or internal). Production can prioritize **auditability** via a staged stack where needed; **single-model voice-to-voice** can stay on the research track alongside NVIDIA streaming speech components. Offline survey and feedback preserves low-latency UX during the live interaction.
