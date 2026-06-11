# Research: Voice-to-voice (“speech-to-speech”) models in industry and academia

**Purpose:** Context for evaluating end-to-end or “native audio” conversational models versus staged ASR → LLM → TTS pipelines.  
**Compiled:** 2026-05-08  

Terms vary: **speech-to-speech**, **native audio**, **spoken language model**, **full-duplex spoken dialogue**. Not every product publishes full architectural detail; categories below distinguish **commercial APIs**, **open models**, **academic lines**, and **adjacent Japanese telecom / VC research**.

---

## 1. What “voice-to-voice” usually means

| Pattern | Idea | Typical audit trail |
|--------|------|---------------------|
| **Unified native audio LM** | One model stack consumes audio (and optionally text/vision) and emits streaming audio tokens or waveform-ready output; minimizes hand-built ASR↔LLM↔TTS glue. | Weaker plain-text guarantees unless the product logs transcripts or auxiliary text channels. |
| **Speech ↔ discrete tokens ↔ LM** | Audio is encoded (neural codec or speech tokenizer); a transformer predicts **speech/audio tokens** and/or auxiliary text (“inner monologue”); decoder turns tokens back to waveform. Often called E2E in papers. | Similar: compliance teams may require explicit STT-side logging as a shim. |
| **Tight cascaded pipeline** | Separate STT / LLM / TTS optimized for latency; marketed as realtime “voice AI” but not one neural E2E model. | **Strong** verbatim text from STT → LLM for procedures and QA. |

**Industry reality:** Many production systems still run **pipelines or hybrid stacks** because of observability, control, swapping vendors, and governance. Fully native offerings are spreading via **hyperscaler and foundation-model APIs**.

---

## 2. Widely cited **industry / API** offerings (native or speech-to-speech positioning)

These are influential for product teams building realtime agents—not an exhaustive vendor list.

- **OpenAI — GPT-4o / GPT Realtime (Realtime API)**  
  Official materials describe native multimodal modeling (text, vision, audio) with **speech-to-speech live sessions** as an option, versus an explicit chained STT→LLM→TTS pattern. Developer docs: https://platform.openai.com/docs/guides/voice-agents  
  Technical/safety framing: GPT-4o System Card — https://arxiv.org/pdf/2410.21276  

- **Google — Gemini “Native Audio” / Live-style APIs**  
  Positioned as processing **raw audio** in a unified low-latency path (vs classical STT→LLM→TTS), with Gemini Live–class entry points via AI Studio / Vertex AI.  
  Overview: https://ai.google.dev/gemini-api/docs/models  
  Example product writeup: https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai  
  Announcements: https://blog.google/products/gemini/gemini-audio-model-up  

- **Amazon — Nova Sonic**  
  Explicit **speech-to-speech** positioning on **Amazon Bedrock**: bidirectional streaming, designed for conversational agents with tool use / RAG in the documented stack.  
  Docs: https://docs.aws.amazon.com/nova/latest/userguide/speech.html  
  Announcement: https://aws.amazon.com/about-aws/whats-new/2025/04/amazon-nova-sonic-speech-to-speech-conversations-bedrock/  
  Model card / technical report hub: https://www.amazon.science/publications/amazon-nova-sonic-technical-report-and-model-card  

- **Zhipu AI — GLM-4-Voice** (China ecosystem)  
  Public reporting and technical summaries describe an **end-to-end speech-oriented LLM** path (speech in / speech out framing, emotion-related claims). Evaluate licensing, data, and ops fit if relevant to your geography.  

**Takeaway:** The **hyperscaler + frontier lab** trajectory is strongly toward APIs that advertise **native audio** or unified speech stacks; enterprises still often **mirror** transcripts for audits.

---

## 3. **Open-source / research lab** references (often self-hostable)

- **Kyutai — Moshi**  
  High visibility **full-duplex** spoken dialogue model and framework from Kyutai: parallel modeling of user and assistant audio, realtime targets, uses neural codec (**Mimi**). Apache 2.0. Strong reference implementation for teams experimenting with true conversational speech models.  
  Paper: https://arxiv.org/abs/2410.00037  
  Code: https://github.com/kyutai-labs/moshi  
  Blog: https://kyutai.org/blog/  

- **Mini-Omni** (streaming speech interaction, open reproductions)  
  Research line on **streaming speech-to-speech** / “talk while thinking” with open weights on Hugging Face; uses components such as Whisper + **SNAC** in described setups. Useful as an academic/comparative baseline.  
  Paper: https://arxiv.org/abs/2408.16725  
  https://github.com/gpt-omni/mini-omni  

- **Freeze-Omni** — low-latency speech-to-speech with **frozen** LLM  
  Interesting when you want to preserve a text LLM backbone while attaching speech adapters. https://arxiv.org/abs/2411.00774  

- **SpeechGPT** (Fudan; EMNLP 2023 Findings)  
  Early influential **speech in + speech out** LLM framing with discrete speech representations and cross-modal instructions—not the latest latency leader but important for the academic genealogy. ACL: https://aclanthology.org/2023.findings-emnlp.1055/  

- **SLAM-TR** — speech-to-speech **translation** (research)  
  Example of tuning a small LLM to map **speech to speech** with speech tokens (synonymous data challenges). Useful when the CEO note aligns with **translation** vs **same-language assistants**. Representative Interspeech / arXiv line (search SLAM-TR + speech tokens).  

- **SALMONN-omni** — full-duplex conversation angle  
  Recent research direction on standalone speech LLM behavior for duplex conversation — https://arxiv.org/html/2505.17060v1  

---

## 4. **Academic frontier** themes (actively researched)

Survey and benchmark directions are consolidating around **overlap, interruption, backchannels**, not only WER latency:

- **“From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models”** — taxonomy of engineered vs learned synchronization and evaluation gaps — https://arxiv.org/html/2509.14515v1  
- **Full-duplex / overlap evaluation** — e.g. **Full-Duplex-Bench** (overlap handling benchmarks) — https://arxiv.org/html/2507.23159v4  
- **Turn-taking modeling** surveys (handling timing of who speaks next) complement “voice-to-voice” stacks; see IWSDS / ACL anthology entries for recent reviews.  

Publication venues: **Interspeech**, **ACL/EMNLP/NAACL**, **ICASSP**, **NeurIPS/ICLR workshops** routinely carry speech LM and codec + LM papers.

---

## 5. **Japanese research / industry note** (“voice-to-voice” adjacent)

Japanese leadership often appears in **telecom-grade speech**, **speaker identity / personalization**, and **real-time voice conversion**, alongside LLM roadmaps—not always the same artifact as Western “speech foundation model APIs.”

- **NTT** has publicly described **individuality reproduction dialogue** and **few-shot/zero-shot personalized speech synthesis** atop their compact LLM roadmap (**“tsuzumi”**) and **Digital Twin / “Another Me”** narratives; also **high-quality low-latency real-time voice conversion** (preserve linguistic content while changing vocal style)—useful where the product goal is **persona fidelity** more than commodity dialog. Examples: https://group.ntt/en/newsrelease/2024/01/17/240117a.html and https://group.ntt/en/newsrelease/2024/06/17/240617a.html  
- Academic labs in Japan also publish across **speech synthesis, recognition, simultaneous translation, VC**; tying a specific citation to “the” Japanese voice-to-voice paper requires a narrower claim (journal / lab name)—the operational pattern is similar worldwide: **open weights are rarer**, **carrier R&D heavier on identity/trust/low latency**.

Mapping to your strategy line: Japanese work is especially relevant if the roadmap includes **trusted brand voice cloning**, **low-latency VC**, **telephony integration**, **JP prosody**.

---

## 6. Practical implications vs your governance sentence

Your internal note stated:

> *Voice-to-voice model (Japanese research) — Interesting for long-term simplification or quality; may not be the first choice when strict auditability and procedural control dominate.*

**Supported by landscape:**

1. **Open / research E2E** models improve **overlap behavior and expressive speech** more easily than brittle pipeline glue, but often give **messier intermediate text** unless you impose parallel ASR/transcript logging or “inner text” probes.  
2. **Hyperscaler native audio APIs** tighten product integration yet still require enterprise design for **transcript retention**, **approval of tool calls**, and **deterministicprocedure layers** external to raw audio.  
3. **Industry default** remains **measurable STT transcript + gated tools** where regulated procedures matter; Japanese carrier-style work reinforces **speaker identity / trust** axes.

---

## 7. Suggested deep-dive follow-ups for a POC

1. Benchmark **Nova Sonic**, **GPT Realtime**, **Gemini Native Audio** against your **golden-path dialogues** plus **overlap / interrupt** cases.  
2. For OSS / self-hosted experiments, prototype **Moshi** or speech-token–based stacks and compare **WER-style sidecar logging** overhead.  
3. If personalization is strategic, allocate a separate stream for **speaker adaptation** (few-shot cloning, VC)—potentially aligning with **NTT-class** disclosures rather than chasing one monolithic western API.
