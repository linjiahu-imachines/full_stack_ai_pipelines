# Project 2 — semantic relevance (input vs reply)

**Date:** 2026-05-15

## Your test case

| Field | Value |
|-------|--------|
| Input (Whisper sidecar) | What is 2 plus 3? |
| **Moshi** reply text | Hey, how was your day? (duplex chitchat) |
| **Project 1** reply text | …2 plus 3 equals **5** (staged LLM) |

This is **expected** for Moshi on factual Q&A, not a broken WAV path.

## Why Moshi does not answer “5”

- **Moshi (`moshiko`)** is a **full-duplex conversational** model: it is trained to hold open-ended dialogue (greetings, small talk, overlap), not to act as a math tutor.
- Replaying a short WAV through the batch API still triggers **social openers** during and after your speech ([Kyutai discussion #157](https://github.com/kyutai-labs/moshi/issues/157)).
- **Moshiko vs Moshika** is only a voice timbre change, not reasoning style.

**Use Moshi in Project 2 for:** latency, duplex behavior, voice-to-voice architecture demos.

**Do not use Moshi for:** factual Q&A, procedures, compliance text — use **Project 1** (Whisper → LLM → TTS).

## Mini-Omni check (same WAV)

We tested [Mini-Omni](https://github.com/gpt-omni/mini-omni) on `test_voice.wav`:

| Mode | Result |
|------|--------|
| **A1_T2** (audio → text answer) | Mentions **2 plus 3 equals 5** |
| **A1_A2** (audio → audio) | Unrelated rambling (same class of problem) |
| **A1_T1** (ASR) | What is two plus three? |

So even another “omni” model may need the **audio→text→speech** path inside the stack for Q&A-style clips, not a single A1_A2 forward pass.

## Recommended test matrix

| Goal | Project | Backend / notes |
|------|---------|-----------------|
| “What is 2+3?” → correct **text + speech** | **P1** | `staged-voice-run` |
| Voice-to-voice **latency / architecture** | **P2** | `--backend moshi` + conversational prompt |
| Omni model with **better Q&A text** (experimental) | **P2** | `--backend mini_omni` (see `project2/README.md`) |
