# Plan: Project 2 — Voice-to-voice (Omni) vs Project 1 (staged baseline)

**Status:** Phase 0–1 implemented (Moshi batch turn, compare harness)  
**Date:** 2026-05-15  
**CEO reference:** [`CEO_ENGLISH_VOICE_TO_VOICE_LANDSCAPE.md`](CEO_ENGLISH_VOICE_TO_VOICE_LANDSCAPE.md)

---

## 1. Goals

| Goal | Detail |
|------|--------|
| **Build Project 2** | End-to-end **speech in → speech out** using Omni-style models (no separate ASR/LLM/TTS chain in the hot path). |
| **Keep Project 1 untouched** | Staged pipeline remains the **Lego baseline** for comparison; no refactors, no shared Python imports from P1 into P2. |
| **Comparable metrics** | Same test WAVs, comparable JSON profiles, optional side-by-side report (latency, output paths, optional quality notes). |
| **Phased delivery** | Start with **one** open model that runs on **imu-thor**; add CEO-listed models only after the harness works. |

## 2. Non-goals (for v1)

- Replacing or merging Project 1 into Project 2.
- Full **duplex** / barge-in benchmarks in v1 (Moshi supports duplex; first milestone is **turn-based** parity with P1 for fair wall-clock comparison).
- Production telephony, governance UI, or Pipecat/Decagon integration.
- GLM-4-Voice / LLaMA-Omni on day one (heavier, licensing/complexity; track as Phase 2+).

---

## 3. Isolation from Project 1 (must not impact P1)

### 3.1 Repository layout

```
speech_ai_exp/
├── project/                 # Project 1 — STAGED (frozen baseline)
│   ├── src/staged_voice/    # do not import from here in P2
│   ├── .venv/               # P1 venv (staged-voice, whisper, kokoro, …)
│   └── …
├── project2/                # Project 2 — VOICE-TO-VOICE (new)
│   ├── src/omni_voice/      # new package name (not staged_voice)
│   ├── .venv/               # separate venv (recommended)
│   ├── data/sample_in/       # symlink or copy same WAVs as P1 for tests
│   ├── audio/out/
│   ├── profiles/
│   ├── configs/
│   ├── experiments/
│   └── pyproject.toml       # separate package: omni-voice
├── compare/                 # OPTIONAL shared layer — no dependency on P1/P2 source
│   ├── schemas/             # common JSON profile shape
│   ├── run_compare.py       # read profiles/demo.json from both projects
│   └── reports/             # generated markdown/HTML tables
└── docs/
    ├── PLAN_PROJECT2_VOICE_TO_VOICE.md   # this file
    └── …
```

### 3.2 Rules

| Rule | Why |
|------|-----|
| **Separate Python package** (`omni-voice` vs `staged-voice`) | Different deps (Moshi stack vs faster-whisper/kokoro). |
| **Separate `.venv`** under `project2/` | Avoid version conflicts (torch, transformers, moshi wheels). |
| **No `import staged_voice` in project2** | P1 stays independently runnable and frozen. |
| **No edits to P1 code** except optional **one-line** pointer in repo-level README (not required inside `project/README.md`). |
| **Duplicate test audio** | Copy or symlink `test_voice.wav` into `project2/data/sample_in/` so runs do not depend on P1 paths. |
| **Shared only at repo root** | `compare/` reads **JSON files** from disk; no runtime coupling. |

### 3.3 What Project 1 remains

- Reference implementation for **CEO “Lego” pipeline**: Whisper → LLM → TTS (eSpeak or Kokoro).
- Source of **stage-level** metrics: `asr_s`, `llm_ttft_s`, `tts_s`, `transcript`, `reply_text`.
- Unchanged CLI: `staged-voice-run`, `demo_review.py`, etc.

---

## 4. CEO models → implementation priority

| Model (CEO doc) | Role | P2 priority | Notes for imu-thor |
|-----------------|------|-------------|-------------------|
| **Moshi (Kyutai)** | Full-duplex, ~200 ms, open weights | **P2.1 — first** | Best-documented OSS path; Apache 2.0; [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi). v1: **batch turn** (WAV → reply WAV), duplex later. |
| **Mini-Omni / Mini-Omni2** | Lightweight E2E, no external TTS | **P2.2** | Good edge story; verify GPU/RAM on Thor. |
| **LLaMA-Omni** | Llama 3.1 8B + speech | **P2.3+** | Larger; enterprise angle; more setup. |
| **GLM-4-Voice** | 9B bilingual | **Defer** | License/geo fit; not English-only baseline. |

**Recommendation:** Ship **Moshi batch-turn runner** first, then add backends behind a common `OmniBackend` protocol (same pattern as P1 backends).

---

## 5. Project 2 — technical shape (mirror P1 ergonomics)

### 5.1 Modes

| Mode | Description | When |
|------|-------------|------|
| **Batch turn** (v1) | Input WAV → model → output WAV + JSON profile | Fair comparison to P1 `run_turn` |
| **Streaming / duplex** (v2) | Mic or streamed chunks, overlap metrics | After batch harness stable |

### 5.2 Package sketch (`omni_voice`)

```
omni_voice/
├── cli.py                 # omni-voice-run
├── config.py
├── profiling.py           # OmniTurnProfile (see §6)
├── pipeline.py            # OmniVoicePipeline.run_turn(wav)
└── backends/
    ├── base_types.py      # OmniBackend protocol
    ├── moshi_backend.py   # P2.1
    └── mini_omni_backend.py  # P2.2 (stub until ready)
```

**CLI (parallel to P1):**

```bash
omni-voice-run \
  --audio data/sample_in/test_voice.wav \
  --profile-json profiles/moshi_run.json \
  --backend moshi
```

**Demo helper (parallel to P1):**

```bash
python3 experiments/demo_review.py --profile profiles/moshi_run.json --play-all --serve
```

(Reuse the **same UX** as P1 `demo_review.py` by copying the script into `project2/` or moving a generic version to `compare/` later — still **no import** from P1.)

### 5.3 Dependencies (separate `pyproject.toml`)

```toml
[project.optional-dependencies]
moshi = ["moshi", "torch", …]   # pin per Kyutai install docs
mini-omni = […]                 # later
```

Install example:

```bash
cd project2
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[moshi]"
```

Do **not** add Moshi to `project/pyproject.toml`.

---

## 6. Comparison harness (apples-to-apples)

### 6.1 Problem

- P1 profile: `StageProfile` (per-stage timings + text).
- P2 profile: end-to-end model; may not expose ASR/LLM/TTS splits natively.

### 6.2 Common schema (`compare/schemas/run_profile.json` concept)

Define a **neutral** JSON shape both sides can emit or be **converted into**:

```json
{
  "stack": "staged" | "moshi" | "mini_omni",
  "audio_path": "...",
  "audio_duration_s": 1.49,
  "e2e_wall_s": 12.3,
  "ttfa_s": 0.45,
  "output_wav_path": "...",
  "transcript": "optional — P1 native; P2 via sidecar Whisper if enabled",
  "reply_text": "optional — P1 native; P2 via sidecar ASR on output if enabled",
  "meta": { }
}
```

| Field | P1 | P2 |
|-------|----|----|
| `e2e_wall_s` | `asr_s + llm_generation_s + tts_s` (approx) | Total `run_turn` time |
| `ttfa_s` | `asr_s + llm_ttft_s` (staged TTFA) | Time to first output audio chunk |
| Per-stage | Native in P1 | N/A or empty |
| Sidecar text | Native | Optional `--sidecar-whisper` on output for audit only (not in hot path) |

### 6.3 Compare CLI (repo root)

```bash
cd compare
python3 run_compare.py \
  --staged profiles/../project/profiles/demo.json \
  --omni profiles/../project2/profiles/moshi_run.json \
  --out reports/vs_staged_moshi.md
```

**P1 is never modified** — `run_compare.py` only **reads** its JSON.

Optional: `run_matrix.py` runs both CLIs on the same WAV list and aggregates.

---

## 7. Implementation phases

### Phase 0 — Scaffold (1–2 days)

- [ ] Create `project2/` tree, `pyproject.toml`, empty `omni_voice` package.
- [ ] Copy/symlink `data/sample_in/test_voice.wav`.
- [ ] `OmniTurnProfile` + `write_profile_json`.
- [ ] Stub `omni-voice-run` that errors with “backend not installed”.
- [ ] Add `compare/` with schema doc + stub `run_compare.py`.
- [ ] Repo-level `speech_ai_exp/README.md` (index: P1 vs P2 vs compare) — **no P1 code changes**.

### Phase 1 — Moshi batch turn (3–5 days)

- [ ] `MoshiBackend`: load model, WAV in → WAV out (turn-based).
- [ ] Profile: `e2e_wall_s`, `ttfa_s`, `meta.moshi_*`.
- [ ] README for `project2/` (install, run, Thor/GPU notes).
- [ ] One successful run on `test_voice.wav`; store profile + output WAV.
- [ ] `compare/run_compare.py` produces table: P1 vs Moshi.

### Phase 2 — Mini-Omni backend (optional)

- [ ] Same `OmniBackend` interface.
- [ ] Second row in comparison reports.

### Phase 3 — Duplex / streaming (later)

- [ ] Moshi streaming session API.
- [ ] New metrics: interrupt handling, overlap ratio (see `RESEARCH_VOICE_TO_VOICE_MODELS.md`).

### Phase 4 — Sidecar audit trail for P2

- [ ] Optional post-hoc Whisper on input/output for `transcript` / `reply_text` in compare JSON (clearly labeled **not** model-internal).

---

## 8. Test matrix (shared inputs)

Use the **same files** for both projects:

| File | Purpose |
|------|---------|
| `test_voice.wav` | Short English question (primary demo) |
| `test.wav` | Secondary / silence+tone |
| (future) 3–5 curated clips | Noise, length, accent |

**Procedure per model:**

1. Run P1 → `project/profiles/<name>_staged.json`
2. Run P2 → `project2/profiles/<name>_omni.json`
3. Run compare → `compare/reports/<name>.md`
4. Demo: `demo_review.py` in each project (or unified in `compare/` later)

---

## 9. Success criteria

| Criterion | Measure |
|-----------|---------|
| P1 still runs unchanged | `staged-voice-run` + Kokoro path unchanged after P2 land |
| P2 runs E2E locally | `omni-voice-run` produces reply WAV + profile |
| Comparison report | Single doc/table: E2E latency, TTFA, paths, optional text |
| CEO alignment | Document which Omni model(s) are implemented vs deferred |

---

## 10. Risks (imu-thor / Jetson)

| Risk | Mitigation |
|------|------------|
| Moshi VRAM / latency | Start batch mode; document GPU requirement; fall back to CPU only if acceptable |
| Separate venv disk size | Two venvs; document `du -sh project/.venv project2/.venv` |
| No fair per-stage P2 metrics | Expected; compare E2E + optional sidecar ASR |
| Duplex harder to benchmark | Defer to Phase 3; CEO value still shown in README |

---

## 11. Documentation deliverables

| Doc | Location |
|-----|----------|
| This plan | `docs/PLAN_PROJECT2_VOICE_TO_VOICE.md` |
| Project 2 README | `project2/README.md` (after scaffold) |
| Moshi backend notes | `docs/RESEARCH_MOSHI_BACKEND.md` (optional) |
| Comparison report template | `compare/reports/README.md` |
| Repo index | `speech_ai_exp/README.md` (links P1, P2, compare, CEO docs) |

---

## 12. Decision summary

| Question | Decision |
|----------|----------|
| Where does P2 live? | **`speech_ai_exp/project2/`** — sibling of `project/` |
| How to avoid impacting P1? | **No imports, no shared venv, no P1 code edits**; compare reads JSON only |
| First Omni model? | **Moshi** (batch turn), then Mini-Omni |
| Comparison to P1? | **`compare/`** with common profile schema + `run_compare.py` |
| Match P1 UX? | Same flags pattern: `--audio`, `--profile-json`, `demo_review.py` copy in P2 |

**Next step when approved:** execute **Phase 0 scaffold** only (no Moshi download until you confirm GPU budget).
