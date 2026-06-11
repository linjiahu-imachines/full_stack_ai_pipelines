# Kokoro-82M — research note (TTS for staged pipeline)

**Purpose:** Optional neural TTS backend for Project 1, aligned with CEO guidance on English “Lego” stacks.  
**Compiled:** 2026-05-15  

## What it is

- **Model:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — ~82M parameter English-oriented TTS (Apache 2.0).
- **Python package:** [`kokoro`](https://github.com/hexgrad/kokoro) (`pip install kokoro>=0.9.4`).
- **Role in pipeline:** Replaces **eSpeak** at the TTS stage; ASR and LLM unchanged.

## Why use it vs eSpeak

| | eSpeak NG | Kokoro-82M |
|---|-----------|------------|
| Quality | Robotic / utility | More natural neural speech |
| Latency | Very fast | Heavier (loads model; GPU helps) |
| Deps | `espeak-ng` binary | `kokoro` + PyTorch + **`espeak-ng`** (phonemization) |
| CEO note | Dev baseline | Named as top open English TTS choice |

## Integration in this repo

- Backend: [`tts_kokoro.py`](../project/src/staged_voice/backends/tts_kokoro.py)
- CLI: `--tts-backend kokoro` (optional `--kokoro-voice`, `--kokoro-lang`, `--kokoro-speed`)
- YAML: [`configs/example_kokoro.yaml`](../project/configs/example_kokoro.yaml)

```bash
pip install -e ".[asr-whisper,llm-hf,tts-kokoro,experiment]"
staged-voice-run \
  --audio data/sample_in/test_voice.wav \
  --profile-json profiles/kokoro_run.json \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --whisper-device cpu \
  --tts-backend kokoro \
  --kokoro-voice af_heart
```

## API sketch (library)

```python
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code="a")  # American English
for result in pipeline("Hello world.", voice="af_heart", speed=1.0):
    sf.write("out.wav", result.audio, 24000)  # 24 kHz mono
```

Common voices (American): `af_heart`, `af_bella`, `am_adam`, etc. — see Kokoro / Hugging Face model card.

## Requirements and caveats

- **`espeak-ng` still required** (phoneme backend for Kokoro).
- First run **downloads** weights (~100MB+) and may pull **spaCy** `en_core_web_sm`.
- **Jetson / edge:** test VRAM/RAM; cold start is slower than eSpeak.
- Output is **24 kHz**; input ASR WAV may be another rate — fine for listen-only reply file.

## References

- https://huggingface.co/hexgrad/Kokoro-82M  
- https://github.com/hexgrad/kokoro  
