# Optional third-party models (not in git)

## Mini-Omni (for `--backend mini_omni`)

```bash
cd project2/third_party
git clone --depth 1 https://github.com/gpt-omni/mini-omni.git
cd ../..
source .venv/bin/activate
pip install litgpt==0.4.3 snac openai-whisper lightning
```

First run downloads `gpt-omni/mini-omni` weights into `third_party/mini-omni/checkpoint/` (~2GB+).

`ffmpeg` is optional; Project 2 loads WAV via `soundfile` when ffmpeg is missing.
