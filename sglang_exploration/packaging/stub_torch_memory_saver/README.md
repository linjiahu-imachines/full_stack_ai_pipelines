This directory builds a **minimal** `torch_memory_saver` distribution so `pip install sglang` can complete on machines without `/usr/local/cuda/include` (the real package compiles C++/CUDA extensions).

SGLang may still require the real extension for some features; this stub is only for bringing up the Python stack and running basic `Engine` tests.
