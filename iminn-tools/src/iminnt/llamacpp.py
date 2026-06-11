from pathlib import Path
from typing import List, Dict, Optional
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, IMI_RISCV_ARCH, \
                    CROSS_TOOLCHAIN_PATH, IMI_ENV, RVV_ENV, USE_DEBUG_SYMS, \
                    PROMPTS_DIR, AMX_REMOTE, AMX_ENV, ORYON_ENV, ORYON_REMOTE

from .registry import register_class
from .core import BenchSimRunner


### LLama.cpp

class LlamaCppBase(BenchSimRunner):
    def __init__(self, 
                target: str,
                build_dir: str, 
                install_dir: str, 
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True,
                build_ref: bool = True,
                features: Optional[list] = None
                ):
        self._build_dir = build_dir
        self._install_dir = install_dir
        self.build_ref = build_ref
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu
        self.features = features or []

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/llama.cpp"

    # Other possible models to add here:
    # https://huggingface.co/tensorblock/optimum-internal-testing_tiny-random-qwen3-GGUF Includes all different quantizations, and uses Qwen architecture
    # https://huggingface.co/tiiuae/falcon-mamba-tiny-dev                   Needs quant, but good for mamba
    # In general, lots of options here: https://huggingface.co/trl-internal-testing
    @property
    def post_init(self) -> List[str]:
        return [
            f"git -C {self.root} remote add upstream https://github.com/ggml-org/llama.cpp",
            f"git -C {self.root} checkout upstream",
            f"git -C {self.root} checkout main",
            f"wget https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories15M-q4_0.gguf -P {self.root}/models",
            f"wget https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories15M-q8_0.gguf -P {self.root}/models",
            f"wget https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories15M.gguf -P {self.root}/models",
            # f"wget https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-F16.gguf -P {self.root}/models",
            # f"wget https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-BF16.gguf -P {self.root}/models", # also, `-F32` exists if BF16 gives trouble
            f"wget https://huggingface.co/prithivMLmods/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.F16.gguf -P {self.root}/models",
            f"wget https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf -P {self.root}/models",
            f"wget https://huggingface.co/Nikity/lille-130m-instruct/resolve/main/gguf/lille-130m-instruct-f32.gguf -P {self.root}/models",
            # f"wget https://huggingface.co/ngxson/tinygemma3_cifar" # Need to convert

        ]

    @property
    def upstream_remote(self) -> Optional[str]:
        return "https://github.com/ggml-org/llama.cpp"

    @property
    def target(self) -> str:
        return self._target

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def is_riscv(self) -> bool:
        return False
    
    @property
    def is_imi(self) -> bool:
        return False
    
    @property
    def is_aarch64(self) -> bool:
        return False

    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "llama.cpp"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""

        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        return [
            f"rm -rf {self.build_dir}",
            f"rm -rf {self.install_dir}",
            f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_args}",
            f"cmake --build {self.build_dir} -j{threads}",
            f"cmake --build {self.build_dir} -j{threads} -t install"
        ]

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {self.build_dir} -j -t install"]

    @property
    def test_bin(self) -> str:
        return f"{self.build_dir}/bin/test-backend-ops"
    
    @property
    def default_bin(self) -> str:
        return self.test_bin

    @property
    def use_qemu(self) -> bool:
        return self._use_qemu

    @property
    def remote_info(self) -> Optional[Dict]:
        return self._remote_info

    def get_sim_results(self, sim_dir: Path, return_perf_info = False):
        stat_file = (sim_dir / "stats.json")
        if not stat_file.exists():
            return None
        with open(str(stat_file), "r") as f:
            data = json.load(f)
        # Just grab the first result
        bench = data["benchmarks"]
        assert len(bench) > 0, f"Expected at least one benchmark result in {stat_file}"
        outputs = {}
        for b in bench:
            name = b["name"].split("/")[0]
            time_ns = float(b["cpu_time"])
            if "cpufreq" not in b:
                assert "context" in data and "mhz_per_cpu" in data["context"]
                cpu_freq = float(data["context"]["mhz_per_cpu"] / (10**3))
            else:
                cpu_freq = float(b["cpufreq"]) / (10**9)
            cycles = int(time_ns * cpu_freq)
            if return_perf_info:
                outputs[name] = {"cycles": cycles, "time_ns": time_ns, "freq": cpu_freq}
            else:
                outputs[name] = time_ns
        return outputs

    @property
    def cmake_args(self):

        base_args = {
            "CMAKE_BUILD_TYPE": "RelWithDebInfo",
            # "CMAKE_BUILD_TYPE": "Debug",
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "BUILD_SHARED_LIBS": "OFF",
            "GGML_OPENMP": "OFF",  # Keep OFF for cross-compilation compatibility
            "GGML_STATIC": "ON",
            "LLAMA_CURL": "OFF",
            "GGML_LLAMAFILE": "OFF",
            "GGML_CPU_REPACK": "ON",
            "GGML_DEFAULT_N_THREADS": "4"  # Changed from 1 to 4 for multi-threading
        }
        for f in self.features:
            base_args[f"GGML_{f}"] = "ON"
        # TODO: Clean this up, ugly logic
        if self.is_riscv or self.is_aarch64:
            base_args["CROSS_STATIC"] = "ON"
            base_args["CMAKE_TOOLCHAIN_FILE"] = f"{CROSS_TOOLCHAIN_PATH}"
        else:
            base_args["CMAKE_EXE_LINKER_FLAGS"] = "-static"
            base_args["CMAKE_SHARED_LINKER_FLAGS"] = "-static"
            base_args["CMAKE_FIND_LIBRARY_SUFFIXES"] = ".a"
        base_args["GGML_REF"] = "ON" if self.build_ref else "OFF"

        if self.is_riscv:
            base_args["GGML_NATIVE"] = "OFF"
            base_args["GGML_RV_ZFH"] = "ON"
            base_args["GGML_RV_ZVFH"] = "ON"
            base_args["GGML_RV_ZICBOP"] = "ON"
            base_args["GGML_RVV"] = "ON"
            base_args["GGML_RVV_VLEN"] = "128"
        elif self.is_aarch64:
            base_args["GGML_NATIVE"] = "OFF"
            base_args["GGML_CPU_KLEIDIAI"] = "ON"
            assert "CROSS_ARCH_VARIANT" in self.env
            base_args["GGML_CPU_ARM_ARCH"] = self.env["CROSS_ARCH_VARIANT"]
        elif not self.remote_info:
            base_args["GGML_NATIVE"] = "ON"
        else:
            base_args["GGML_NATIVE"] = "OFF"
        if USE_DEBUG_SYMS:
            base_args["DEBUG_SYMS"] = "ON"
        if self.is_imi:
            base_args["GGML_CPU_IMI"] = "ON"
        return base_args
    
    @property
    def default_runs(self) -> Dict:
        tests = {
            "optest_help": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "--help"},
            "debug_test": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o MUL_MAT -b REF -p type_a=q8_0,type_b=f32,m=16,n=1,k=256,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\]"},
            "test_mul_mat_all": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o MUL_MAT -b REF"},
            "optest_add": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o ADD -b REF -p type=f16"},
            "opperf_add": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o ADD -b CPU -p type=f32,ne=\[4096,1,1,1\],nr=\[1,1,1,1\]"},
            "optest_scratch": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o FLASH_ATTN_EXT -b REF"},
            "optest_out_prod": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o OUT_PROD -b REF"},
            "optest_norm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o NORM -b REF"},
            "optest_rms_norm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o RMS_NORM -b REF"},
            "optest_group_norm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o GROUP_NORM -b REF"},
            "optest_l2_norm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o L2_NORM -b REF"},
            "optest_scale": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o SCALE -b REF"},
            "optest_softmax": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o SOFT_MAX -b REF"},
            "optest_ce_loss_back": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o CROSS_ENTROPY_LOSS_BACK -b REF"},
            "optest_conv1d_transpose": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o CONV_TRANSPOSE_1D -b REF"},
            "optest_conv2d_transpose": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o CONV_TRANSPOSE_2D -b REF"},
            "optest_conv2d": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o CONV_2D_DW -b REF"},
            "optest_rwkv7": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o RWKV_WKV7 -b REF"},
            "optest_rwkv6": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o RWKV_WKV6 -b REF"},
            "optest_silu": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "test -o SILU -b REF"},
            "opperf_mul_mat_bf16_prefill_lg": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=4096,n=32,k=14336"},
            "opperf_mul_mat_bf16_decode_lg": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=4096,n=1,k=14336"},
            "opperf_mul_mat_bf16_prefill": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=128,n=32,k=256"},
            "opperf_mul_mat_bf16_decode": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=128,n=1,k=256"},
            "opperf_mul_mat_bf16_prefill_sm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=16,n=32,k=256"},
            "opperf_mul_mat_bf16_decode_sm": {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": "perf -o MUL_MAT -b CPU -p type_a=bf16,type_b=f32,m=16,n=1,k=256"},
            "smollm": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/smollm-135M-q4_0.gguf -r 1 -t 1 -ngl 0"},
            "stories_debug_q4_0": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"--progress -v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 0,1"},
            "stories_debug_q8_0": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"--progress -v -m {self.root}/models/stories15M-q8_0.gguf --repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 0,1"},
            "stories_prefill_bench": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 8,0"},
            "stories_decode_bench": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 1 -ngl 0 -n 0 -p 0 -pg 0,32"},
            "stories_prefill_bench_bf16": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/stories15M-bf16.gguf --repetitions 1 --threads 1 -ngl 0 -n 0 -p 32 --no-warmup"},
            "stories_decode_bench_bf16": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/stories15M-bf16.gguf --repetitions 1 --threads 1 -ngl 0 -n 32 -p 0 --no-warmup"},
           "test_q8_0_stories": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/stories15M-q8_0.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q8_0_lille": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/lille-130m-instruct-q8_0.gguf \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q8_0_smollm": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/smollm2-135m-q8_0.gguf  \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q4_0_stories": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q4_0_lille": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/lille-130m-instruct-q4_0.gguf \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q4_0_smollm": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/smollm2-135m-q4_0.gguf  \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "gpt_oss_scaled_test": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/gpt-oss-20b-pruned.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 16 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello.txt"},
           "gemma3_test_q8_0": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/gemma-3-270m-q8_0.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q4_0_stories_moe": {"bin": f"{self.install_dir}/bin/llama-cli", 
                            "args": f"-m {self.root}/models/stories15M_MOE-q4_0.gguf \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q8_0_stories_moe": {"bin": f"{self.install_dir}/bin/llama-cli", 
                            "args": f"-m {self.root}/models/stories15M_MOE-q8_0.gguf \
                                        --seed 42 \
                                        -t 1 -ngl 0 -n 32 \
                                        -no-cnv -st --no-warmup \
                                        --file {PROMPTS_DIR}/hello-world.txt"},
            "test_mxfp4_stories_moe": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/stories15M_MOE-mxfp4.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "stories_test_q4_0_hello": {"bin": f"{self.install_dir}/bin/llama-cli", 
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --seed 42 \
                                    -t 1 -ngl 0 -n 16 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello.txt"},
            "llama_cli_help": {"bin": f"{self.install_dir}/bin/llama-cli", "args": "--help"},

            # Multi-threaded test variants (2 threads)
            "test_q4_0_stories_2t": {"bin": f"{self.install_dir}/bin/llama-cli",
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --seed 42 \
                                    -t 2 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q8_0_stories_2t": {"bin": f"{self.install_dir}/bin/llama-cli",
                        "args": f"-m {self.root}/models/stories15M-q8_0.gguf \
                                    --seed 42 \
                                    -t 2 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},

            # Multi-threaded test variants (4 threads)
            "test_q4_0_stories_4t": {"bin": f"{self.install_dir}/bin/llama-cli",
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --seed 42 \
                                    -t 4 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},
            "test_q8_0_stories_4t": {"bin": f"{self.install_dir}/bin/llama-cli",
                        "args": f"-m {self.root}/models/stories15M-q8_0.gguf \
                                    --seed 42 \
                                    -t 4 -ngl 0 -n 32 \
                                    -no-cnv -st --no-warmup \
                                    --file {PROMPTS_DIR}/hello-world.txt"},

            # Multi-threaded llama-bench variants
            "stories_prefill_bench_2t": {"bin": f"{self.install_dir}/bin/llama-bench",
                        "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 2 -ngl 0 -n 0 -p 0 -pg 8,0"},
            "stories_decode_bench_2t": {"bin": f"{self.install_dir}/bin/llama-bench",
                        "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 2 -ngl 0 -n 0 -p 0 -pg 0,32"},
            "stories_prefill_bench_4t": {"bin": f"{self.install_dir}/bin/llama-bench",
                        "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 4 -ngl 0 -n 0 -p 0 -pg 8,0"},
            "stories_decode_bench_4t": {"bin": f"{self.install_dir}/bin/llama-bench",
                        "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf --repetitions 1 --threads 4 -ngl 0 -n 0 -p 0 -pg 0,32"},

            # Multi-threaded batched-bench variants
            "stories_q4_0_bbench_2t": {"bin": f"{self.install_dir}/bin/llama-batched-bench",
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --no-op-offload --cpu-moe --batch-size 256 --ubatch-size 32 -npp 16 -ntg 4 -npl 0 \
                                    --threads 2 --threads-batch 2 --n-gpu-layers 0 --predict 1 \
                                    --kv-unified"},
            "stories_q4_0_bbench_4t": {"bin": f"{self.install_dir}/bin/llama-batched-bench",
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --no-op-offload --cpu-moe --batch-size 256 --ubatch-size 32 -npp 16 -ntg 4 -npl 0 \
                                    --threads 4 --threads-batch 4 --n-gpu-layers 0 --predict 1 \
                                    --kv-unified"},

            "print_layers": {"bin": f"{self.install_dir}/bin/llama-eval-callback", 
                            "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                        --seed 42                                \
                                        -t 1 -ngl 0 -n 16                        \
                                        --prompt Hello"},
            "smollm_decode": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/smollm-135M-q4_0.gguf -r 1 -t 1 -ngl 0 -p 1 -n 32"},
            "stories_decode": {"bin": f"{self.install_dir}/bin/llama-bench", "args": f"-v -m {self.root}/models/stories15M-q4_0.gguf -r 1 -t 1 -ngl 0 -p 1 -n 32"},
            "stories_q4_0_bench": {"bin": f"{self.install_dir}/bin/llama-bench", 
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --repetitions 1 --no-warmup \
                                    --threads 1 --n-gpu-layers 0 --n-gen 0 --n-prompt 32"
                                },
            "stories_q4_0_bbench": {"bin": f"{self.install_dir}/bin/llama-batched-bench", 
                        "args": f"-m {self.root}/models/stories15M-q4_0.gguf \
                                    --no-op-offload --cpu-moe --batch-size 256 --ubatch-size 32 -npp 16 -ntg 4 -npl 0 \
                                    --threads 1 --threads-batch 1 --n-gpu-layers 0 --predict 1 \
                                    --kv-unified \
                                        "
                                    },
            "gemma3_bench_prefill_q8_0": {"bin": f"{self.install_dir}/bin/llama-bench", 
                        "args": f"-m {self.root}/models/gemma-3-270m-Q8_0.gguf \
                                    --repetitions 1 --no-warmup \
                                    --threads 1 --n-gpu-layers 0 --n-gen 0 --n-prompt 32"},
            "gemma3_bench_decode_q8_0": {"bin": f"{self.install_dir}/bin/llama-bench", 
                        "args": f"-m {self.root}/models/gemma-3-270m-Q8_0.gguf \
                                    --repetitions 1 --no-warmup \
                                    --threads 1 --n-gpu-layers 0 --n-gen 32 --n-prompt 1"},
            "tiny_moe_debug": {"bin": f"{self.install_dir}/bin/llama-cli", 
                                "args": f"-m /projects/sho_script/models/Tiny-Moe.mxfp4.gguf \
                                    --threads 1 \
                                    --file {PROMPTS_DIR}/hello-world.txt \
                                    --seed 1234 \
                                    --temp 0 \
                                    --top-k 1 \
                                    --top-p 1.0 \
                                    -n 10"},
            "test_q4_0_stories_moe_dbg": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/stories15M_MOE-q4_0.gguf \
                            --seed 42 \
                            -t 1 -ngl 0 -n 1 \
                            -no-cnv -st --no-warmup \
                            --file {PROMPTS_DIR}/hello-world.txt"},
            # GLM-4.6V-Flash model tests (multimodal vision-language model)
            "test_glm_4_6v_text": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                            --seed 42 \
                            -t 1 -ngl 0 -n 32 \
                            -no-cnv -st --no-warmup \
                            --file {PROMPTS_DIR}/hello-world.txt"},
            "test_glm_4_6v_multimodal": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                            --mmproj {self.root}/models/mmproj-GLM-4.6V-Flash-Q8_0.gguf \
                            --seed 42 \
                            -t 1 -ngl 0 -n 32 \
                            -no-cnv -st --no-warmup \
                            --file {PROMPTS_DIR}/hello-world.txt"},
            # Optimized GLM configurations for simulation (fewer tokens for faster sim)
            "test_glm_4_6v_text_fast": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                            --seed 42 \
                            -t 1 -ngl 0 -n 8 -c 512 \
                            -no-cnv -st --no-warmup \
                            --file {PROMPTS_DIR}/hello-world.txt"},
            "test_glm_4_6v_text_minimal": {"bin": f"{self.install_dir}/bin/llama-cli", 
                "args": f"-m {self.root}/models/GLM-4.6V-Flash-Q4_K_M.gguf \
                            --seed 42 \
                            -t 1 -ngl 0 -n 4 -c 256 \
                            -no-cnv -st --no-warmup \
                            --file {PROMPTS_DIR}/hello-world.txt"},
            }
        for qt in ["q8_0", "q4_0", "mxfp4", "mxfp8", "q2_k", "q6_k", "f32", "f16", "bf16"]:
            tests[f"optest_{qt}_all"] = {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": f"test -o MUL_MAT -b REF -p type_a={qt},type_b=f32"}
            tests[f"optest_{qt}_m16_n16"] = {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": f"test -o MUL_MAT -b REF -p type_a={qt},type_b=f32,m=16,n=16,k=256,bs=\[1,1\],nr=\[1,1\]"}
            tests[f"optest_{qt}_m16_n2"] =  {"bin": f"{self.install_dir}/bin/test-backend-ops", "args": f"test -o MUL_MAT -b REF -p type_a={qt},type_b=f32,m=16,n=2,k=256,bs=\[1,1\],nr=\[1,1\]"}

        return tests

@register_class("llama_x86")
class LlamaCppX86(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_x86",
            "linux-build-x86",
            "llamacpp-x86-install",
            use_qemu=False
        )

@register_class("llama_x86_bench")
class LlamaCppX86Bench(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_x86_bench",
            "linux-build-x86-bench",
            "llamacpp-x86-bench-install",
            use_qemu=False,
            build_ref=False
        )

@register_class("llama_imi")
class LlamaCppIMI(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_imi",
            "linux-build-imi",
            "llamacpp-imi-install",
            env=IMI_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True
    
    @property
    def is_imi(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT

@register_class("llama_imi_bench")
class LlamaCppIMIBench(LlamaCppBase):

    def __init__(self):
        super().__init__(
            "llama_imi_bench",
            "linux-build-imi-bench",
            "llamacpp-imi-bench-install",
            env=IMI_ENV,
            build_ref=False
        )

    @property
    def is_riscv(self) -> bool:
        return True
    
    @property
    def is_imi(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT


@register_class("llama_rvv")
class LlamaCppRVV(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_rvv",
            "linux-build-rvv",
            "llamacpp-rvv-install",
            env=RVV_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT

@register_class("llama_rvv_bench")
class LlamaCppRVVBench(LlamaCppBase):

    def __init__(self):
        super().__init__(
            "llama_rvv_bench",
            "linux-build-rvv-bench",
            "llamacpp-rvv-bench-install",
            env=RVV_ENV,
            build_ref=False
        )

    @property
    def is_riscv(self) -> bool:
        return True
    
    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT

@register_class("llama_amx_bench")
class LlamaCppAMXBench(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_amx_bench",
            "linux-build-amx-bench",
            "llamacpp-amx-bench-install",
            use_qemu=False,
            remote_info=AMX_REMOTE,
            build_ref=False,
            env=AMX_ENV,
            features = ["SSE42", "AVX","F16C", "AVX2", "BMI2", "FMA", "AVX512", "AVX512_VBMI","AVX512_VNNI", "AVX512_BF16", "AMX_TILE", "AMX_INT8", "AMX_BF16"]
        )

@register_class("llama_oryon_bench")
class LlamaCppOryonBench(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_oryon_bench",
            "linux-build-oryon-bench",
            "llamacpp-oryon-bench-install",
            use_qemu=False,
            remote_info=ORYON_REMOTE,
            build_ref=False,
            env=ORYON_ENV,
            features = []
        )

    @property
    def is_aarch64(self) -> bool:
        return True

@register_class("llama_oryon")
class LlamaCppOryonBench(LlamaCppBase):
    def __init__(self):
        super().__init__(
            "llama_oryon",
            "linux-build-oryon",
            "llamacpp-oryon-install",
            use_qemu=False,
            remote_info=ORYON_REMOTE,
            build_ref=True,
            env=ORYON_ENV,
            features = []
        )

    @property
    def is_aarch64(self) -> bool:
        return True