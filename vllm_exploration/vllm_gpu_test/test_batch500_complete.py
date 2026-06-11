"""
Complete 4-Way Performance Comparison: Batch Size 500
=====================================================
Tests:
  1. Transformers on CPU  (500 prompts)
  2. vLLM on CPU          (500 prompts)
  3. Transformers on GPU  (500 prompts)
  4. vLLM on GPU          (500 prompts)

With CPU core monitoring and full performance metrics.
"""

import time
import json
import os
import torch
import psutil

from gpu_env import apply_jetson_vllm_runtime_patches, vllm_llm_kwargs

apply_jetson_vllm_runtime_patches()
from threading import Thread
from datetime import datetime

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if "VLLM_ATTENTION_BACKEND" not in os.environ and torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" if major >= 10 else "TRITON_ATTN"
elif "VLLM_ATTENTION_BACKEND" not in os.environ:
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

MODEL_NAME = "facebook/opt-125m"
BATCH_SIZE = 500
MAX_NEW_TOKENS = 30
PROMPTS = [f"Question {i}: What is the meaning of" for i in range(BATCH_SIZE)]
CPU_THREADS = int(os.environ.get("CPU_THREADS", "0"))


def configure_cpu_threads():
    """Apply a fair CPU thread cap for both Transformers and vLLM."""
    if CPU_THREADS <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    # Keep PyTorch inter-op low for stable CPU baselines.
    try:
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class CPUMonitor:
    """Monitor CPU core utilization in a background thread."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self._running = False
        self._thread = None
        self.samples = []

    def start(self):
        self._running = True
        self.samples = []
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        return self._summarize()

    def _loop(self):
        proc = psutil.Process(os.getpid())
        while self._running:
            per_cpu = psutil.cpu_percent(interval=self.interval, percpu=True)
            self.samples.append({
                "per_cpu": per_cpu,
                "process_threads": proc.num_threads(),
                "process_mem_mb": proc.memory_info().rss / 1024 / 1024,
            })

    def _summarize(self):
        if not self.samples:
            return None
        n_cores = len(self.samples[0]["per_cpu"])
        avg_per_core = [0.0] * n_cores
        for s in self.samples:
            for i, v in enumerate(s["per_cpu"]):
                avg_per_core[i] += v
        avg_per_core = [v / len(self.samples) for v in avg_per_core]

        active = sum(1 for v in avg_per_core if v > 5.0)
        avg_total = sum(avg_per_core) / n_cores
        max_threads = max(s["process_threads"] for s in self.samples)
        avg_mem = sum(s["process_mem_mb"] for s in self.samples) / len(self.samples)

        top10 = sorted(enumerate(avg_per_core), key=lambda x: x[1], reverse=True)[:10]

        return {
            "logical_cpus": n_cores,
            "active_threads": active,
            "avg_cpu_pct": round(avg_total, 1),
            "max_process_threads": max_threads,
            "avg_mem_mb": round(avg_mem, 0),
            "top10": [(cid, round(u, 1)) for cid, u in top10],
        }


def print_header(title):
    print(f"\n{'='*80}")
    print(title)
    print("=" * 80)


def print_results(label, elapsed, batch, extra=None):
    tpp = elapsed / batch * 1000
    thr = batch / elapsed
    print(f"\n  {label}")
    print(f"  Total time:      {elapsed:.2f}s")
    print(f"  Per prompt:      {tpp:.2f}ms")
    print(f"  Throughput:      {thr:.2f} prompts/sec")
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
    return {"time_s": round(elapsed, 2), "per_prompt_ms": round(tpp, 2),
            "throughput": round(thr, 2), **(extra or {})}


# ── TEST 1: Transformers on CPU ─────────────────────────────────────────────

def test1_transformers_cpu():
    print_header("TEST 1: Transformers on CPU  (500 prompts)")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    configure_cpu_threads()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Model loaded on CPU")

    mon = CPUMonitor()
    mon.start()

    t0 = time.time()
    inputs = tok(PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                        temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    elapsed = time.time() - t0

    cpu_stats = mon.stop()
    extra = {}
    if cpu_stats:
        extra["Active threads (>5%)"] = f"{cpu_stats['active_threads']}/{cpu_stats['logical_cpus']}"
        extra["Avg CPU util"] = f"{cpu_stats['avg_cpu_pct']}%"
        extra["Max proc threads"] = cpu_stats["max_process_threads"]
        extra["Avg memory"] = f"{cpu_stats['avg_mem_mb']:.0f} MB"

    res = print_results("Transformers on CPU", elapsed, BATCH_SIZE, extra)
    res["cpu_stats"] = cpu_stats
    return res


# ── TEST 2: vLLM on CPU ────────────────────────────────────────────────────

def test2_vllm_cpu():
    print_header("TEST 2: vLLM on CPU  (500 prompts)")
    import subprocess

    # vLLM CPU requires a separate CPU-specific build.
    # Resolve interpreter from env var first, then common local paths.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpu_python_candidates = [
        os.environ.get("VLLM_CPU_PYTHON", ""),
        os.path.join(project_root, "vllm_test", "venv", "bin", "python3"),
        os.path.join(project_root, "vllm_cpu_venv", "bin", "python3"),
    ]
    cpu_python = next((p for p in cpu_python_candidates if p and os.path.exists(p)), None)
    test_script = os.path.join(project_root, "vllm_cpu_test", "test2_vllm_cpu.py")

    if cpu_python is None:
        print("  ❌ vLLM CPU interpreter not found.")
        print("  Set VLLM_CPU_PYTHON to a valid interpreter, e.g.")
        print("  export VLLM_CPU_PYTHON=~/projects/vllm_exploration/vllm_cpu_venv/bin/python3")
        return {"status": "FAILED", "reason": "vLLM CPU interpreter not found"}

    print(f"  Using CPU-build vLLM interpreter: {cpu_python}")
    print(f"  Running subprocess: {cpu_python} {test_script}")

    try:
        env = os.environ.copy()
        # Avoid forwarding helper vars that vLLM doesn't recognize.
        env.pop("VLLM_CPU_PYTHON", None)
        env.pop("VLLM_ATTENTION_BACKEND", None)
        if CPU_THREADS > 0:
            env["OMP_NUM_THREADS"] = str(CPU_THREADS)
            env["MKL_NUM_THREADS"] = str(CPU_THREADS)
            env["OPENBLAS_NUM_THREADS"] = str(CPU_THREADS)
            env["NUMEXPR_NUM_THREADS"] = str(CPU_THREADS)
            env["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

        proc = subprocess.run(
            [cpu_python, test_script],
            capture_output=True, text=True, timeout=1800, env=env,
            cwd=os.path.dirname(test_script),
        )
        # Print all output
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT_JSON__:"):
                result = json.loads(line.split(":", 1)[1])
            else:
                print(f"  {line}")

        if proc.returncode != 0:
            print(f"\n  stderr: {proc.stderr[-500:]}")
            return {"status": "FAILED", "reason": f"exit code {proc.returncode}"}

        # Format the result for the summary
        extra = {}
        cpu_stats = result.get("cpu_stats")
        if cpu_stats:
            extra["Active threads (>5%)"] = f"{cpu_stats['active_threads']}/{cpu_stats['logical_cpus']}"
            extra["Avg CPU util"] = f"{cpu_stats['avg_cpu_pct']}%"
            extra["Max proc threads"] = cpu_stats["max_process_threads"]
            extra["Avg memory"] = f"{cpu_stats['avg_mem_mb']:.0f} MB"
        if result.get("init_time_s"):
            extra["Init time"] = f"{result['init_time_s']:.2f}s"

        res = print_results("vLLM on CPU", result["time_s"], BATCH_SIZE, extra)
        res["cpu_stats"] = cpu_stats
        res["init_time_s"] = result.get("init_time_s")
        return res

    except subprocess.TimeoutExpired:
        print(f"\n  ❌ vLLM CPU test timed out (>1800s)")
        return {"status": "FAILED", "reason": "timeout"}
    except Exception as e:
        print(f"\n  ❌ vLLM on CPU FAILED: {e}")
        return {"status": "FAILED", "reason": str(e)}


# ── TEST 3: Transformers on GPU ─────────────────────────────────────────────

def test3_transformers_gpu():
    print_header("TEST 3: Transformers on GPU  (500 prompts)")
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA is not available on this machine.")
        return {"status": "SKIPPED", "reason": "CUDA not available"}
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    device = "cuda:0"
    model = model.to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    chunk = 50
    for i in range(0, BATCH_SIZE, chunk):
        batch = PROMPTS[i:i+chunk]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=64).to(device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                            temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
        torch.cuda.empty_cache()
    elapsed = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    res = print_results("Transformers on GPU", elapsed, BATCH_SIZE,
                        {"Peak GPU mem": f"{peak:.2f} GB"})
    res["peak_gpu_gb"] = round(peak, 2)
    return res


# ── TEST 4: vLLM on GPU ────────────────────────────────────────────────────

def test4_vllm_gpu():
    print_header("TEST 4: vLLM on GPU  (500 prompts)")
    if not torch.cuda.is_available():
        print("  ⚠️  SKIPPED: CUDA is not available on this machine.")
        return {"status": "SKIPPED", "reason": "CUDA not available"}
    from vllm import LLM, SamplingParams

    print(f"  Initializing vLLM on GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.reset_peak_memory_stats()

    init_t0 = time.time()
    kw = vllm_llm_kwargs(tensor_parallel_size=1, max_num_seqs=BATCH_SIZE)
    kw.update(
        model=MODEL_NAME,
        dtype="float16",
        enforce_eager=True,
    )
    llm = LLM(**kw)
    init_time = time.time() - init_t0
    print(f"  vLLM init: {init_time:.2f}s")

    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_NEW_TOKENS)

    t0 = time.time()
    llm.generate(PROMPTS, sp)
    elapsed = time.time() - t0

    peak = torch.cuda.max_memory_allocated() / 1024**3
    res = print_results("vLLM on GPU", elapsed, BATCH_SIZE,
                        {"Init time": f"{init_time:.2f}s", "Peak GPU mem": f"{peak:.2f} GB"})
    res["init_time_s"] = round(init_time, 2)
    res["peak_gpu_gb"] = round(peak, 2)
    return res


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print("COMPLETE 4-WAY PERFORMANCE COMPARISON")
    print(f"Date:       {ts}")
    print(f"Model:      {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE}")
    if CPU_THREADS > 0:
        print(f"CPU thread cap: {CPU_THREADS}")
    print(f"CPU:        {psutil.cpu_count()} logical threads "
          f"({psutil.cpu_count(logical=False)} physical cores)")
    print(f"RAM:        {psutil.virtual_memory().total / 1024**3:.1f} GB")
    if torch.cuda.is_available():
        print(f"GPU:        {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    print("=" * 80)

    results = {}

    # Test 1
    try:
        results["test1"] = test1_transformers_cpu()
    except Exception as e:
        print(f"  ❌ TEST 1 FAILED: {e}")
        results["test1"] = {"status": "FAILED", "reason": str(e)}

    # Test 2
    try:
        results["test2"] = test2_vllm_cpu()
    except Exception as e:
        print(f"  ❌ TEST 2 FAILED: {e}")
        results["test2"] = {"status": "FAILED", "reason": str(e)}

    # Test 3
    try:
        results["test3"] = test3_transformers_gpu()
    except Exception as e:
        print(f"  ❌ TEST 3 FAILED: {e}")
        results["test3"] = {"status": "FAILED", "reason": str(e)}

    # Test 4
    try:
        results["test4"] = test4_vllm_gpu()
    except Exception as e:
        print(f"  ❌ TEST 4 FAILED: {e}")
        results["test4"] = {"status": "FAILED", "reason": str(e)}

    # ── Summary ─────────────────────────────────────────────────────────────
    print_header("FINAL COMPARISON")

    fmt = "  {:<28s} {:>10s} {:>14s} {:>12s}"
    print(fmt.format("Test", "Time", "Throughput", "GPU Mem"))
    print("  " + "-" * 66)

    labels = [
        ("1. Transformers CPU", "test1"),
        ("2. vLLM CPU", "test2"),
        ("3. Transformers GPU", "test3"),
        ("4. vLLM GPU", "test4"),
    ]
    for label, key in labels:
        r = results.get(key, {})
        if r.get("status") == "FAILED":
            print(fmt.format(label, "FAILED", "-", "-"))
        elif r.get("status") == "SKIPPED":
            print(fmt.format(label, "SKIPPED", "-", "-"))
        elif "time_s" in r:
            t = f"{r['time_s']:.2f}s"
            tp = f"{r['throughput']:.1f} p/s"
            gm = f"{r.get('peak_gpu_gb', '-')}" if isinstance(r.get('peak_gpu_gb'), float) else "-"
            if gm != "-":
                gm += " GB"
            print(fmt.format(label, t, tp, gm))
        else:
            print(fmt.format(label, "N/A", "N/A", "N/A"))

    # speedups
    t1 = results.get("test1", {}).get("time_s")
    t2 = results.get("test2", {}).get("time_s")
    t3 = results.get("test3", {}).get("time_s")
    t4 = results.get("test4", {}).get("time_s")

    print()
    if t1 and t2:
        print(f"  CPU vLLM vs CPU Transformers:         {t1/t2:.1f}x faster")
    if t1 and t3:
        print(f"  GPU Transformers vs CPU Transformers: {t1/t3:.1f}x faster")
    if t1 and t4:
        print(f"  GPU vLLM vs CPU Transformers:         {t1/t4:.1f}x faster")
    if t3 and t4:
        print(f"  GPU vLLM vs GPU Transformers:         {t3/t4:.1f}x faster")
    if t2 and t4:
        print(f"  GPU vLLM vs CPU vLLM:                 {t2/t4:.1f}x faster")
    print("=" * 80)

    # save JSON for report generation
    results["_meta"] = {
        "date": ts, "model": MODEL_NAME, "batch_size": BATCH_SIZE,
        "cpu": f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical",
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
    with open("batch500_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to batch500_results.json")


if __name__ == "__main__":
    main()
