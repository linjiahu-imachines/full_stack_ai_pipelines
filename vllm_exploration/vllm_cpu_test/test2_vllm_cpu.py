"""
Test 2: vLLM on CPU (500 prompts)
Run with the CPU-build venv: vllm_test/venv/bin/python3
Outputs JSON result to stdout.

IMPORTANT: Do NOT set VLLM_CPU_KVCACHE_SPACE or VLLM_CPU_OMP_THREADS_BIND;
           let vLLM auto-configure for best results.
"""
import time
import json
import os
import psutil
from threading import Thread

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "facebook/opt-125m"
BATCH_SIZE = 500
MAX_NEW_TOKENS = 30
PROMPTS = [f"Question {i}: What is the meaning of" for i in range(BATCH_SIZE)]


class CPUMonitor:
    def __init__(self, interval=0.5):
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
            self._thread.join(timeout=5)
        return self._summarize()

    def _loop(self):
        while self._running:
            per_cpu = psutil.cpu_percent(interval=self.interval, percpu=True)
            try:
                proc = psutil.Process(os.getpid())
                threads = proc.num_threads()
                mem = proc.memory_info().rss / 1024 / 1024
            except Exception:
                threads = 0
                mem = 0
            self.samples.append({
                "per_cpu": per_cpu,
                "process_threads": threads,
                "process_mem_mb": mem,
            })

    def _summarize(self):
        if not self.samples:
            return None
        n = len(self.samples[0]["per_cpu"])
        avg = [0.0] * n
        for s in self.samples:
            for i, v in enumerate(s["per_cpu"]):
                avg[i] += v
        avg = [v / len(self.samples) for v in avg]
        active = sum(1 for v in avg if v > 5.0)
        top10 = sorted(enumerate(avg), key=lambda x: x[1], reverse=True)[:10]
        return {
            "logical_cpus": n,
            "active_threads": active,
            "avg_cpu_pct": round(sum(avg) / n, 1),
            "max_process_threads": max(s["process_threads"] for s in self.samples),
            "avg_mem_mb": round(sum(s["process_mem_mb"] for s in self.samples) / len(self.samples), 0),
            "top10": [(cid, round(u, 1)) for cid, u in top10],
        }


def main():
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    print(f"vLLM platform: {current_platform.device_type}", flush=True)
    print(f"Initializing vLLM CPU with model {MODEL_NAME}...", flush=True)

    mon = CPUMonitor()
    mon.start()

    init_t0 = time.time()
    llm = LLM(
        model=MODEL_NAME,
        max_num_seqs=5,
        enforce_eager=True,
    )
    init_time = time.time() - init_t0
    print(f"vLLM CPU init: {init_time:.2f}s", flush=True)

    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=MAX_NEW_TOKENS)

    print(f"Generating {BATCH_SIZE} prompts...", flush=True)
    t0 = time.time()
    outputs = llm.generate(PROMPTS, sp)
    elapsed = time.time() - t0

    cpu_stats = mon.stop()

    result = {
        "time_s": round(elapsed, 2),
        "per_prompt_ms": round(elapsed / BATCH_SIZE * 1000, 2),
        "throughput": round(BATCH_SIZE / elapsed, 2),
        "init_time_s": round(init_time, 2),
        "cpu_stats": cpu_stats,
    }

    print(f"__RESULT_JSON__:{json.dumps(result, default=str)}", flush=True)

    print(f"\nvLLM on CPU Results:", flush=True)
    print(f"  Init time:   {init_time:.2f}s", flush=True)
    print(f"  Total time:  {elapsed:.2f}s", flush=True)
    print(f"  Per prompt:  {elapsed/BATCH_SIZE*1000:.2f}ms", flush=True)
    print(f"  Throughput:  {BATCH_SIZE/elapsed:.2f} prompts/sec", flush=True)
    if cpu_stats:
        print(f"  Active threads: {cpu_stats['active_threads']}/{cpu_stats['logical_cpus']}", flush=True)
        print(f"  Avg CPU util:   {cpu_stats['avg_cpu_pct']}%", flush=True)
        print(f"  Max threads:    {cpu_stats['max_process_threads']}", flush=True)
        print(f"  Avg memory:     {cpu_stats['avg_mem_mb']:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
