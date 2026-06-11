# Higher-Level Frameworks Integration Guide

**Date:** January 6, 2025  
**Purpose:** Guide for integrating higher-level ML/LLM frameworks (Ray Serve, LangChain, etc.) with llama.cpp in QEMU RISC-V system mode.

---

## Table of Contents

1. [Overview and Architecture](#1-overview-and-architecture)
2. [Architecture Options](#2-architecture-options)
3. [Python Bindings for llama.cpp](#3-python-bindings-for-llamacpp)
4. [Ray Serve Integration](#4-ray-serve-integration)
5. [LangChain Integration](#5-langchain-integration)
6. [Real-World Inference Patterns](#6-real-world-inference-patterns)
7. [Networking and API Setup](#7-networking-and-api-setup)
8. [Performance Considerations](#8-performance-considerations)
9. [Recommended Approach](#9-recommended-approach)
10. [Next Steps and Examples](#10-next-steps-and-examples)

---

## 1. Overview and Architecture

### Current State

You have a working setup with:
- **QEMU RISC-V system mode** (`qemu-system-riscv64`) running Linux guest
- **llama.cpp** compiled for RISC-V with IMI extensions
- **9p filesystem sharing** for seamless file access between host and guest
- **Multi-core support** (`-smp cpus=2` or more)
- **llama-cli binary** running inference in the guest

### Goal

Extend this setup to support:
- **Ray Serve**: Distributed model serving and deployment
- **LangChain**: LLM application framework for building complex workflows
- **Real inference APIs**: HTTP/gRPC endpoints, request handling, batching
- **Production-like patterns**: Load balancing, monitoring, scaling

### Key Challenges

1. **Python Ecosystem**: Ray Serve and LangChain are Python-based; RISC-V Linux guest may have limited Python packages
2. **Networking**: APIs need network connectivity (host-guest or external)
3. **Cross-Architecture**: Python packages compiled for x86_64 won't run in RISC-V guest
4. **Performance**: Additional framework overhead vs. direct llama-cli execution
5. **Deployment Complexity**: More moving parts, harder to debug

---

## 2. Architecture Options

You have two main architectural approaches:

### Option A: Host-Based Orchestration (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST (x86_64 Linux)                      │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  Ray Serve      │  │  LangChain      │                  │
│  │  (Python)       │  │  (Python)       │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                     │                            │
│           │ HTTP/gRPC           │                            │
│           │                     │                            │
│  ┌────────▼─────────────────────▼────────┐                  │
│  │  API Gateway / Load Balancer          │                  │
│  └────────┬──────────────────────────────┘                  │
│           │                                                 │
│           │ Process Management                              │
│           │ (QEMU + llama-cli via subprocess)               │
│  ┌────────▼──────────────────────────────┐                  │
│  │  QEMU System Mode (Guest)             │                  │
│  │  ┌─────────────────────────────────┐  │                  │
│  │  │  RISC-V Linux Guest OS          │  │                  │
│  │  │  ┌───────────────────────────┐  │  │                  │
│  │  │  │  llama-cli (RISC-V)       │  │  │                  │
│  │  │  │  (via 9p filesystem)      │  │  │                  │
│  │  │  └───────────────────────────┘  │  │                  │
│  │  └─────────────────────────────────┘  │                  │
│  └────────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Full Python ecosystem available (Ray Serve, LangChain, etc.)
- ✅ No cross-compilation needed for Python packages
- ✅ Easier debugging and development
- ✅ Standard networking (localhost, ports, etc.)
- ✅ Can use existing Python tooling and libraries
- ✅ Better performance monitoring and observability

**Disadvantages:**
- ❌ Requires process management (start/stop QEMU instances)
- ❌ More complex request routing
- ❌ Need IPC between host Python and guest llama-cli

**Use Case:** Most practical for development, testing, and production-like deployments.

---

### Option B: Guest-Based Execution

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST (x86_64 Linux)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  QEMU System Mode (Guest)                            │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  RISC-V Linux Guest OS                         │  │  │
│  │  │                                                 │  │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │  │  │
│  │  │  │ Ray Serve│  │ LangChain│  │ llama-cpp│    │  │  │
│  │  │  │ (RISC-V) │  │ (RISC-V) │  │ Python   │    │  │  │
│  │  │  │          │  │          │  │ Bindings │    │  │  │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘    │  │  │
│  │  │       │              │              │          │  │  │
│  │  │       └──────────────┴──────────────┘          │  │  │
│  │  │                    │                           │  │  │
│  │  │            llama.cpp (RISC-V)                  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Networking: Host port forwarding to guest                  │
└─────────────────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Everything runs in guest (more realistic)
- ✅ Simpler architecture (single execution environment)
- ✅ Better for testing guest OS behavior

**Disadvantages:**
- ❌ Limited Python packages for RISC-V
- ❌ Need to cross-compile Python extensions
- ❌ Harder to debug (guest OS constraints)
- ❌ Slower development cycle
- ❌ May not have all required dependencies

**Use Case:** Only if you specifically need to test guest OS behavior or have a full RISC-V Python ecosystem.

---

## 3. Python Bindings for llama.cpp

### 3.1 llama-cpp-python

The most popular Python bindings for llama.cpp:

**Repository:** https://github.com/abetlen/llama-cpp-python

**Features:**
- ✅ Full Python API for llama.cpp
- ✅ Supports GGUF models
- ✅ Async/await support
- ✅ LangChain integration built-in
- ✅ Multiple backends (CPU, CUDA, Metal, etc.)

### 3.2 Integration Approaches

#### Approach 1: Use llama-cli via Subprocess (Simplest)

**No Python bindings needed** - just call `llama-cli` as a subprocess:

```python
import subprocess
import json
from pathlib import Path

class LlamaCppWrapper:
    def __init__(self, llama_cli_path: Path, model_path: Path):
        self.llama_cli_path = llama_cli_path
        self.model_path = model_path
    
    def generate(self, prompt: str, max_tokens: int = 128, threads: int = 2) -> str:
        """Generate text using llama-cli via subprocess."""
        # Write prompt to temp file
        prompt_file = Path("/tmp/prompt.txt")
        prompt_file.write_text(prompt)
        
        # Run llama-cli
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-t", str(threads),
            "-n", str(max_tokens),
            "--file", str(prompt_file),
            "--no-warmup",
            "-ngl", "0"  # CPU only
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output (llama-cli prints to stdout)
        return result.stdout

# Usage
wrapper = LlamaCppWrapper(
    llama_cli_path=Path("/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli"),
    model_path=Path("/mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf")
)

response = wrapper.generate("Hello, how are you?", max_tokens=64)
```

**Pros:**
- ✅ No compilation needed
- ✅ Works with existing llama-cli binary
- ✅ Simple to implement

**Cons:**
- ❌ Slower (process overhead)
- ❌ No streaming support
- ❌ Harder to manage state (context, KV cache)
- ❌ Limited control over inference parameters

---

#### Approach 2: Build llama-cpp-python for RISC-V (Complex)

**If you need native Python bindings in guest:**

1. **Cross-compile Python C extension:**
   ```bash
   # This is non-trivial and may require:
   # - RISC-V Python toolchain
   # - Cross-compiled NumPy, etc.
   # - Patched llama-cpp-python build system
   ```

2. **Better alternative:** Use host Python with llama-cpp-python, call QEMU-guest llama-cli

**Not recommended** unless you have specific requirements.

---

#### Approach 3: Host Python + Guest llama-cli (Recommended)

**Use Python on host, orchestrate QEMU + llama-cli:**

```python
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional

class QEMULlamaCppService:
    """Service that runs llama-cli in QEMU guest via 9p filesystem."""
    
    def __init__(
        self,
        qemu_bin: Path,
        kernel: Path,
        rootfs: Path,
        shared_path: Path,
        llama_cli_path: Path,
        model_path: Path
    ):
        self.qemu_bin = qemu_bin
        self.kernel = kernel
        self.rootfs = rootfs
        self.shared_path = shared_path
        self.llama_cli_path = llama_cli_path  # Path in guest
        self.model_path = model_path  # Path in guest
        
        # Paths in host (via 9p)
        self.host_llama_cli = shared_path / llama_cli_path.relative_to("/mnt/shared")
        self.host_model = shared_path / model_path.relative_to("/mnt/shared")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        threads: int = 2,
        temperature: float = 0.8
    ) -> str:
        """Generate text using llama-cli in QEMU guest."""
        
        # Write prompt to shared directory
        prompt_file = self.shared_path / "tmp" / "prompt.txt"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(prompt)
        
        # Guest path
        guest_prompt_file = f"/mnt/shared/tmp/prompt.txt"
        guest_output_file = f"/mnt/shared/tmp/output_{uuid.uuid4()}.txt"
        
        # Create QEMU command script (to run in guest)
        qemu_script = self._create_qemu_script(guest_prompt_file, guest_output_file, max_tokens, threads)
        
        # Run QEMU with script (one-shot execution)
        # This is complex - you'd need to:
        # 1. Boot QEMU
        # 2. Mount 9p
        # 3. Run llama-cli
        # 4. Capture output
        # 5. Shutdown QEMU
        # 
        # For production, better to keep QEMU running and use IPC
        
        # Simplified: Assume QEMU is already running and you can SSH/execute in guest
        # For now, use subprocess approach (see Approach 1)
        pass

# This is complex - see "Recommended Approach" section
```

---

## 4. Ray Serve Integration

### 4.1 Overview

Ray Serve is a framework for building and deploying scalable ML models. It provides:
- **Model deployment**: Serve models as HTTP/gRPC endpoints
- **Scaling**: Automatic scaling based on load
- **Batching**: Request batching for efficiency
- **Routing**: Load balancing and routing

### 4.2 Architecture with QEMU

**Recommended: Host-based Ray Serve, Guest-based Inference**

```
┌─────────────────────────────────────────────────────────────┐
│                    HOST (x86_64)                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Ray Serve                                           │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  @serve.deployment                             │  │  │
│  │  │  class LlamaCppDeployment:                     │  │  │
│  │  │      def __init__(self):                       │  │  │
│  │  │          self.qemu_runner = QEMULlamaRunner()  │  │  │
│  │  │      async def __call__(self, prompt):         │  │  │
│  │  │          return self.qemu_runner.generate(...) │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  QEMU Process Manager                                │  │
│  │  (Manages QEMU instances, llama-cli execution)      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation Example

```python
from ray import serve
from ray.serve import Application
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any

# Simple wrapper (Approach 1: subprocess)
class LlamaCppRunner:
    def __init__(self, llama_cli_path: Path, model_path: Path):
        self.llama_cli_path = llama_cli_path
        self.model_path = model_path
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text asynchronously."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_sync,
            prompt,
            **kwargs
        )
    
    def _generate_sync(self, prompt: str, max_tokens: int = 128, threads: int = 2) -> str:
        """Synchronous generation (runs in executor)."""
        prompt_file = Path("/tmp/prompt.txt")
        prompt_file.write_text(prompt)
        
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-t", str(threads),
            "-n", str(max_tokens),
            "--file", str(prompt_file),
            "--no-warmup",
            "-ngl", "0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout

# Ray Serve deployment
@serve.deployment(
    num_replicas=2,  # Run 2 replicas
    ray_actor_options={"num_cpus": 2}
)
class LlamaCppDeployment:
    def __init__(self):
        llama_cli = Path("/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli")
        model = Path("/mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf")
        self.runner = LlamaCppRunner(llama_cli, model)
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 128)
        threads = request.get("threads", 2)
        
        response = await self.runner.generate(prompt, max_tokens=max_tokens, threads=threads)
        
        return {
            "response": response,
            "prompt": prompt
        }

# Deploy
app = LlamaCppDeployment.bind()
```

**Run Ray Serve:**
```bash
# Start Ray
ray start --head

# Deploy
serve run app:app
```

**Test:**
```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 64}'
```

---

## 5. LangChain Integration

### 5.1 Overview

LangChain is a framework for building LLM applications with:
- **Chains**: Composable workflows
- **Agents**: Autonomous decision-making
- **Memory**: Conversation memory
- **Tools**: External integrations

### 5.2 Integration Approaches

#### Approach 1: Custom LLM Wrapper

**Create a LangChain-compatible wrapper:**

```python
from langchain.llms.base import LLM
from typing import Optional, List
from pathlib import Path
import subprocess

class LlamaCppLLM(LLM):
    """LangChain wrapper for llama-cli."""
    
    llama_cli_path: Path
    model_path: Path
    threads: int = 2
    max_tokens: int = 128
    
    @property
    def _llm_type(self) -> str:
        return "llama_cpp_cli"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text from prompt."""
        prompt_file = Path("/tmp/prompt.txt")
        prompt_file.write_text(prompt)
        
        cmd = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-t", str(self.threads),
            "-n", str(self.max_tokens),
            "--file", str(prompt_file),
            "--no-warmup",
            "-ngl", "0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    
    @property
    def _identifying_params(self) -> dict:
        return {
            "llama_cli_path": str(self.llama_cli_path),
            "model_path": str(self.model_path),
            "threads": self.threads
        }

# Usage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = LlamaCppLLM(
    llama_cli_path=Path("/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli"),
    model_path=Path("/mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf"),
    threads=2,
    max_tokens=128
)

# Create a chain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short story about {topic}:"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("a robot learning to paint")
print(result)
```

#### Approach 2: Use llama-cpp-python (if available)

**If you can use llama-cpp-python on host:**

```python
from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="/path/to/model.gguf",
    n_threads=2,
    n_ctx=4096,
    verbose=True
)

chain = LLMChain(llm=llm, prompt=prompt)
```

---

## 6. Real-World Inference Patterns

### 6.1 HTTP API Server (FastAPI)

**Simple REST API:**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import subprocess

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    threads: int = 2
    temperature: float = 0.8

class GenerateResponse(BaseModel):
    response: str
    prompt: str
    tokens_generated: int

class LlamaCppService:
    def __init__(self):
        self.llama_cli = Path("/mnt/shared/dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli")
        self.model = Path("/mnt/shared/dev_env/llama.cpp/models/stories15M-q4_0.gguf")
    
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        prompt_file = Path("/tmp/prompt.txt")
        prompt_file.write_text(request.prompt)
        
        cmd = [
            str(self.llama_cli),
            "-m", str(self.model),
            "-t", str(request.threads),
            "-n", str(request.max_tokens),
            "--file", str(prompt_file),
            "--no-warmup",
            "-ngl", "0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        response_text = result.stdout.strip()
        
        return GenerateResponse(
            response=response_text,
            prompt=request.prompt,
            tokens_generated=len(response_text.split())
        )

service = LlamaCppService()

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    return service.generate(request)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Run:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Test:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 64}'
```

---

### 6.2 Streaming Responses

**For real-time token streaming (requires more complex integration):**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import subprocess
from pathlib import Path

app = FastAPI()

def generate_stream(prompt: str, max_tokens: int = 128):
    """Generate tokens and stream them."""
    # llama-cli doesn't support streaming output easily
    # You'd need to:
    # 1. Parse llama-cli output in real-time, OR
    # 2. Use llama-cpp-python (if available), OR
    # 3. Modify llama-cli to support streaming
    
    # Simplified: yield full response
    prompt_file = Path("/tmp/prompt.txt")
    prompt_file.write_text(prompt)
    
    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL),
        "-n", str(max_tokens),
        "--file", str(prompt_file),
        "-ngl", "0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    yield result.stdout

@app.post("/generate/stream")
async def generate_stream_endpoint(prompt: str, max_tokens: int = 128):
    return StreamingResponse(
        generate_stream(prompt, max_tokens),
        media_type="text/plain"
    )
```

---

### 6.3 Batch Processing

**Process multiple prompts in batch:**

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List

class BatchProcessor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.service = LlamaCppService()
    
    def process_batch(self, prompts: List[str]) -> List[str]:
        """Process prompts in parallel."""
        futures = [
            self.executor.submit(self.service.generate, prompt)
            for prompt in prompts
        ]
        return [future.result() for future in futures]

# Usage
processor = BatchProcessor(max_workers=4)
prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]
results = processor.process_batch(prompts)
```

---

## 7. Networking and API Setup

### 7.1 QEMU Networking Options

#### Option 1: User Mode Networking (Simplest)

**QEMU provides NAT networking:**

```bash
qemu-system-riscv64 \
  -netdev user,id=net0,hostfwd=tcp::8080-:8000 \
  -device virtio-net-device,netdev=net0 \
  ...
```

**Guest can access host on `10.0.2.2`:**

```bash
# In guest
curl http://10.0.2.2:8000/health
```

**Host can access guest on `localhost:8080` (forwarded to guest:8000):**

```bash
# On host
curl http://localhost:8080/health
```

---

#### Option 2: TAP Networking (More Complex)

**Requires root/sudo, provides bridge networking:**

```bash
# Create TAP interface
sudo ip tuntap add mode tap name tap0
sudo ip link set tap0 up
sudo ip addr add 192.168.100.1/24 dev tap0

# QEMU command
qemu-system-riscv64 \
  -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
  -device virtio-net-device,netdev=net0 \
  ...
```

---

### 7.2 API Gateway Pattern

**For production-like setup:**

```
┌─────────────────────────────────────────────────────────────┐
│                    External Clients                         │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/gRPC
┌───────────────────────▼─────────────────────────────────────┐
│  API Gateway (Nginx / Traefik / Envoy)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Load Balancing
┌───────────────────────▼─────────────────────────────────────┐
│  Ray Serve / FastAPI (Multiple Instances)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Process Management
┌───────────────────────▼─────────────────────────────────────┐
│  QEMU + llama-cli (Multiple Instances)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Performance Considerations

### 8.1 Overhead Analysis

| Component | Overhead | Notes |
|-----------|----------|-------|
| Direct `llama-cli` | Baseline | Fastest |
| Python subprocess wrapper | +5-10% | Process overhead |
| FastAPI HTTP server | +10-20% | HTTP parsing, JSON |
| Ray Serve | +15-25% | Additional routing, serialization |
| LangChain | +20-30% | Framework overhead |

### 8.2 Optimization Strategies

1. **Keep QEMU instances warm:**
   - Don't start/stop QEMU for each request
   - Use connection pooling
   - Reuse llama-cli processes (if possible)

2. **Batch requests:**
   - Process multiple prompts together
   - Use async/await for concurrency

3. **Caching:**
   - Cache common prompts/responses
   - Use KV cache effectively (already in llama.cpp)

4. **Resource allocation:**
   - Match `-smp cpus=N` to `-t N` threads
   - Allocate enough RAM (4GB+ recommended)

---

## 9. Recommended Approach

### 9.1 For Development and Testing

**Use Host-Based Orchestration with Simple Wrapper:**

1. **Python on host** (full ecosystem available)
2. **Simple subprocess wrapper** for llama-cli (Approach 1)
3. **FastAPI for HTTP API** (simple, fast)
4. **QEMU via 9p filesystem** (no sudo needed)

**Example structure:**
```
/home/linhu/repo/iminn-tools/
├── src/iminnt/
│   ├── llamacpp_api.py      # FastAPI wrapper
│   └── llamacpp_service.py  # Service class
├── scripts/
│   └── run_api_server.sh    # Start API server
└── examples/
    └── test_api.py          # Test scripts
```

---

### 9.2 For Production-Like Deployment

**Use Ray Serve with QEMU Process Manager:**

1. **Ray Serve on host** (scaling, load balancing)
2. **QEMU process manager** (manages QEMU instances)
3. **9p filesystem** (shared models/binaries)
4. **Monitoring and logging** (Ray dashboard, Prometheus)

**Example structure:**
```
/home/linhu/repo/iminn-tools/
├── src/iminnt/
│   ├── ray_serve/
│   │   ├── deployment.py     # Ray Serve deployment
│   │   └── qemu_manager.py  # QEMU instance manager
│   └── ...
└── scripts/
    └── deploy_ray_serve.sh  # Deployment script
```

---

### 9.3 For LangChain Integration

**Use Custom LLM Wrapper:**

1. **Custom `LlamaCppLLM` class** (LangChain-compatible)
2. **Host-based execution** (Python ecosystem)
3. **Subprocess wrapper** (simplest approach)

**Example:**
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from iminnt.llamacpp_langchain import LlamaCppLLM

llm = LlamaCppLLM(...)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("your prompt")
```

---

## 10. Next Steps and Examples

### 10.1 Immediate Next Steps

1. **Create simple Python wrapper:**
   - Implement `LlamaCppService` class
   - Test with existing llama-cli binary
   - Add error handling and logging

2. **Set up FastAPI server:**
   - Create `/generate` endpoint
   - Add health check
   - Test with curl/Postman

3. **Integrate with existing codebase:**
   - Add to `src/iminnt/` structure
   - Create CLI commands (`iminnt api serve`)
   - Document in README

4. **Add examples:**
   - Basic inference example
   - Batch processing example
   - LangChain integration example

---

### 10.2 Example Implementation Plan

**Phase 1: Basic API (Week 1)**
- [ ] Create `LlamaCppService` class
- [ ] Implement FastAPI server
- [ ] Add basic `/generate` endpoint
- [ ] Test with existing models

**Phase 2: Integration (Week 2)**
- [ ] Integrate with `iminnt` CLI
- [ ] Add configuration management
- [ ] Add logging and monitoring
- [ ] Document API endpoints

**Phase 3: Advanced Features (Week 3-4)**
- [ ] Add batch processing
- [ ] Implement caching
- [ ] Add streaming support (if possible)
- [ ] Performance optimization

**Phase 4: Framework Integration (Week 5+)**
- [ ] Ray Serve integration
- [ ] LangChain wrapper
- [ ] Production deployment guide
- [ ] Load testing and benchmarking

---

### 10.3 Code Structure Recommendation

```
/home/linhu/repo/iminn-tools/
├── src/iminnt/
│   ├── llamacpp/
│   │   ├── __init__.py
│   │   ├── service.py          # LlamaCppService class
│   │   ├── api.py              # FastAPI app
│   │   ├── langchain.py        # LangChain wrapper
│   │   └── ray_serve.py        # Ray Serve deployment
│   └── ...
├── scripts/
│   ├── run_api_server.sh       # Start API server
│   └── deploy_ray_serve.sh     # Deploy Ray Serve
├── examples/
│   ├── basic_inference.py
│   ├── langchain_example.py
│   └── ray_serve_example.py
└── docs/
    └── higher_level_frameworks_guide.md  # This file
```

---

## Summary

### Key Recommendations

1. **Use host-based orchestration** (Python on host, QEMU in guest)
2. **Start simple** (subprocess wrapper → FastAPI → Ray Serve)
3. **Leverage existing infrastructure** (9p filesystem, llama-cli binary)
4. **Focus on API layer first** (HTTP endpoints, request handling)
5. **Add frameworks incrementally** (FastAPI → LangChain → Ray Serve)

### Architecture Decision Tree

```
Do you need Python frameworks?
├─ No → Continue using llama-cli directly
└─ Yes → Host-based orchestration
    ├─ Simple API? → FastAPI + subprocess wrapper
    ├─ LangChain? → Custom LLM wrapper
    └─ Production scale? → Ray Serve + QEMU manager
```

### Performance Expectations

- **Direct llama-cli**: Baseline (fastest)
- **Python wrapper**: +5-10% overhead
- **FastAPI**: +10-20% overhead
- **Ray Serve**: +15-25% overhead
- **LangChain**: +20-30% overhead

### Next Actions

1. Implement `LlamaCppService` class
2. Create FastAPI server
3. Test with existing models
4. Document and integrate with `iminnt` CLI
5. Add framework integrations incrementally

---

**Questions or Issues?**

- Check existing QEMU documentation: `docs/qemu_all_in_one_guide.md`
- Review llama.cpp integration: `docs/llamacpp_lifecycle.md`
- Examine codebase structure: `docs/architecture_overview.md`
