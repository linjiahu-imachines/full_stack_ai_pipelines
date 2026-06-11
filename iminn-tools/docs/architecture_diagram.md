# System Architecture Diagram

**Purpose:** Visual representation of the full-stack AI/ML/Agentic System architecture from multiple viewpoints.

**Status:** Current implementation (Phase 1-5 Complete)

---

## Full Stack Architecture Overview

```mermaid
graph TB
    subgraph "Application Layer"
        A1[LangChain Applications]
        A2[LlamaIndex Applications]
        A3[Ray Serve Applications]
        A4[Custom Python Apps]
    end

    subgraph "Framework Layer"
        F1[LangChain Framework]
        F2[LlamaIndex Framework]
        F3[Ray Serve Framework]
        F4[FastAPI Framework]
    end

    subgraph "Service Wrapper Layer"
        SW1[RISCVRISCLLM<br/>LangChain Wrapper]
        SW2[RISCVRISCLLM<br/>LlamaIndex Wrapper]
        SW3[RISCVRISCLLMDeployment<br/>Ray Serve Wrapper]
        SW4[FastAPI Endpoints<br/>/generate, /health]
    end

    subgraph "Service Layer"
        SVC[LlamaCppService<br/>Python Service Wrapper]
    end

    subgraph "Host OS (x86_64 Linux)"
        OS1[Python Runtime]
        OS2[QEMU User Mode<br/>qemu-riscv64]
        OS3[System Calls Interface]
    end

    subgraph "Binary Layer"
        BIN[llama-cli<br/>RISC-V Binary<br/>with IMI Extensions]
    end

    subgraph "QEMU Emulation Layer"
        QEMU[QEMU RISC-V Emulator<br/>IMI CPU Model: imicpu-v1]
    end

    subgraph "Model Layer"
        MODEL[GGUF Model File<br/>stories15M-q4_0.gguf]
    end

    A1 --> F1
    A2 --> F2
    A3 --> F3
    A4 --> F4

    F1 --> SW1
    F2 --> SW2
    F3 --> SW3
    F4 --> SW4

    SW1 --> SVC
    SW2 --> SVC
    SW3 --> SVC
    SW4 --> SVC

    SVC --> OS1
    SVC --> OS2
    OS1 --> OS3
    OS2 --> QEMU
    QEMU --> BIN

    BIN --> MODEL

    style A1 fill:#e1f5ff
    style A2 fill:#e1f5ff
    style A3 fill:#e1f5ff
    style A4 fill:#e1f5ff
    style F1 fill:#fff4e1
    style F2 fill:#fff4e1
    style F3 fill:#fff4e1
    style F4 fill:#fff4e1
    style SW1 fill:#f0e1ff
    style SW2 fill:#f0e1ff
    style SW3 fill:#f0e1ff
    style SW4 fill:#f0e1ff
    style SVC fill:#e1ffe1
    style OS1 fill:#ffe1e1
    style OS2 fill:#ffe1e1
    style OS3 fill:#ffe1e1
    style QEMU fill:#ffffe1
    style BIN fill:#e1e1ff
    style MODEL fill:#f5f5f5
```

---

## Layered View by Component Type

### View 1: Application Perspective

```
┌─────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ LangChain    │  │ LlamaIndex   │  │ Ray Serve    │ │
│  │ Application  │  │ Application  │  │ Application  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  • Agent workflows  • RAG pipelines  • Model serving  │
│  • Chain execution  • Document Q&A   • Scaling        │
│  • Tool calling    • Indexing       • Load balancing │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
                    Uses Framework APIs
                            ↓
┌─────────────────────────────────────────────────────────┐
│  FRAMEWORK LAYER                                        │
├─────────────────────────────────────────────────────────┤
│  • LangChain Core    • LlamaIndex Core  • Ray Serve   │
│  • FastAPI           • Pydantic         • Ray Core    │
└─────────────────────────────────────────────────────────┘
```

### View 2: Framework Perspective

```
┌─────────────────────────────────────────────────────────┐
│  FRAMEWORK LAYER                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LangChain  ──┐                                        │
│               ├──>  RISCVRISCLLM Wrapper              │
│  LlamaIndex ──┤     (Framework-specific LLM adapter)  │
│               │                                        │
│  Ray Serve ───┼──>  RISCVRISCLLMDeployment            │
│               │     (Ray Serve deployment wrapper)    │
│               │                                        │
│  FastAPI  ────┘     /generate, /health endpoints     │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
                    Calls Service Layer
                            ↓
┌─────────────────────────────────────────────────────────┐
│  SERVICE WRAPPER LAYER                                  │
├─────────────────────────────────────────────────────────┤
│  LlamaCppService (Python)                              │
│  • Path validation                                      │
│  • QEMU command building                                │
│  • Subprocess execution                                 │
│  • Output parsing                                       │
└─────────────────────────────────────────────────────────┘
```

### View 3: Service Perspective

```
┌─────────────────────────────────────────────────────────┐
│  SERVICE LAYER (Python)                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  LlamaCppService                                       │
│  ┌───────────────────────────────────────────────────┐ │
│  │ 1. Validate paths (llama-cli, model)             │ │
│  │ 2. Build QEMU command:                            │ │
│  │    qemu-riscv64 [args] llama-cli [inference args]│ │
│  │ 3. Execute subprocess (Popen)                     │ │
│  │ 4. Parse stdout/stderr                            │ │
│  │ 5. Extract generated text                         │ │
│  │ 6. Return response                                 │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Dependencies:                                          │
│  • subprocess (Python stdlib)                          │
│  • pathlib (Python stdlib)                             │
│  • logging (Python stdlib)                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
                    Executes via Host OS
                            ↓
┌─────────────────────────────────────────────────────────┐
│  HOST OS (x86_64 Linux)                                 │
├─────────────────────────────────────────────────────────┤
│  • Python Runtime (3.x)                                 │
│  • QEMU Binary (qemu-riscv64)                           │
│  • System Call Interface                                │
└─────────────────────────────────────────────────────────┘
```

### View 4: OS Perspective

```
┌─────────────────────────────────────────────────────────┐
│  HOST OS: x86_64 Linux                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Python Process (x86_64 native)                   │ │
│  │  • LlamaCppService                                 │ │
│  │  • Framework code (LangChain, etc.)                │ │
│  │  • Application code                                │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  QEMU Process (x86_64 native, user mode)          │ │
│  │  • Translates RISC-V syscalls → x86_64 syscalls   │ │
│  │  • Executes llama-cli (RISC-V binary)             │ │
│  │  • CPU Emulation: imicpu-v1 (IMI extensions)      │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Process Communication:                                 │
│  • subprocess.Popen() creates QEMU process             │
│  • stdin/stdout/stderr pipes                           │
│  • File I/O (model files, config)                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### View 5: QEMU Perspective

```
┌─────────────────────────────────────────────────────────┐
│  QEMU USER MODE (x86_64 Host Process)                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  QEMU RISC-V Emulator                             │ │
│  │  • CPU Model: imicpu-v1 (IMI extensions)         │ │
│  │  • ISA: RV64IMI (Base + IMI custom extensions)   │ │
│  │  • System Call Translation                        │ │
│  │  • Binary Translation (TCG)                       │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  RISC-V Guest Process (inside QEMU)               │ │
│  │  • llama-cli binary (RISC-V ELF)                  │ │
│  │  • Memory: Guest virtual memory                   │ │
│  │  • Instructions: RV64IMI                          │ │
│  │  • IMI Extensions: Custom instructions executed   │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Execution Flow:                                        │
│  1. Load RISC-V binary (llama-cli)                     │
│  2. Translate RISC-V instructions → x86_64             │
│  3. Execute on host CPU (x86_64)                       │
│  4. Translate syscalls (read, write, mmap, etc.)       │
│  5. Return to QEMU wrapper                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
                    Accesses Host File System
                            ↓
┌─────────────────────────────────────────────────────────┐
│  BINARY & MODEL LAYER                                   │
├─────────────────────────────────────────────────────────┤
│  • llama-cli (RISC-V binary, compiled with IMI)        │
│  • GGUF model file (stories15M-q4_0.gguf)              │
│  • File access via QEMU → Host OS → File System        │
└─────────────────────────────────────────────────────────┘
```

### View 6: Binary/Model Perspective

```
┌─────────────────────────────────────────────────────────┐
│  RISC-V BINARY (llama-cli)                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Compiled for RISC-V with IMI extensions          │ │
│  │  • ISA: RV64IMI                                 │ │
│  │  • Instructions: Base RISC-V + IMI custom ops   │ │
│  │  • Binary type: Static or dynamic ELF           │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Execution (via QEMU):                                  │
│  • Loads GGUF model file                               │
│  • Performs inference (uses IMI instructions)          │
│  • Generates text tokens                               │
│  • Outputs to stdout                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                            ↓
                    Reads Model File
                            ↓
┌─────────────────────────────────────────────────────────┐
│  MODEL LAYER                                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GGUF Format File                                       │
│  ┌───────────────────────────────────────────────────┐ │
│  │  • Model weights (quantized)                      │ │
│  │  • Metadata (vocab, config)                       │ │
│  │  • File: stories15M-q4_0.gguf                     │ │
│  │  • Location: dev_env/llama.cpp/models/            │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Access:                                                │
│  • llama-cli reads via mmap/read syscalls              │
│  • QEMU translates syscalls to host file I/O           │
│  • Model loaded into guest virtual memory              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Diagram

```mermaid
sequenceDiagram
    participant App as Application<br/>(LangChain/LlamaIndex)
    participant FW as Framework<br/>(LangChain/LlamaIndex Core)
    participant Wrapper as Framework Wrapper<br/>(RISCVRISCLLM)
    participant Service as Service Layer<br/>(LlamaCppService)
    participant Python as Python Runtime<br/>(x86_64)
    participant QEMU as QEMU User Mode<br/>(qemu-riscv64)
    participant Binary as llama-cli<br/>(RISC-V Binary)
    participant Model as Model File<br/>(GGUF)

    App->>FW: invoke LLM
    FW->>Wrapper: _generate() or complete()
    Wrapper->>Service: service.generate(prompt, params)
    Service->>Python: subprocess.Popen([qemu-riscv64, ...])
    Python->>QEMU: Create QEMU process
    QEMU->>Binary: Load & execute RISC-V binary
    Binary->>Model: mmap/read model file
    Model-->>Binary: Model data
    Binary->>Binary: Inference (uses IMI extensions)
    Binary-->>QEMU: stdout/stderr (generated text)
    QEMU-->>Python: Process output
    Python-->>Service: stdout/stderr strings
    Service->>Service: Parse output, extract text
    Service-->>Wrapper: Generated text
    Wrapper-->>FW: LLMResult/CompletionResponse
    FW-->>App: Final response
```

---

## Architecture Layers Summary

| Layer | Component | Technology | Location | Execution |
|-------|-----------|------------|----------|-----------|
| **Application** | User applications | Python | Host (x86_64) | Native |
| **Framework** | LangChain, LlamaIndex, Ray Serve | Python | Host (x86_64) | Native |
| **Service Wrapper** | RISCVRISCLLM, FastAPI endpoints | Python | Host (x86_64) | Native |
| **Service** | LlamaCppService | Python | Host (x86_64) | Native |
| **Host OS** | Linux kernel, system calls | OS | Host (x86_64) | Native |
| **QEMU** | qemu-riscv64 | C binary | Host (x86_64) | Native (emulates) |
| **Binary** | llama-cli | C++ compiled | Guest (RISC-V) | Emulated via QEMU |
| **Model** | GGUF file | Binary format | Host file system | Accessed via QEMU |

---

## Key Architectural Points

### 1. **Host-Based Orchestration (Option A)**
- All Python code runs natively on x86_64 host
- Only `llama-cli` binary runs in RISC-V emulation
- Frameworks (LangChain, etc.) execute natively

### 2. **QEMU User Mode**
- Simpler than system mode (no full OS emulation)
- Translates RISC-V syscalls to x86_64 syscalls
- Single process execution model
- Lower overhead than system mode

### 3. **IMI Extensions Verification**
- Verified at the binary execution level (via QEMU)
- `llama-cli` uses IMI instructions during inference
- QEMU CPU model (`imicpu-v1`) handles IMI extensions

### 4. **Process Communication**
- Python → QEMU: subprocess.Popen() with pipes
- QEMU → Binary: Binary execution within QEMU process
- File I/O: QEMU translates RISC-V file operations to host file system

### 5. **Scalability Points**
- **Ray Serve**: Multiple replicas, each with own QEMU process
- **FastAPI**: Can handle concurrent requests (each spawns QEMU)
- **QEMU Overhead**: Each inference spawns new QEMU process (can be optimized with persistent processes)

---

## Component Interaction Matrix

| Component | Interacts With | Communication Method |
|-----------|---------------|---------------------|
| **Application** | Framework | Python function calls |
| **Framework** | Service Wrapper | Python function calls |
| **Service Wrapper** | Service | Python function calls |
| **Service** | Host OS | subprocess.Popen() |
| **Host OS** | QEMU | Process creation, pipes |
| **QEMU** | Binary | Binary execution, syscall translation |
| **Binary** | Model | File I/O (via QEMU) |

---

## File Locations in Codebase

```
Application Code:
  scripts/test_langchain_integration.py
  scripts/test_llamaindex_integration.py
  scripts/test_ray_serve.py

Framework Wrappers:
  src/iminnt/llamacpp_langchain.py     → RISCVRISCLLM (LangChain)
  src/iminnt/llamacpp_llamaindex.py    → RISCVRISCLLM (LlamaIndex)
  src/iminnt/llamacpp_ray_serve.py     → RISCVRISCLLMDeployment
  src/iminnt/llamacpp_api.py           → FastAPI endpoints

Service Layer:
  src/iminnt/llamacpp_service.py       → LlamaCppService

Infrastructure:
  src/iminnt/constants.py              → Paths, configuration
  src/iminnt/utils.py                  → Utilities (shell, etc.)

Binaries:
  dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli  (RISC-V)
  dev_env/csqemu-v9/bin/qemu-riscv64                    (x86_64)

Models:
  dev_env/llama.cpp/models/stories15M-q4_0.gguf
```

---

## References

- **Current Implementation**: `docs/option_a_quickstart.md`
- **Framework Investigation**: `docs/framework_investigation.md`
- **Service Implementation**: `src/iminnt/llamacpp_service.py`
- **QEMU Setup**: `docs/qemu_all_in_one_guide.md`
