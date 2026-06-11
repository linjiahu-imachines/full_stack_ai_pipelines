# Company Overall Architecture - Full Tech Stack
**Purpose:** High-level architecture showing complete scope from applications to hardware  
**Date:** January 2026  
**Scope:** End-to-end AI workload deployment on RISC-V with custom extensions

---

## Executive Architecture Overview

```mermaid
graph TB
    subgraph "Application Layer"
        APP1[Enterprise LLM Applications]
        APP2[Robotics & Edge Applications]
        APP3[Vision-Language Applications]
        APP4[Research & Development]
    end

    subgraph "Framework Layer"
        FW1[LangChain<br/>Agent Workflows]
        FW2[LlamaIndex<br/>RAG Systems]
        FW3[Ray Serve<br/>Distributed Serving]
        FW4[PyTorch/TensorFlow<br/>Model Training/Inference]
    end

    subgraph "Serving/Runtime Layer"
        SRV1[vLLM<br/>LLM Serving Engine]
        SRV2[llama.cpp<br/>Edge Inference]
        SRV3[Ray Cluster<br/>Orchestration]
        SRV4[FastAPI<br/>HTTP Endpoints]
    end

    subgraph "Compiler/Toolchain Layer"
        COMP1[IREE Compiler<br/>MXFP4 Support]
        COMP2[Triton Backend<br/>RISC-V Target]
        COMP3[MLIR<br/>IR & Optimization]
        COMP4[RISC-V Toolchain<br/>GCC/Clang]
    end

    subgraph "SDK & Libraries"
        SDK1[CUDA-equivalent SDK]
        SDK2[MXFP4 Quantization]
        SDK3[Matrix Microkernels]
        SDK4[Runtime Libraries]
    end

    subgraph "Validation & Testing Layer"
        VAL1[Functional Simulator<br/>QEMU]
        VAL2[Performance Simulator<br/>PSIM]
        VAL3[FPGA Prototype<br/>4-6 Cores]
    end

    subgraph "Hardware Layer"
        HW1[RISC-V Custom Chip<br/>with IMI Extensions]
        HW2[MXFP4 Matrix Units]
        HW3[HBM/SRAM Memory]
        HW4[Multi-Core Architecture]
    end

    APP1 --> FW1
    APP2 --> FW2
    APP3 --> FW3
    APP4 --> FW4

    FW1 --> SRV1
    FW2 --> SRV2
    FW3 --> SRV3
    FW4 --> SRV1

    SRV1 --> COMP1
    SRV2 --> COMP2
    SRV3 --> COMP1
    SRV4 --> SRV1

    COMP1 --> SDK1
    COMP2 --> SDK2
    COMP3 --> SDK3
    COMP4 --> SDK4

    SDK1 --> VAL1
    SDK2 --> VAL2
    SDK3 --> VAL3
    SDK4 --> VAL1

    VAL1 --> HW1
    VAL2 --> HW1
    VAL3 --> HW1

    HW1 --> HW2
    HW1 --> HW3
    HW1 --> HW4

    style APP1 fill:#e1f5ff
    style APP2 fill:#e1f5ff
    style APP3 fill:#e1f5ff
    style APP4 fill:#e1f5ff
    style FW1 fill:#fff4e1
    style FW2 fill:#fff4e1
    style FW3 fill:#fff4e1
    style FW4 fill:#fff4e1
    style SRV1 fill:#f0e1ff
    style SRV2 fill:#f0e1ff
    style SRV3 fill:#f0e1ff
    style SRV4 fill:#f0e1ff
    style COMP1 fill:#e1ffe1
    style COMP2 fill:#e1ffe1
    style COMP3 fill:#e1ffe1
    style COMP4 fill:#e1ffe1
    style SDK1 fill:#ffe1e1
    style SDK2 fill:#ffe1e1
    style SDK3 fill:#ffe1e1
    style SDK4 fill:#ffe1e1
    style VAL1 fill:#ffffe1
    style VAL2 fill:#ffffe1
    style VAL3 fill:#ffffe1
    style HW1 fill:#e1e1ff
    style HW2 fill:#e1e1ff
    style HW3 fill:#e1e1ff
    style HW4 fill:#e1e1ff
```

---

## Detailed Layer-by-Layer Architecture

### Layer 1: Application Layer (End User)

**Purpose:** Customer-facing AI applications and use cases

```mermaid
graph LR
    subgraph "Enterprise Data Center"
        E1[Llama-4 Serving]
        E2[GPT-OSS Serving]
        E3[DeepSeek Models]
    end
    
    subgraph "Edge & Robotics"
        R1[Autonomous Robots]
        R2[Automotive AI]
        R3[On-Device AI]
    end
    
    subgraph "Research & Development"
        RD1[Custom Kernels]
        RD2[Novel Architectures]
        RD3[Proprietary Models]
    end
    
    subgraph "Vision-Language"
        VL1[Multi-Modal Fusion]
        VL2[Computer Vision]
        VL3[VLMs]
    end

    style E1 fill:#e1f5ff
    style E2 fill:#e1f5ff
    style E3 fill:#e1f5ff
    style R1 fill:#ffe1f5
    style R2 fill:#ffe1f5
    style R3 fill:#ffe1f5
    style RD1 fill:#f5ffe1
    style RD2 fill:#f5ffe1
    style RD3 fill:#f5ffe1
    style VL1 fill:#f5e1ff
    style VL2 fill:#f5e1ff
    style VL3 fill:#f5e1ff
```

**Owner:** Product/Business teams with Software Stack support  
**Timeline:** Customer deployments start Q3-Q4 2026

---

### Layer 2: Framework Layer (Developer Experience)

**Purpose:** High-level AI/ML frameworks for application development

| Framework | Purpose | Integration Point | Status |
|-----------|---------|------------------|--------|
| **LangChain** | Agent workflows, chains, tools | RISCVRISCLLM wrapper | ✅ Phase 3 Complete |
| **LlamaIndex** | RAG, document indexing, querying | RISCVRISCLLM wrapper | ✅ Phase 5 Complete |
| **Ray Serve** | Distributed serving, scaling | RISCVRISCLLMDeployment | ✅ Phase 4 Complete |
| **PyTorch** | Model training, torch.compile | IREE backend (Compiler Team) | ❌ Q2 2026 |
| **TensorFlow** | Model training, inference | IREE backend (Compiler Team) | ❌ Q2-Q3 2026 |

**Owner:** 
- **Software Stack:** Framework integration layers, wrappers, documentation
- **Compiler Team:** IREE/Triton backends (dependency)
- **Framework vendors:** Upstream support

**Customer Question:** "How much code modification is required?"  
**Answer:** Minimal - use standard APIs, just change backend (coordinated with Compiler Team)

---

### Layer 3: Serving/Runtime Layer (Inference Engines)

**Purpose:** Production-grade inference serving for LLMs and AI models

```mermaid
graph TB
    subgraph "High-Throughput Serving"
        V1[vLLM Engine<br/>PagedAttention<br/>Continuous Batching]
        V2[vLLM Platform Plugin<br/>RISC-V Worker<br/>Memory Manager]
    end
    
    subgraph "Edge/Lightweight Serving"
        L1[llama.cpp<br/>GGML Backend<br/>RVV Intrinsics]
        L2[Quantization Support<br/>MXFP4, GGUF]
    end
    
    subgraph "Orchestration"
        R1[Ray Cluster<br/>Multi-Node<br/>Disaggregated Inference]
        R2[Ray Placement Groups<br/>Hybrid NVIDIA/RISC-V]
    end
    
    subgraph "API Layer"
        API[FastAPI Server<br/>HTTP Endpoints<br/>OpenAI-Compatible]
    end

    V1 --> V2
    L1 --> L2
    R1 --> R2
    V2 --> API
    L2 --> API
    R2 --> API

    style V1 fill:#e1f5ff
    style V2 fill:#e1f5ff
    style L1 fill:#ffe1f5
    style L2 fill:#ffe1f5
    style R1 fill:#f5ffe1
    style R2 fill:#f5ffe1
    style API fill:#f5e1ff
```

**Owner:** Software Stack (PRIMARY)  
**Dependencies:** Compiler Team (IREE backend for vLLM integration)  
**Timeline:**
- llama.cpp: ✅ Q1 2026 (done, needs enhancement)
- vLLM: ❌ Q2-Q3 2026 (critical path, blocked by IREE)
- Ray multi-node: ⚠️ Q3 2026 (partial, needs enhancement)

---

### Layer 4: Compiler/Toolchain Layer (The "Brain")

**Purpose:** Transform high-level models into optimized RISC-V machine code

```mermaid
graph TB
    subgraph "Frontend (Model Input)"
        IN1[PyTorch Models<br/>torch.compile]
        IN2[TensorFlow Models<br/>tf.function]
        IN3[JAX Models]
        IN4[ONNX Models]
    end
    
    subgraph "MLIR Compilation Pipeline"
        M1[StableHLO IR]
        M2[MLIR Dialects<br/>Custom RISC-V ISA]
        M3[Graph Optimization<br/>Layer Fusion]
        M4[MXFP4 Tensor<br/>Identification]
    end
    
    subgraph "IREE Compiler"
        I1[Data Tiling<br/>for RISC-V Layout]
        I2[Microkernel Slots<br/>riscv_mxfp4_matmul]
        I3[Code Generation<br/>RISC-V + IMI Extensions]
    end
    
    subgraph "Triton Path (Custom Kernels)"
        T1[Triton Python Code]
        T2[Triton IR]
        T3[MLIR Lowering]
        T4[RISC-V Matrix Instr]
    end
    
    subgraph "Backend (Binary Output)"
        OUT1[RISC-V Binary<br/>with IMI Extensions]
        OUT2[MXFP4 Optimized<br/>Matrix Operations]
    end

    IN1 --> M1
    IN2 --> M1
    IN3 --> M1
    IN4 --> M1
    
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> I1
    
    I1 --> I2
    I2 --> I3
    I3 --> OUT1
    
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> OUT1
    
    OUT1 --> OUT2

    style IN1 fill:#e1f5ff
    style IN2 fill:#e1f5ff
    style IN3 fill:#e1f5ff
    style IN4 fill:#e1f5ff
    style M1 fill:#fff4e1
    style M2 fill:#fff4e1
    style M3 fill:#fff4e1
    style M4 fill:#fff4e1
    style I1 fill:#f0e1ff
    style I2 fill:#f0e1ff
    style I3 fill:#f0e1ff
    style T1 fill:#e1ffe1
    style T2 fill:#e1ffe1
    style T3 fill:#e1ffe1
    style T4 fill:#e1ffe1
    style OUT1 fill:#ffe1e1
    style OUT2 fill:#ffe1e1
```

**Owner:** Compiler Team (PRIMARY) - Software Stack works closely with this team  
**Software Stack Coordination:** Ensure backends meet integration requirements  
**Timeline:** Q2-Q3 2026 (critical path)  
**Customer Question:** "Do you have a CUDA-equivalent SDK?"  
**Answer:** Yes - IREE + Triton provide CUDA-like developer experience (Compiler Team delivers)

**Key Components:**
- **IREE (Compiler Team):** Whole-graph optimization, MXFP4 native support (Jan 2026 release)
- **Triton (Compiler Team):** Custom kernel support, NVIDIA code migration path
- **MLIR (Compiler Team):** Intermediate representation, optimization passes
- **RISC-V Toolchain (Compiler Team):** Final binary generation (GCC/Clang)

---

### Layer 5: SDK & Libraries (Developer Tools)

**Purpose:** Tools and libraries for RISC-V AI development

| Component | Purpose | Owner | Status |
|-----------|---------|-------|--------|
| **CUDA-equivalent SDK Core** | CUDA-like API for RISC-V | Compiler Team | ❌ Q2-Q3 |
| **SDK Documentation & Examples** | Tutorials, migration guides | SW Stack | ❌ Q2-Q3 |
| **MXFP4 Quantization Toolkit** | Model quantization to MXFP4 | Compiler Team | ❌ Q2 |
| **Matrix Microkernels (ukernels)** | Optimized RISC-V matrix ops | Compiler Team | ❌ Q2 |
| **Runtime Libraries** | Memory management, scheduling | SW Stack | ⚠️ Q2 |
| **Model Conversion Tools** | PyTorch → MXFP4 pipeline | SW Stack | ❌ Q2 |
| **Performance Profiling Tools** | Tracing, analysis | SW Stack + PSim | ⚠️ Q2-Q3 |

**Owner:** 
- **Compiler Team:** SDK core, microkernels, quantization toolkit
- **Software Stack:** SDK documentation, examples, integration, model conversion
- **Timeline:** Q2-Q3 2026

---

### Layer 6: Validation & Testing Layer (Pre-Silicon)

**Purpose:** Validate software stack before hardware tape-out

```mermaid
graph TB
    subgraph "Functional Validation (QEMU)"
        Q1[QEMU User Mode<br/>Single Process<br/>Syscall Translation]
        Q2[QEMU System Mode<br/>Full OS<br/>Multi-Core Support]
        Q3[Custom CPU Model<br/>imicpu-v1<br/>IMI Extensions]
    end
    
    subgraph "Performance Validation (PSIM)"
        P1[Cycle-Accurate Simulation]
        P2[Performance Metrics<br/>IPC, Memory BW<br/>Cache Behavior]
        P3[Benchmark Execution<br/>MLPerf Suite]
    end
    
    subgraph "Hardware Prototype (FPGA)"
        F1[Local FPGA<br/>4-6 Cores<br/>Q1 2026]
        F2[Hardware Validation<br/>Real Timing<br/>Power Measurement]
        F3[Integration Testing<br/>Multi-Core<br/>Multi-Node]
    end
    
    subgraph "Cloud Simulation"
        C1[Synopsys Cloud<br/>Remote Access<br/>Current]
    end

    Q1 --> Q3
    Q2 --> Q3
    P1 --> P2
    P2 --> P3
    F1 --> F2
    F2 --> F3

    style Q1 fill:#e1f5ff
    style Q2 fill:#e1f5ff
    style Q3 fill:#e1f5ff
    style P1 fill:#ffe1f5
    style P2 fill:#ffe1f5
    style P3 fill:#ffe1f5
    style F1 fill:#f5ffe1
    style F2 fill:#f5ffe1
    style F3 fill:#f5ffe1
    style C1 fill:#f5e1ff
```

#### QEMU (Functional Simulator)
**Purpose:** Verify functional correctness of RISC-V binaries

**Current Status:**
- ✅ QEMU user mode working (Phase 1-5 complete)
- ⚠️ QEMU system mode available (needs integration)
- ✅ Custom CPU model (imicpu-v1) supports IMI extensions

**Use Cases:**
- Rapid software development and testing
- Functional correctness validation
- Framework integration testing

**Owner:** Software Stack (usage) + Hardware team (CPU model definition)  
**Timeline:** ✅ Currently working

#### PSIM (Performance Simulator)
**Purpose:** Predict hardware performance before tape-out

**Capabilities:**
- Cycle-accurate simulation
- Memory hierarchy modeling
- Performance metrics (IPC, bandwidth, latency)
- MLPerf benchmark execution

**Use Cases:**
- Performance target validation
- Architecture exploration
- Customer performance claims

**Owner:** PSim Team (PRIMARY) + Software Stack (benchmark integration)  
**Timeline:** Q2-Q3 2026 (critical for customer demos)

#### FPGA (Hardware Prototype)
**Purpose:** Real hardware validation with actual timing

**Availability:**
- Q1 2026: Local FPGA with 4-6 cores
- Real timing, power measurements
- Multi-core integration testing

**Use Cases:**
- Hardware-in-the-loop testing
- Real performance validation
- Tape-out readiness verification

**Owner:** Hardware Team (FPGA setup) + Software Stack (SW stack deployment)  
**Timeline:** Q1 2026 (VP requirement)

#### Synopsys Cloud
**Purpose:** Remote simulation access (current)

**Current Status:** ✅ Available now  
**Use Cases:** Remote development, simulation without local setup

---

### Layer 7: Hardware Layer (Physical Chip)

**Purpose:** Custom RISC-V processor with AI extensions

```mermaid
graph TB
    subgraph "RISC-V Core"
        C1[Multi-Core Architecture<br/>4-6+ Cores]
        C2[RV64IMI ISA<br/>IMI Custom Extensions]
        C3[Vector Unit<br/>RVV 1.0/2.0]
    end
    
    subgraph "AI Accelerator Units"
        A1[MXFP4 Matrix Units<br/>E2M1 + E8M0 Scales]
        A2[Matrix Multiplication<br/>Custom Instructions]
        A3[Spatial Data Locality<br/>2D/3D Convolutions]
    end
    
    subgraph "Memory Hierarchy"
        M1[HBM Memory<br/>High Bandwidth]
        M2[SRAM Cache<br/>Low Latency]
        M3[Non-Contiguous<br/>KV-Cache Support]
    end
    
    subgraph "System Integration"
        S1[Multi-Node<br/>Interconnect]
        S2[PCIe/NVLink<br/>Host Interface]
        S3[Power Management]
    end

    C1 --> C2
    C2 --> C3
    A1 --> A2
    A2 --> A3
    M1 --> M2
    M2 --> M3
    S1 --> S2
    S2 --> S3

    style C1 fill:#e1f5ff
    style C2 fill:#e1f5ff
    style C3 fill:#e1f5ff
    style A1 fill:#ffe1f5
    style A2 fill:#ffe1f5
    style A3 fill:#ffe1f5
    style M1 fill:#f5ffe1
    style M2 fill:#f5ffe1
    style M3 fill:#f5ffe1
    style S1 fill:#f5e1ff
    style S2 fill:#f5e1ff
    style S3 fill:#f5e1ff
```

**Owner:** Hardware Team (PRIMARY)  
**Timeline:** Tape-out end of September 2026

**Key Features:**
- **MXFP4 Support:** Native OCP Microscaling (E2M1 + E8M0)
- **Multi-Core:** 4-6+ cores for parallel inference
- **Memory:** HBM for bandwidth, SRAM for low latency
- **Interconnect:** Multi-node support for clusters

**Software Stack Dependency:**
- ISA specification (needed ASAP for compiler development)
- Matrix unit details (needed ASAP for microkernel development)
- Memory architecture (needed Q1 for PagedAttention)

---

## Development & Deployment Timeline

```mermaid
gantt
    title 2026 Development Timeline
    dateFormat YYYY-MM-DD
    section Software Stack
    llama.cpp native RISC-V       :done, 2026-01-01, 2026-03-31
    IREE integration              :crit, 2026-02-01, 2026-06-30
    Triton backend                :crit, 2026-02-01, 2026-05-31
    vLLM platform plugin          :crit, 2026-04-01, 2026-08-31
    MLPerf benchmarks             :crit, 2026-04-01, 2026-08-31
    Ray multi-node                :2026-05-01, 2026-08-31
    
    section Hardware
    ISA specification             :crit, 2026-01-01, 2026-02-28
    FPGA availability             :crit, 2026-01-01, 2026-03-31
    Tape-out                      :milestone, 2026-09-30, 0d
    
    section Validation
    QEMU validation               :done, 2026-01-01, 2026-03-31
    PSIM integration              :2026-04-01, 2026-08-31
    FPGA testing                  :2026-03-01, 2026-09-30
    
    section Customer Demos
    Performance benchmarks        :crit, 2026-06-01, 2026-09-30
    Enterprise demos              :2026-08-01, 2026-12-31
```

---

## Ownership Matrix

| Layer | Component | Primary Owner | Secondary Owner | Status |
|-------|-----------|--------------|----------------|--------|
| **Application** | Customer apps | Product/Business | SW Stack (support) | Q3-Q4 2026 |
| **Framework** | LangChain/LlamaIndex | SW Stack | Framework vendors | ✅ Done |
| **Framework** | PyTorch/TensorFlow integration | SW Stack | Compiler Team (backends) | ❌ Q2-Q3 |
| **Serving** | vLLM plugin | SW Stack | Compiler Team (IREE) | ❌ Q2-Q3 |
| **Serving** | llama.cpp | SW Stack | - | ✅ Done (enhance Q1-Q2) |
| **Serving** | Ray Serve | SW Stack | - | ⚠️ Partial (enhance Q3) |
| **Compiler** | IREE integration | Compiler Team | SW Stack (coordination) | ❌ Q2-Q3 |
| **Compiler** | Triton backend | Compiler Team | SW Stack (coordination) | ❌ Q2 |
| **Compiler** | RISC-V toolchain | Compiler Team | Hardware team | ⚠️ Available |
| **SDK** | CUDA-equivalent core | Compiler Team | SW Stack (docs) | ❌ Q2-Q3 |
| **SDK** | Microkernels | Compiler Team | Hardware team | ❌ Q2 |
| **Validation** | QEMU | SW Stack (usage) | Hardware (CPU model) | ✅ Working |
| **Validation** | PSIM | PSim Team | SW Stack (benchmarks) | Q2-Q3 |
| **Validation** | FPGA | Hardware Team | SW Stack (SW deployment) | Q1 |
| **Hardware** | RISC-V chip | Hardware Team | - | Tape-out Sept |

---

## Critical Dependencies Flow

```mermaid
graph LR
    HW[Hardware Team<br/>ISA Spec<br/>Matrix Units<br/>Memory Arch]
    SW[Software Stack<br/>IREE/Triton<br/>vLLM/llama.cpp<br/>SDK]
    PSIM[PSim Team<br/>Performance Sim<br/>MLPerf Infra]
    CUST[Customer Demos<br/>Benchmarks<br/>Performance Claims]

    HW -->|URGENT: Q1| SW
    SW -->|Q2-Q3| PSIM
    PSIM -->|Q2-Q3| CUST
    HW -->|Q1: FPGA| SW
    SW -->|Q2-Q3: Integration| PSIM

    style HW fill:#ffe1e1
    style SW fill:#e1f5ff
    style PSIM fill:#ffe1f5
    style CUST fill:#e1ffe1
```

**Critical Path:**
1. Hardware specs (ISA, matrix units) → Software Stack (IREE/Triton) → Q1-Q2
2. Software Stack (benchmarks) → PSim (validation) → Customer demos → Q2-Q3
3. FPGA availability → Software Stack (multi-core) → Q1
4. All paths converge → Tape-out readiness → Sept 2026

---

## Technology Stack Summary

### Upper Stack (Application to Runtime)
- **Frameworks:** LangChain, LlamaIndex, PyTorch, TensorFlow
- **Serving:** vLLM, llama.cpp, Ray Serve, FastAPI
- **Owner:** Software Stack (PRIMARY)
- **Status:** 30% complete (basic done, production components needed)

### Middle Stack (Compiler to SDK)
- **Compiler:** IREE, Triton, MLIR, RISC-V toolchain
- **SDK:** CUDA-equivalent, microkernels, quantization tools
- **Owner:** Software Stack (PRIMARY)
- **Status:** 10% complete (critical path, Q2-Q3)

### Lower Stack (Validation to Hardware)
- **Validation:** QEMU (functional), PSIM (performance), FPGA (hardware)
- **Hardware:** RISC-V chip with MXFP4 matrix units
- **Owner:** Software Stack (QEMU), PSim Team (PSIM), Hardware Team (FPGA/Chip)
- **Status:** 
  - QEMU: ✅ Working
  - PSIM: ⚠️ Q2-Q3 integration
  - FPGA: ⚠️ Q1 availability
  - Hardware: Sept tape-out

---

## Key Takeaways

### Complete End-to-End Stack
- **7 layers** from applications to hardware
- **3 validation paths** (QEMU, PSIM, FPGA) before production
- **2 critical paths:**
  1. Software stack → PSIM → Customer demos
  2. Hardware specs → Software stack → Tape-out

### Software Stack is the Central Hub
- Connects applications to hardware
- Depends on hardware specs (critical blocker)
- Feeds into validation (PSIM, FPGA)
- Delivers customer-facing components (SDK, benchmarks)

### Critical Timeline
- **Q1:** FPGA ready, multi-core support, hardware specs finalized
- **Q2:** IREE/Triton/vLLM integration, MLPerf start
- **Q3:** Performance validation, customer demos, tape-out readiness
- **Sept:** Tape-out deadline

---

**Document Status:** Complete company architecture overview  
**Next Steps:** Use this for VP presentation to show overall scope and dependencies
