# Software Stack Scope - CORRECTED
**Role:** Director of Software Stack  
**Date:** January 2026  
**Key Clarification:** Compiler/toolchain is a separate team you work closely with

---

## ✅ YOUR SCOPE (What You Own)

### 1. Framework Integration Layer
**Owner:** Software Stack (PRIMARY)

- **LangChain Integration** ✅ Complete (Phase 3)
  - RISCVRISCLLM wrapper
  - Agent workflows
  - Tool calling support

- **LlamaIndex Integration** ✅ Complete (Phase 5)
  - RISCVRISCLLM wrapper
  - RAG systems
  - Document indexing and querying

- **PyTorch/TensorFlow Integration** ❌ Q2-Q3 2026
  - Integration layers between frameworks and RISC-V backend
  - Migration guides and documentation
  - Example models and tutorials
  - **Depends on:** Compiler Team's IREE backend

### 2. Serving/Runtime Layer
**Owner:** Software Stack (PRIMARY)

- **vLLM Platform Plugin** ❌ Q2-Q3 2026
  - Custom memory managers and worker types
  - PagedAttention for RISC-V
  - KV-cache memory management
  - **Depends on:** Compiler Team's IREE backend

- **llama.cpp Integration** ✅ Complete (Phase 1-5)
  - GGML backend
  - QEMU execution
  - Framework wrappers
  - **Enhancements needed:** Native RISC-V, multi-core (Q1-Q2)

- **Ray Serve Integration** ⚠️ Partial (Phase 4)
  - Distributed serving
  - Horizontal scaling
  - **Enhancements needed:** Multi-node, disaggregation (Q3)

- **FastAPI Server** ✅ Complete (Phase 2)
  - HTTP endpoints
  - OpenAI-compatible API

### 3. SDK Integration & Documentation
**Owner:** Software Stack (documentation) + Compiler Team (core)

- **SDK Documentation** ❌ Q2-Q3 2026
  - Tutorials and examples
  - Migration guides (CUDA → RISC-V)
  - API reference
  - Best practices

- **Model Conversion Tools** ❌ Q2 2026
  - PyTorch → MXFP4 pipeline
  - Format validation tools

### 4. Benchmarking & Performance Validation
**Owner:** Software Stack (PRIMARY) + PSim Team (infrastructure)

- **MLPerf Integration** ❌ Q2-Q3 2026
  - Benchmark suite integration
  - Model preparation
  - Results analysis

- **NVIDIA Comparisons** ❌ Q2-Q3 2026
  - Benchmark execution
  - Performance analysis
  - Cost (TCO) analysis

- **Diversity Benchmarks** ❌ Q2-Q3 2026
  - CNNs, ViTs, VLMs
  - Multi-modal models
  - Performance reports

### 5. System Integration
**Owner:** Software Stack (PRIMARY)

- **Multi-Core Support** ⚠️ Q1 2026
  - llama.cpp multi-threading
  - FPGA testing (4-6 cores)

- **Multi-Node System** ❌ Q3 2026
  - Ray cluster orchestration
  - Disaggregated inference

- **Robotics Pipeline** ❌ Q2-Q3 2026
  - Real-time inference
  - Vision-language models
  - Edge deployment patterns

### 6. Validation (QEMU)
**Owner:** Software Stack (usage) + Hardware Team (CPU model)

- **QEMU Functional Validation** ✅ Working
  - User mode execution
  - System mode (available)
  - Custom CPU model (imicpu-v1)

---

## 🔴 NOT YOUR SCOPE (Compiler Team - You Work Closely With Them)

### Compiler/Toolchain Layer
**Owner:** Compiler Team (PRIMARY)

- **IREE Compiler** ❌ Q2-Q3 2026
  - MLIR dialect for custom RISC-V ISA
  - MXFP4 tensor identification and lowering
  - Data Tiling for RISC-V hardware
  - Microkernel library (ukernels)
  - Graph optimization and layer fusion

- **Triton Backend** ❌ Q2 2026
  - Triton-to-RISC-V compiler backend
  - MLIR lowering from Triton to RISC-V ISA
  - Custom kernel support
  - Legacy NVIDIA kernel migration

- **RISC-V Toolchain** ⚠️ Available
  - GCC/Clang for RISC-V
  - Binary generation
  - Linker and assembler

- **CUDA-equivalent SDK Core** ❌ Q2-Q3 2026
  - Core API implementation
  - Runtime libraries (low-level)
  - Matrix operations API

- **MXFP4 Quantization Toolkit** ❌ Q2 2026
  - Quantization algorithms
  - Low-level format conversion

- **Matrix Microkernels (ukernels)** ❌ Q2 2026
  - Hand-optimized C++ microkernels
  - RISC-V matrix instruction usage
  - Performance-critical operations

### Your Coordination with Compiler Team
- Ensure backends meet your integration requirements
- Provide use cases and requirements
- Validate end-to-end workflows
- Test integration with frameworks and serving layers

---

## ❌ NOT YOUR SCOPE (Other Teams)

### Hardware Team
- Custom ISA design and specification
- Matrix unit hardware implementation
- Memory architecture (HBM/SRAM) design
- FPGA hardware availability
- RISC-V chip tape-out

### PSim Team
- Performance simulation infrastructure
- MLPerf infrastructure setup
- Performance validation framework

---

## 🔑 YOUR CRITICAL DEPENDENCIES

### From Compiler Team (CRITICAL BLOCKER)
**Status:** ❌ All critical items not started

1. **IREE Backend for RISC-V**
   - Needed for: vLLM integration, PyTorch/TensorFlow support
   - Timeline: Q2-Q3 2026
   - **Impact:** Blocks 80% of your deliverables

2. **Triton Backend for RISC-V**
   - Needed for: Custom kernel support, framework flexibility
   - Timeline: Q2 2026
   - **Impact:** Blocks advanced use cases

3. **CUDA-equivalent SDK Core**
   - Needed for: Developer experience, migration guides
   - Timeline: Q2-Q3 2026
   - **Impact:** Blocks SDK documentation

### From Hardware Team (URGENT)
**Status:** ⚠️ Needed ASAP

1. **ISA Specification**
   - Needed for: Compiler Team's backend development
   - Timeline: ASAP (blocks Compiler Team)

2. **Matrix Unit Architecture**
   - Needed for: Compiler Team's microkernel development
   - Timeline: ASAP (blocks Compiler Team)

3. **FPGA Availability**
   - Needed for: Multi-core testing
   - Timeline: Q1 2026

4. **Memory Architecture (HBM/SRAM)**
   - Needed for: PagedAttention implementation
   - Timeline: Q1 2026

### From PSim Team
**Status:** ⚠️ Q2-Q3 needed

1. **Performance Simulation Infrastructure**
   - Needed for: MLPerf benchmarks
   - Timeline: Q2 2026

2. **MLPerf Integration Support**
   - Needed for: Official results
   - Timeline: Q2-Q3 2026

---

## 📊 SCOPE SUMMARY

### What You Own (Software Stack)
- **Framework Integration:** LangChain, LlamaIndex, PyTorch/TensorFlow integration layers
- **Serving/Runtime:** vLLM, llama.cpp, Ray Serve, FastAPI
- **SDK Integration:** Documentation, examples, migration guides
- **Benchmarking:** MLPerf, NVIDIA comparisons, diversity benchmarks
- **System Integration:** Multi-core, multi-node, robotics pipeline

### What You DON'T Own (But Work Closely With)
- **Compiler/Toolchain:** IREE, Triton, MLIR, microkernels (Compiler Team)
- **Hardware:** ISA, matrix units, FPGA, chip (Hardware Team)
- **Simulation:** PSIM infrastructure (PSim Team)

### Your Value Proposition
- **Customer-Facing:** All customer deliverables (benchmarks, docs, demos)
- **Integration:** Connect frameworks to RISC-V backend (via Compiler Team)
- **Validation:** Prove system works end-to-end
- **Deployment:** Production-ready serving and orchestration

---

## 🎯 CRITICAL SUCCESS FACTORS

### 1. Compiler Team Deliverables (CRITICAL BLOCKER)
**Timeline:** Q2-Q3 2026  
**Impact:** 80% of your work depends on this

Your deliverables blocked by Compiler Team:
- vLLM integration (needs IREE backend)
- PyTorch/TensorFlow integration (needs IREE backend)
- SDK documentation (needs SDK core)
- Advanced framework features (needs Triton backend)

### 2. Framework Integration (YOUR PRIMARY WORK)
**Timeline:** Q2-Q3 2026  
**Impact:** Customer-facing, high visibility

Your primary deliverables:
- PyTorch/TensorFlow integration layers
- vLLM platform plugin
- Migration guides and documentation
- Example models and tutorials

### 3. Benchmarking (CUSTOMER REQUIREMENT)
**Timeline:** Q2-Q3 2026  
**Impact:** Customer demos, sales enablement

Your primary deliverables:
- MLPerf benchmark integration
- NVIDIA comparison results
- Diversity benchmarks (CNNs, ViTs, VLMs)
- Performance analysis reports

### 4. Multi-Core Support (FPGA READINESS)
**Timeline:** Q1 2026  
**Impact:** Hardware validation

Your primary deliverables:
- Multi-core llama.cpp
- FPGA deployment pipeline
- Performance scaling analysis

---

## 📅 TIMELINE & PRIORITIES

### Q1 2026 (Jan-Mar) - Foundation
**Your Work:**
- ✅ Multi-core llama.cpp support
- ✅ FPGA deployment pipeline
- ✅ Native RISC-V binary (remove QEMU)

**Dependencies:**
- ⚠️ Hardware Team: FPGA availability, ISA specification
- ⚠️ Compiler Team: Can start work once ISA available

### Q2 2026 (Apr-Jun) - Integration
**Your Work:**
- ❌ PyTorch/TensorFlow integration layers
- ❌ vLLM platform plugin
- ❌ MLPerf benchmark integration (start)
- ❌ Robotics pipeline components

**Dependencies:**
- 🔴 Compiler Team: IREE backend, Triton backend (CRITICAL BLOCKER)
- ⚠️ PSim Team: Performance infrastructure

### Q3 2026 (Jul-Sep) - Production
**Your Work:**
- ❌ vLLM full production integration
- ❌ Ray multi-node cluster
- ❌ MLPerf official results
- ❌ Diversity benchmarks
- ❌ Tape-out readiness validation

**Dependencies:**
- 🔴 Compiler Team: Production-ready backends
- ⚠️ PSim Team: Performance validation
- ⚠️ Hardware Team: Final specs, tape-out

### Q4 2026 (Oct-Dec) - Customer Demos
**Your Work:**
- ❌ Enterprise demos
- ❌ Customer-ready documentation
- ❌ Migration guides complete

---

## 🔗 TEAM COORDINATION

### With Compiler Team (CRITICAL)
**Coordination Frequency:** Weekly (recommended)

Your inputs to Compiler Team:
- Framework integration requirements
- API/interface specifications
- Use cases and example workloads
- End-to-end validation results

Compiler Team inputs to you:
- Backend availability and features
- API changes and updates
- Performance characteristics
- Known limitations

### With Hardware Team
**Coordination Frequency:** Bi-weekly

Your inputs to Hardware Team:
- Software requirements for ISA/matrix units
- Multi-core integration feedback
- FPGA testing results

Hardware Team inputs to you:
- ISA specification updates
- FPGA availability and access
- Performance targets

### With PSim Team
**Coordination Frequency:** Bi-weekly

Your inputs to PSim Team:
- Benchmark workloads
- Performance metrics requirements
- Validation scenarios

PSim Team inputs to you:
- Simulation infrastructure access
- Performance results
- Bottleneck analysis

---

## 💬 KEY TALKING POINTS FOR VP

### Scope Clarification
"I own the framework and serving layers (layers 2-3), working closely with the Compiler Team who owns the compiler/toolchain layer (layer 4)."

### Critical Dependencies
"80% of my deliverables are blocked by the Compiler Team's IREE and Triton backends, which are scheduled for Q2-Q3."

### Your Value
"I deliver all customer-facing components: benchmarks, documentation, migration guides, and production-ready serving infrastructure."

### Timeline Risk
"My Q2-Q3 deliverables depend entirely on Compiler Team's Q2-Q3 deliverables. Any slip in compiler work cascades to my work."

### Coordination Need
"I need clear coordination mechanisms with Compiler Team to ensure backends meet integration requirements."

---

**Last Updated:** January 2026  
**Status:** Scope corrected - Compiler work is separate team
