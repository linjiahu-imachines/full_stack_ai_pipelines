# VP 1:1 Slide Outline - Software Stack Director
**Prepared for:** 1:1 with VP Sergey  
**Date:** January 2026

---

## Slide 1: Executive Summary

**Overall Company Scope (2026):**
- Migrate AI workloads to RISC-V with custom extensions (MXFP4)
- Three parallel execution paths: Fast-Track, Production Pipeline, Cost-Optimizer
- Target: 50% TCO reduction vs NVIDIA
- Tape-out readiness: End of September 2026

**Software Stack Scope (My Ownership):**
- **Framework Integration:** LangChain, LlamaIndex, PyTorch/TensorFlow integration layers
- **Serving Stack:** vLLM platform plugin, llama.cpp, Ray orchestration
- **SDK Integration:** Framework wrappers, documentation, migration guides
- **Benchmarking:** MLPerf integration, performance validation
- **System Integration:** Multi-core, multi-node, robotics pipeline

**Critical Dependency (Compiler Team - work closely with):**
- **Compiler/Toolchain:** IREE, Triton backend, MLIR dialects, RISC-V GCC/Clang

**Key Message:** Software stack is the critical enabler for all three execution paths and customer adoption.

---

## Slide 2: Current State Assessment

### ✅ Completed (Foundation)
- **llama.cpp Integration:** Working via QEMU, framework wrappers (LangChain, LlamaIndex, Ray Serve)
- **Basic Service Layer:** FastAPI server, service wrappers
- **Development Infrastructure:** QEMU user mode, Synopsys cloud access

### ⚠️ In Progress
- **Ray Serve:** Basic integration done, needs multi-node and disaggregation
- **Multi-core Support:** Partial implementation, needs FPGA readiness

### ❌ Critical Gaps (Blocking Production)
- **IREE Integration:** 0% - Blocks Path 2 (Production Pipeline)
- **Triton Backend:** 0% - Blocks Path B (Custom Kernel Path)
- **vLLM Platform Plugin:** 0% - Blocks enterprise adoption
- **MLPerf Benchmarks:** 0% - Blocks customer demos
- **PyTorch/TensorFlow Compatibility:** 0% - Blocks customer migration

**Key Message:** Foundation is solid, but critical production components are missing.

---

## Slide 3: Software Stack Ownership & Scope

### Framework Integration Tier - PRIMARY OWNER
- **PyTorch/TensorFlow Integration:**
  - Integration layers between frameworks and RISC-V backend
  - Migration guides and documentation
  - Example models and tutorials
  - Coordinate with Compiler Team for backend readiness
- **LangChain/LlamaIndex:**
  - Framework wrappers (RISCVRISCLLM)
  - Agent workflows and RAG systems
  - Production optimization

### Compiler Tier (DEPENDENCY - Compiler Team)
- **IREE Backend (Compiler Team owns):**
  - MLIR dialect for custom RISC-V AI extensions
  - MXFP4 tensor identification and lowering
  - Data Tiling for RISC-V hardware
- **Triton Backend (Compiler Team owns):**
  - Triton-to-RISC-V compiler backend
  - MLIR lowering from Triton to RISC-V ISA
- **Your Coordination:** Ensure backends meet integration requirements

### Serving Tier (The "Engine") - PRIMARY OWNER
- **vLLM Platform Plugin:**
  - Custom memory managers and worker types
  - PagedAttention for RISC-V
  - KV-cache memory management
- **llama.cpp Enhancement:**
  - Native RISC-V binary (remove QEMU)
  - Multi-core support (4-6 cores)
  - MXFP4 quantization

### Orchestration Tier (The "Scale") - PRIMARY OWNER
- **Ray Cluster:**
  - Multi-node support
  - Disaggregated inference (prefill/decode split)
  - Hybrid NVIDIA/RISC-V clusters
- **Robotics Pipeline:**
  - Real-time inference service
  - Vision-language model support
  - Edge deployment patterns

### SDK & Compatibility - PRIMARY OWNER
- PyTorch `torch.compile` backend
- TensorFlow compatibility layer
- CUDA-equivalent SDK API
- Migration guides (minimal code changes)

### Benchmarking - PRIMARY OWNER (with PSim)
- MLPerf Inference integration
- NVIDIA comparison benchmarks
- Diversity benchmarks (CNNs, ViTs, VLMs)
- Performance analysis reports

**Key Message:** Software stack owns framework-to-deployment layers, working closely with Compiler Team for backend integration.

---

## Slide 4: Roadmap Alignment with CEO's Plan

### CEO's 4-Phase Plan → Software Stack Deliverables

| Phase | Timeline | CEO Milestone | Software Deliverable | Status |
|-------|----------|---------------|---------------------|--------|
| **Phase 1: Validation** | Q1 2026 | llama.cpp Demo | llama.cpp + GGML + RVV | ✅ **DONE** |
| **Phase 2: Portability** | Q2 2026 | Triton Backend | Triton + torch.compile | ⚠️ **START Q1** |
| **Phase 3: Scale** | Q3 2026 | vLLM Integration | vLLM + IREE (MXFP4) | ❌ **START Q2** |
| **Phase 4: Economy** | Q4 2026 | Ray Cluster | Ray + Disaggregation | ⚠️ **ENHANCE Q3** |

### CEO's Three Tracks → Software Components

| Track | Customer | Software Component | Status |
|-------|----------|-------------------|--------|
| **Track A: Enterprise** | Data centers (Llama-4) | vLLM + IREE + MXFP4 | ❌ **NOT STARTED** |
| **Track B: High-Perf Lab** | R&D (custom kernels) | Triton + MLIR | ❌ **NOT STARTED** |
| **Track C: Edge** | Robotics, Automotive | llama.cpp + RVV | ✅ **DONE** (needs enhancement) |

**Key Message:** Software stack is on track for Phase 1, but needs to accelerate Phases 2-4.

---

## Slide 5: Critical Path Items (2026)

### Q1 2026 (Jan-Mar) - FPGA Readiness
**Critical Path:**
- [ ] Multi-core support (4-6 cores for FPGA)
- [ ] FPGA deployment pipeline (local FPGA in Q1)
- [ ] Native RISC-V binary (remove QEMU for production)
- [ ] Performance baseline establishment

**Dependencies:** Hardware team (FPGA access, core specs)

### Q2 2026 (Apr-Jun) - Production Pipeline
**Critical Path:**
- [ ] IREE integration (MXFP4 support, MLIR dialects)
- [ ] Triton backend (custom kernel path)
- [ ] vLLM platform plugin (boilerplate + basic functionality)
- [ ] MLPerf benchmarks (initial results)
- [ ] PyTorch compatibility (torch.compile backend)
- [ ] Robotics pipeline (main components)

**Dependencies:** Hardware team (ISA spec, matrix unit details)

### Q3 2026 (Jul-Sep) - Scale & Optimization
**Critical Path:**
- [ ] vLLM full integration (production-ready)
- [ ] Ray multi-node cluster (disaggregated inference)
- [ ] MLPerf official results (customer-ready)
- [ ] Diversity benchmarks (CNNs, ViTs, VLMs)
- [ ] Compiler performance (dynamic shapes, sparsity)
- [ ] **Tape-out readiness** (SW stack runs on simulators/FPGA)

**Dependencies:** PSim team (performance validation), Hardware team (final specs)

**Key Message:** Q2-Q3 are critical - all production components must be ready for tape-out.

---

## Slide 6: Customer Questions → Deliverables Mapping

### Customer Question → Software Stack Deliverable

| Customer Question | Deliverable | Owner | Timeline |
|------------------|------------|-------|----------|
| **"Performance benchmarks vs NVIDIA?"** | MLPerf suite + NVIDIA comparison | SW Stack + PSim | Q2-Q3 |
| **"Official MLPerf results?"** | MLPerf submission + model list | SW Stack + PSim | Q3 |
| **"Code modification for PyTorch/TensorFlow?"** | Compatibility layer + migration guide | SW Stack | Q2-Q3 |
| **"CUDA-equivalent SDK?"** | SDK API + documentation | SW Stack | Q2-Q3 |
| **"Compiler handles dynamic shapes/sparsity?"** | IREE dynamic/sparse support | SW Stack | Q2-Q3 |
| **"Performance on CNNs/ViTs/VLMs?"** | Diversity benchmark suite | SW Stack | Q2-Q3 |

**Key Message:** All customer questions map to software stack deliverables - we are the customer-facing team.

---

## Slide 7: Dependencies & Risks

### Critical Dependencies

**Hardware Team:**
- [ ] Custom ISA specification (for IREE/Triton backends) - **URGENT**
- [ ] Matrix unit architecture details (for microkernels) - **URGENT**
- [ ] Memory architecture (HBM/SRAM) for PagedAttention
- [ ] FPGA availability (Q1 2026) - **URGENT**
- [ ] Performance targets (for benchmarking)

**PSim Team:**
- [ ] Performance simulation infrastructure
- [ ] MLPerf integration support
- [ ] Performance validation framework

### High-Risk Items

1. **IREE Integration Complexity**
   - Risk: More complex than anticipated
   - Mitigation: Start early (Q1), proof-of-concept first

2. **Timeline Compression**
   - Risk: Tape-out moved to Sept (from June), but plans unchanged
   - Mitigation: Prioritize critical path, parallel execution

3. **Performance Targets**
   - Risk: May not meet NVIDIA comparison targets
   - Mitigation: Establish baseline early, focus on TCO

**Key Message:** Success depends on timely hardware specs and PSim collaboration.

---

## Slide 8: Resource Needs & Questions

### Resource Needs
- **Team Size:** [To be discussed - depends on parallel execution paths]
- **External Dependencies:** IREE/Triton upstream engagement
- **Infrastructure:** FPGA access, simulation resources

### Questions for VP Sergey

**Strategic:**
1. Which parallel path should we prioritize if resources are constrained?
2. When do we need customer-ready demos? (affects MLPerf timeline)
3. What team size/resources are available?

**Technical:**
1. When will we have final ISA specification? (Critical for IREE/Triton)
2. What's the exact timeline for local FPGA availability?
3. What are the specific performance targets for customer demos?

**Process:**
1. How should we coordinate with PSim team on MLPerf?
2. Are we contributing IREE/Triton backends upstream or keeping proprietary?
3. Will we have early customer feedback to guide priorities?

**Key Message:** Need clarity on priorities, resources, and dependencies to execute effectively.

---

## Slide 9: Next Steps & Action Items

### Immediate (This Week)
- [ ] Review CEO's plan in detail
- [ ] Prepare feedback on CEO's presentation
- [ ] Finalize this slide deck
- [ ] Identify specific resource needs

### Short-term (This Month)
- [ ] Start IREE integration proof-of-concept
- [ ] Study vLLM platform plugin architecture
- [ ] Engage with PSim team on MLPerf integration
- [ ] Coordinate with Hardware team on ISA/memory specs

### Q1 2026 (Jan-Mar)
- [ ] Multi-core support implementation
- [ ] FPGA deployment pipeline
- [ ] Native RISC-V binary development
- [ ] Performance baseline establishment

**Key Message:** Ready to execute, but need dependencies resolved and priorities confirmed.

---

## Key Talking Points Summary

### What I Own
- Entire software stack from compiler to deployment
- All customer-facing deliverables (benchmarks, SDK, compatibility)
- Critical path for all three execution paths

### What I Need
- Hardware specs (ISA, matrix units, memory) - **URGENT**
- FPGA access timeline - **URGENT**
- PSim collaboration on MLPerf
- Resource allocation clarity

### Critical Success Factors
1. IREE integration (foundation for production)
2. vLLM plugin (enables enterprise adoption)
3. MLPerf benchmarks (required for customers)
4. Multi-core support (required for FPGA)

### Risks
- Timeline compression (tape-out moved, plans unchanged)
- Integration complexity (IREE/vLLM)
- Performance target achievement
- Resource allocation for parallel paths

---

**Slide Count:** 9 slides (recommended for 30-45 min meeting)  
**Focus:** Scope clarity, dependencies, risks, and action items
