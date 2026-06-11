# VP 1:1 Scope Analysis - Software Stack Director
**Prepared for:** 1:1 with VP Sergey  
**Date:** January 2026  
**Purpose:** Define overall scope and software stack ownership boundaries

---

## Executive Summary

**Overall Company Scope (2026):**
- Migrate AI workloads to RISC-V processor with custom extensions (MXFP4 support)
- Deliver production-grade enterprise ecosystem with three parallel execution paths
- Demonstrate competitive performance vs NVIDIA (50% TCO reduction target)
- Support tape-out readiness by end of September 2026

**Software Stack Scope (Your Ownership):**
- **Framework Integration:** LangChain, LlamaIndex, PyTorch/TensorFlow integration layers
- **Serving Stack:** vLLM platform plugin, llama.cpp integration, Ray orchestration
- **SDK Integration:** PyTorch/TensorFlow compatibility layers, framework wrappers
- **Benchmarking:** MLPerf integration, performance validation, customer demos
- **System Integration:** Multi-core, multi-node support, robotics pipeline

**Critical Dependency (Compiler Team - separate team you work with):**
- **Compiler/Toolchain:** IREE, Triton backend, MLIR dialects, RISC-V GCC/Clang

---

## I. Overall Company Scope (2026 Roadmap)

### A. Three Parallel Execution Paths (CEO's Strategy)

#### **Path 1: Fast-Track (llama.cpp) - Immediate PoC**
- **Goal:** Proof-of-concept by manually mapping math to RISC-V intrinsics
- **Timeline:** Immediate (Q1 2026)
- **Deliverable:** Working llama.cpp demo on RISC-V
- **Owner:** Software Stack (with Hardware team)
- **Status:** ✅ **COMPLETE** (Phase 1-5 done: llama.cpp + QEMU working)

#### **Path 2: Production Pipeline (vLLM + IREE) - Scalable Path**
- **Goal:** Automated compiler handling complex models (Llama-4, GPT-OSS)
- **Timeline:** Q2-Q3 2026
- **Deliverable:** vLLM platform plugin + IREE MXFP4 support
- **Owner:** **Software Stack (PRIMARY OWNER)**
- **Status:** ⚠️ **IN PROGRESS** (llama.cpp done, vLLM/IREE integration needed)

#### **Path 3: Cost-Optimizer (Ray + Disaggregation) - Enterprise Scale**
- **Goal:** Offload expensive NVIDIA decode tasks to efficient RISC-V clusters
- **Timeline:** Q3-Q4 2026
- **Deliverable:** Ray cluster with disaggregated prefill/decode
- **Owner:** **Software Stack (PRIMARY OWNER)**
- **Status:** ⚠️ **PARTIAL** (Ray Serve integration done, disaggregation needed)

### B. Critical Milestones (VP Sergey's Requirements)

| Milestone | Timeline | Owner | Software Stack Involvement |
|-----------|----------|-------|---------------------------|
| **Tape-out Ready** | End of Sept 2026 | Hardware | ✅ SW stack must run on simulators/FPGA |
| **FPGA Availability** | Q1 2026 (4-6 cores) | Hardware | ✅ SW stack must support multi-core |
| **Performance Demos** | Q2-Q3 2026 | PSim + SW Stack | ✅ **PRIMARY OWNER** - MLPerf benchmarks |
| **AI SDK Demo** | Q2 2026 | SW Stack | ✅ **PRIMARY OWNER** - IREE + frameworks |
| **Robotics Pipeline** | Q2-Q3 2026 | SW Stack | ✅ **PRIMARY OWNER** - Main components |
| **Multi-node System** | Q3 2026 | SW Stack | ✅ **PRIMARY OWNER** - Ray on simulators |

---

## II. Software Stack Scope (Your Ownership)

### A. Compiler Tier (DEPENDENCY - Separate Compiler Team)

**Note:** Compiler/toolchain work is owned by a **separate Compiler Team** that you work closely with, but do not directly own.

#### **1. IREE Integration (COMPILER TEAM - DEPENDENCY)**
**Status:** ⚠️ **NOT STARTED** - Critical dependency

**Compiler Team Deliverables:**
- [ ] IREE MLIR dialect for custom RISC-V AI extensions
- [ ] MXFP4 tensor identification and lowering pipeline
- [ ] Data Tiling implementation for RISC-V hardware
- [ ] Microkernel library (ukernels) for matrix operations
- [ ] Integration with PyTorch `torch.compile` workflow

**Your Software Stack Dependency:**
- Need IREE backend working to enable vLLM, PyTorch integration
- Need to coordinate on API/interface requirements
- Need to validate end-to-end workflows

**Timeline:** Q2-Q3 2026 (critical path for your production pipeline)

#### **2. Triton Backend (COMPILER TEAM - DEPENDENCY)**
**Status:** ⚠️ **NOT STARTED** - Critical dependency

**Compiler Team Deliverables:**
- [ ] Triton compiler backend for RISC-V
- [ ] MLIR lowering from Triton to RISC-V ISA
- [ ] Test suite with common Triton kernels
- [ ] Migration guide for NVIDIA Triton code

**Your Software Stack Dependency:**
- Need Triton backend for custom kernel support in applications
- Need to coordinate on kernel integration patterns
- Need to validate with real-world use cases

**Timeline:** Q2 2026 (critical dependency for your framework integrations)

### B. Framework Integration Tier (PRIMARY OWNER)

#### **1. PyTorch/TensorFlow Integration (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Customer requirement

**Requirements (from customer questions):**
- ✅ "How much code modification is required to run standard PyTorch or TensorFlow models?"
- ✅ Integration layers between frameworks and RISC-V backend
- ✅ Coordinate with Compiler Team for backend readiness

**Deliverables:**
- [ ] PyTorch integration layer (coordinate with Compiler Team's IREE backend)
- [ ] TensorFlow compatibility layer
- [ ] Migration guide (minimal code changes)
- [ ] Example models (CNNs, ViTs, VLMs)
- [ ] Framework-specific optimizations

**Dependencies:**
- Compiler Team: IREE backend, Triton backend (critical blocker)
- Hardware team: Performance targets

**Timeline:** Q2-Q3 2026

#### **2. LangChain/LlamaIndex Integration (PRIMARY OWNER)**
**Status:** ✅ **COMPLETE** - Phase 3 & 5 done

**Current State:**
- ✅ RISCVRISCLLM wrapper for LangChain
- ✅ RISCVRISCLLM wrapper for LlamaIndex
- ✅ Agent workflows, RAG systems

**Enhancements Needed:**
- [ ] Production optimization
- [ ] Advanced agent features
- [ ] Multi-modal support

**Timeline:** Q2 2026 (enhancements)

### C. Serving Tier (The "Engine")

#### **1. vLLM Platform Plugin (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Critical for enterprise adoption

**Requirements:**
- ✅ Implement `vllm-riscv` platform plugin
- ✅ Define custom memory managers and worker types
- ✅ Implement PagedAttention memory logic in RISC-V driver
- ✅ Support KV-cache memory management (non-contiguous)
- ✅ Link to IREE-compiled libraries or Triton kernels

**Deliverables:**
- [ ] vLLM platform plugin boilerplate code
- [ ] RISC-V memory worker implementation
- [ ] PagedAttention support for RISC-V
- [ ] Integration with IREE-compiled MXFP4 kernels
- [ ] Test suite with Llama-4 and GPT-OSS models

**Dependencies:**
- IREE integration (compiler tier)
- Hardware team: Memory architecture, HBM/SRAM details

**Timeline:** Q2-Q3 2026 (critical path for Path 2: Production Pipeline)

#### **2. llama.cpp Integration (ENHANCEMENT)**
**Status:** ✅ **COMPLETE** (Phase 1-5) - Needs enhancement for production

**Current State:**
- ✅ Basic llama.cpp execution via QEMU user mode
- ✅ Framework integrations (LangChain, LlamaIndex, Ray Serve)
- ✅ FastAPI server

**Enhancements Needed:**
- [ ] Native RISC-V binary (remove QEMU dependency)
- [ ] Multi-core support (4-6 cores for FPGA)
- [ ] MXFP4 quantization support
- [ ] Performance optimization (RVV intrinsics)
- [ ] Production deployment patterns

**Timeline:** Q1-Q2 2026 (FPGA readiness)

### D. Orchestration Tier (The "Scale")

#### **1. Ray Cluster Integration (PRIMARY OWNER)**
**Status:** ⚠️ **PARTIAL** (Phase 4 complete, needs enhancement)

**Current State:**
- ✅ Ray Serve integration with llama.cpp (Phase 4)
- ✅ Horizontal scaling with replicas
- ✅ Load balancing

**Enhancements Needed:**
- [ ] Multi-node Ray cluster support
- [ ] Disaggregated inference (prefill on NVIDIA, decode on RISC-V)
- [ ] Ray Placement Groups for RISC-V nodes
- [ ] Hybrid cluster orchestration
- [ ] Performance optimization for multi-node

**Deliverables:**
- [ ] Ray cluster configuration for RISC-V nodes
- [ ] Disaggregated inference pipeline
- [ ] Hybrid NVIDIA/RISC-V cluster demo
- [ ] Cost analysis vs pure NVIDIA cluster

**Timeline:** Q3 2026 (Path 3: Cost-Optimizer)

#### **2. Robotics Pipeline (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - VP requirement

**Requirements:**
- ✅ Show main components of robotics pipeline
- ✅ Support real-time inference (latency-critical)
- ✅ Multi-modal processing (vision + language)
- ✅ Edge deployment patterns

**Deliverables:**
- [ ] Robotics pipeline architecture
- [ ] Real-time inference service
- [ ] Vision-language model (VLM) support
- [ ] Edge deployment guide

**Timeline:** Q2-Q3 2026

### E. Model Format & SDK Support

#### **1. Model Format Support (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED**

**Requirements:**
- ✅ MXFP4 quantization support
- ✅ GGUF format (current)
- ✅ ONNX compatibility (optional)
- ✅ Model conversion tools

**Deliverables:**
- [ ] MXFP4 quantization toolkit
- [ ] Model conversion pipeline (PyTorch → MXFP4)
- [ ] Format validation tools

**Timeline:** Q2 2026

#### **2. SDK Integration & Documentation (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Customer requirement

**Requirements (from customer questions):**
- ✅ "Do you have a CUDA-equivalent SDK?"
- ✅ SDK documentation and examples
- ✅ Developer experience similar to CUDA

**Deliverables:**
- [ ] SDK documentation (coordinate with Compiler Team)
- [ ] Example code and tutorials
- [ ] Migration guide from CUDA
- [ ] API reference documentation

**Dependencies:**
- Compiler Team: CUDA-equivalent SDK implementation

**Timeline:** Q2-Q3 2026

### F. Benchmarking & Performance Validation

#### **1. MLPerf Integration (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Critical customer requirement

**Requirements (from customer questions):**
- ✅ "What are your official MLPerf results?"
- ✅ "Which specific models were used for your internal performance claims?"
- ✅ "Performance benchmarks and comparisons between NVIDIA and your product"

**Deliverables:**
- [ ] MLPerf Inference benchmark suite integration
- [ ] Performance results for key models (Llama-4, GPT-OSS, CNNs, ViTs)
- [ ] NVIDIA comparison benchmarks
- [ ] Official MLPerf submission (if applicable)
- [ ] Performance analysis reports

**Dependencies:**
- PSim team: Performance simulation infrastructure
- Hardware team: Performance targets

**Timeline:** Q2-Q3 2026 (critical for customer demos)

#### **2. Diversity Benchmarks (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Customer requirement

**Requirements (from customer questions):**
- ✅ "Beyond LLMs, how does your architecture perform across diverse AI domains?"
- ✅ "Computer Vision (CNNs/ViTs) performance"
- ✅ "Vision-Language Models (VLMs) performance"
- ✅ "How does your hardware handle spatial data locality, 2D/3D convolutions?"
- ✅ "Heterogeneous data processing required for multi-modal fusion"

**Deliverables:**
- [ ] CNN benchmark suite (ResNet, EfficientNet, etc.)
- [ ] Vision Transformer (ViT) benchmarks
- [ ] Vision-Language Model (VLM) benchmarks
- [ ] Spatial convolution performance analysis
- [ ] Multi-modal fusion performance analysis

**Timeline:** Q2-Q3 2026

### G. System Integration & Infrastructure

#### **1. Multi-Core Support (PRIMARY OWNER)**
**Status:** ⚠️ **IN PROGRESS** - FPGA requirement

**Requirements:**
- ✅ Support 4-6 cores on FPGA (Q1 2026)
- ✅ Multi-threaded inference
- ✅ Core-to-core communication
- ✅ Load balancing across cores

**Deliverables:**
- [ ] Multi-core llama.cpp implementation
- [ ] Threading model for RISC-V
- [ ] Core affinity and scheduling
- [ ] Performance scaling analysis

**Timeline:** Q1 2026 (FPGA readiness)

#### **2. Simulator & FPGA Support (PRIMARY OWNER)**
**Status:** ⚠️ **PARTIAL** (QEMU working, FPGA needed)

**Requirements:**
- ✅ Run SW stack on Synopsys cloud (current)
- ✅ Run SW stack on local FPGA (Q1 2026)
- ✅ Performance validation on simulators
- ✅ Hardware-in-the-loop testing

**Deliverables:**
- [ ] FPGA deployment pipeline
- [ ] Simulator integration (PSim collaboration)
- [ ] Performance validation framework
- [ ] Hardware-in-the-loop test suite

**Timeline:** Q1-Q2 2026

#### **3. Compiler Performance (PRIMARY OWNER)**
**Status:** ⚠️ **NOT STARTED** - Customer requirement

**Requirements (from customer questions):**
- ✅ "How efficiently does your graph compiler handle dynamic shapes?"
- ✅ "How efficiently does your graph compiler handle sparsity?"
- ✅ "Without manual optimization?"

**Deliverables:**
- [ ] Dynamic shape handling in IREE
- [ ] Sparse tensor support
- [ ] Automatic optimization pipeline
- [ ] Performance comparison (manual vs automatic)

**Timeline:** Q2-Q3 2026

---

## III. Gap Analysis: Current State vs Requirements

### ✅ What We Have (Completed)

1. **Basic Inference Stack:**
   - ✅ llama.cpp integration via QEMU
   - ✅ Framework wrappers (LangChain, LlamaIndex, Ray Serve)
   - ✅ FastAPI server
   - ✅ Basic service layer

2. **Development Infrastructure:**
   - ✅ QEMU user mode working
   - ✅ Synopsys cloud access
   - ✅ Build and test infrastructure

### ⚠️ Critical Gaps (Must Address)

1. **Compiler Stack (HIGHEST PRIORITY):**
   - ❌ IREE integration (0% complete)
   - ❌ Triton backend (0% complete)
   - ❌ MXFP4 support (0% complete)
   - **Impact:** Blocks Path 2 (Production Pipeline)

2. **Serving Stack:**
   - ❌ vLLM platform plugin (0% complete)
   - ⚠️ llama.cpp needs native RISC-V (currently QEMU only)
   - **Impact:** Blocks enterprise adoption

3. **Benchmarking:**
   - ❌ MLPerf integration (0% complete)
   - ❌ NVIDIA comparisons (0% complete)
   - ❌ Diversity benchmarks (0% complete)
   - **Impact:** Blocks customer demos

4. **System Integration:**
   - ⚠️ Multi-core support (partial)
   - ❌ FPGA deployment (0% complete)
   - ❌ Multi-node Ray cluster (0% complete)
   - **Impact:** Blocks tape-out readiness

5. **SDK & Compatibility:**
   - ❌ PyTorch/TensorFlow compatibility (0% complete)
   - ❌ CUDA-equivalent SDK (0% complete)
   - **Impact:** Blocks customer adoption

---

## IV. Roadmap Alignment with CEO's Plan

### CEO's 4-Phase Migration Roadmap → Software Stack Deliverables

| CEO Phase | Timeline | CEO Milestone | Software Stack Deliverable | Status |
|-----------|----------|--------------|---------------------------|--------|
| **Phase 1: Validation** | Q1 2026 | llama.cpp Demo | ✅ llama.cpp + GGML + RVV Intrinsics | ✅ **DONE** |
| **Phase 2: Portability** | Q2 2026 | Triton Backend | ⚠️ Triton + torch.compile | ⚠️ **IN PROGRESS** |
| **Phase 3: Scale** | Q3 2026 | vLLM Integration | ❌ vLLM + IREE (Native MXFP4) | ❌ **NOT STARTED** |
| **Phase 4: Economy** | Q4 2026 | Ray Cluster | ⚠️ Ray Serve + Disaggregated Prefill | ⚠️ **PARTIAL** |

### CEO's Three Tracks → Software Stack Components

| Track | Customer Profile | Software Stack Component | Status |
|-------|-----------------|-------------------------|--------|
| **Track A: Enterprise LLM Fleet** | Data centers (Llama-4/DeepSeek) | vLLM + IREE + MXFP4 | ❌ **NOT STARTED** |
| **Track B: High-Performance Lab** | R&D teams (proprietary kernels) | Triton + MLIR + RISC-V Extensions | ❌ **NOT STARTED** |
| **Track C: Latency-Critical Edge** | Robotics, Automotive | llama.cpp + GGML + RVV | ✅ **DONE** (needs enhancement) |

---

## V. Key Deliverables & Milestones (2026)

### Q1 2026 (Jan-Mar) - FPGA Readiness

**Critical Path:**
- [ ] **Multi-core support** (4-6 cores for FPGA)
- [ ] **FPGA deployment pipeline** (local FPGA in Q1)
- [ ] **Native RISC-V binary** (remove QEMU dependency for production)
- [ ] **Performance baseline** (establish metrics)

**Owner:** Software Stack (with Hardware team)

### Q2 2026 (Apr-Jun) - Production Pipeline

**Critical Path:**
- [ ] **IREE integration** (MXFP4 support, MLIR dialects)
- [ ] **Triton backend** (custom kernel path)
- [ ] **vLLM platform plugin** (boilerplate + basic functionality)
- [ ] **MLPerf benchmarks** (initial results)
- [ ] **PyTorch compatibility** (torch.compile backend)
- [ ] **Robotics pipeline** (main components)

**Owner:** Software Stack (PRIMARY)

### Q3 2026 (Jul-Sep) - Scale & Optimization

**Critical Path:**
- [ ] **vLLM full integration** (production-ready)
- [ ] **Ray multi-node cluster** (disaggregated inference)
- [ ] **MLPerf official results** (customer-ready)
- [ ] **Diversity benchmarks** (CNNs, ViTs, VLMs)
- [ ] **Compiler performance** (dynamic shapes, sparsity)
- [ ] **Tape-out readiness** (SW stack runs on simulators/FPGA)

**Owner:** Software Stack (PRIMARY)

### Q4 2026 (Oct-Dec) - Enterprise Scale

**Critical Path:**
- [ ] **Cost optimization** (50% TCO reduction validation)
- [ ] **Enterprise demos** (customer-ready)
- [ ] **SDK documentation** (CUDA-equivalent)
- [ ] **Migration guides** (PyTorch/TensorFlow)

**Owner:** Software Stack (PRIMARY)

---

## VI. Customer Questions → Deliverables Mapping

### Question 1: "Performance benchmarks and comparisons between NVIDIA and your product"
**Deliverable:** MLPerf benchmark suite + NVIDIA comparison reports  
**Owner:** Software Stack (with PSim team)  
**Timeline:** Q2-Q3 2026

### Question 2: "Official MLPerf results, and which specific models were used"
**Deliverable:** MLPerf submission + model list documentation  
**Owner:** Software Stack (with PSim team)  
**Timeline:** Q3 2026

### Question 3: "How much code modification is required to run standard PyTorch or TensorFlow models?"
**Deliverable:** PyTorch/TensorFlow compatibility layer + migration guide  
**Owner:** Software Stack (PRIMARY)  
**Timeline:** Q2-Q3 2026

### Question 4: "Do you have a CUDA-equivalent SDK?"
**Deliverable:** CUDA-equivalent SDK API + documentation  
**Owner:** Software Stack (PRIMARY)  
**Timeline:** Q2-Q3 2026

### Question 5: "How efficiently does your graph compiler handle dynamic shapes and sparsity without manual optimization?"
**Deliverable:** IREE dynamic shape/sparsity support + performance analysis  
**Owner:** Software Stack (PRIMARY)  
**Timeline:** Q2-Q3 2026

### Question 6: "Beyond LLMs, how does your architecture perform across diverse AI domains (CNNs/ViTs/VLMs)?"
**Deliverable:** Diversity benchmark suite (CNNs, ViTs, VLMs) + performance reports  
**Owner:** Software Stack (PRIMARY)  
**Timeline:** Q2-Q3 2026

---

## VII. Dependencies & Cross-Team Collaboration

### Hardware Team Dependencies
- [ ] Custom ISA specification (for IREE/Triton backends)
- [ ] Matrix unit architecture details (for microkernels)
- [ ] Memory architecture (HBM/SRAM) for PagedAttention
- [ ] FPGA availability (Q1 2026)
- [ ] Performance targets (for benchmarking)

### PSim Team Dependencies
- [ ] Performance simulation infrastructure
- [ ] Benchmark execution environment
- [ ] Performance validation framework
- [ ] MLPerf integration support

### IREE/Triton Upstream
- [ ] IREE MXFP4 support (✅ available in Jan 2026 release)
- [ ] Triton backend architecture documentation
- [ ] Upstream contributions (if needed)

---

## VIII. Risk Assessment

### High Risk Items

1. **IREE Integration Complexity**
   - **Risk:** IREE integration may be more complex than anticipated
   - **Mitigation:** Start early (Q1), engage with IREE community, proof-of-concept first

2. **vLLM Platform Plugin Development**
   - **Risk:** vLLM architecture may require significant customization
   - **Mitigation:** Study vLLM codebase early, create minimal viable plugin first

3. **Performance Targets**
   - **Risk:** May not meet NVIDIA comparison targets
   - **Mitigation:** Establish baseline early, iterate on optimization, focus on TCO not just raw performance

4. **Timeline Compression**
   - **Risk:** Tape-out moved to Sept (from June), but plans remain same
   - **Mitigation:** Prioritize critical path items, parallel execution where possible

### Medium Risk Items

1. **Multi-core Support**
   - **Risk:** Threading model may need significant rework
   - **Mitigation:** Leverage existing llama.cpp threading, test early on FPGA

2. **MLPerf Integration**
   - **Risk:** MLPerf may require significant infrastructure work
   - **Mitigation:** Start with subset of benchmarks, leverage PSim team

---

## IX. Recommended Slide Structure for VP 1:1

### Slide 1: Executive Summary
- Overall scope overview
- Software stack ownership boundaries
- Key milestones alignment

### Slide 2: Current State Assessment
- ✅ What's complete (llama.cpp, frameworks)
- ⚠️ What's in progress (Ray Serve)
- ❌ Critical gaps (IREE, vLLM, MLPerf)

### Slide 3: Software Stack Scope (Your Ownership)
- Compiler tier (IREE, Triton)
- Serving tier (vLLM, llama.cpp)
- Orchestration tier (Ray, robotics)
- SDK & compatibility
- Benchmarking

### Slide 4: Roadmap Alignment
- CEO's 4-phase plan → Software deliverables
- Three tracks → Software components
- Timeline alignment

### Slide 5: Critical Path Items (Q1-Q3)
- Q1: FPGA readiness (multi-core, native binary)
- Q2: Production pipeline (IREE, vLLM, MLPerf)
- Q3: Scale & optimization (Ray cluster, diversity benchmarks)

### Slide 6: Customer Questions → Deliverables
- Map each customer question to specific deliverable
- Timeline and ownership

### Slide 7: Dependencies & Risks
- Hardware team dependencies
- PSim team dependencies
- Risk mitigation strategies

### Slide 8: Resource Needs
- Team size requirements
- External dependencies
- Budget considerations (if applicable)

---

## X. Key Talking Points for VP 1:1

### What You Own (Software Stack)
1. **Compiler Stack:** IREE, Triton, MLIR dialects, microkernels
2. **Serving Stack:** vLLM plugin, llama.cpp enhancements, Ray orchestration
3. **SDK & Frameworks:** PyTorch/TensorFlow compatibility, CUDA-equivalent SDK
4. **Benchmarking:** MLPerf integration, performance validation, customer demos
5. **System Integration:** Multi-core, multi-node, robotics pipeline

### What You Need (Dependencies)
1. **Hardware Team:** ISA spec, matrix unit details, memory architecture, FPGA access
2. **PSim Team:** Performance simulation infrastructure, MLPerf support
3. **Timeline Clarity:** Confirm tape-out readiness requirements

### Critical Success Factors
1. **IREE Integration:** Foundation for production pipeline (Path 2)
2. **vLLM Plugin:** Enables enterprise adoption
3. **MLPerf Benchmarks:** Required for customer demos
4. **Multi-core Support:** Required for FPGA readiness (Q1)

### Risks to Discuss
1. Timeline compression (tape-out moved to Sept, but plans unchanged)
2. IREE/vLLM integration complexity
3. Performance target achievement
4. Resource allocation for parallel execution paths

---

## XI. Action Items (Post-Meeting)

### Immediate Actions (This Week)
- [ ] Review CEO's plan in detail
- [ ] Prepare feedback on CEO's presentation
- [ ] Finalize slide deck for VP 1:1
- [ ] Identify specific resource needs

### Short-term Actions (This Month)
- [ ] Start IREE integration proof-of-concept
- [ ] Study vLLM platform plugin architecture
- [ ] Engage with PSim team on MLPerf integration
- [ ] Coordinate with Hardware team on ISA/memory specs

### Medium-term Actions (Q1 2026)
- [ ] Multi-core support implementation
- [ ] FPGA deployment pipeline
- [ ] Native RISC-V binary development
- [ ] Performance baseline establishment

---

## XII. Questions for VP Sergey

### Strategic Questions
1. **Priority Clarification:** Which of the three parallel paths (Fast-Track, Production Pipeline, Cost-Optimizer) should we prioritize if resources are constrained?
2. **Customer Timeline:** When do we need customer-ready demos? This affects MLPerf timeline.
3. **Resource Allocation:** What team size/resources are available for software stack development?

### Technical Questions
1. **Hardware Specs:** When will we have final ISA specification and matrix unit details? (Critical for IREE/Triton)
2. **FPGA Access:** What's the exact timeline for local FPGA availability? (Q1 mentioned, but need specifics)
3. **Performance Targets:** What are the specific performance targets we need to hit for customer demos?

### Process Questions
1. **Collaboration Model:** How should we coordinate with PSim team on MLPerf benchmarks?
2. **Upstream Contributions:** Are we contributing IREE/Triton backends upstream, or keeping them proprietary?
3. **Customer Engagement:** Will we have early customer feedback to guide development priorities?

---

**Document Status:** Draft for VP 1:1 preparation  
**Next Steps:** Review, refine, prepare slides based on this analysis
