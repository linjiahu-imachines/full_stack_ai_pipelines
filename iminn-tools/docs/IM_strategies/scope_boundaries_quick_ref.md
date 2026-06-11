# Software Stack Scope Boundaries - Quick Reference
**For:** Director of Software Stack  
**Purpose:** Quick reference for scope clarity

---

## ✅ IN SCOPE (Your Ownership)

### Framework Integration
- ✅ LangChain integration (RISCVRISCLLM wrapper)
- ✅ LlamaIndex integration (RISCVRISCLLM wrapper)
- ✅ PyTorch integration layer (coordinate with Compiler Team)
- ✅ TensorFlow integration layer (coordinate with Compiler Team)
- ✅ Framework-specific optimizations and examples

### Serving Stack
- ✅ vLLM platform plugin development
- ✅ llama.cpp enhancements (native RISC-V, multi-core)
- ✅ PagedAttention implementation for RISC-V
- ✅ KV-cache memory management

### Orchestration & Deployment
- ✅ Ray cluster integration (multi-node, disaggregation)
- ✅ Robotics pipeline components
- ✅ Edge deployment patterns
- ✅ Hybrid NVIDIA/RISC-V clusters

### SDK Integration & Documentation
- ✅ SDK documentation and tutorials (coordinate with Compiler Team)
- ✅ Migration guides (CUDA to RISC-V)
- ✅ Example code and best practices
- ✅ Framework integration guides

### Benchmarking & Performance
- ✅ MLPerf Inference integration
- ✅ NVIDIA comparison benchmarks
- ✅ Diversity benchmarks (CNNs, ViTs, VLMs)
- ✅ Performance analysis and reporting

### System Integration
- ✅ Multi-core support (4-6 cores)
- ✅ Multi-node system support
- ✅ FPGA deployment pipeline
- ✅ Simulator integration

---

## ❌ OUT OF SCOPE (Other Teams)

### Compiler Team
- ❌ IREE compiler backend implementation
- ❌ Triton-to-RISC-V backend implementation
- ❌ MLIR dialect development for custom RISC-V ISA
- ❌ Microkernel (ukernels) implementation
- ❌ Data Tiling implementation
- ❌ CUDA-equivalent SDK core implementation
- ❌ RISC-V toolchain (GCC/Clang)

**Your Coordination:** Work closely with Compiler Team to ensure backends meet integration requirements

### Hardware Team
- ❌ Custom ISA design and specification
- ❌ Matrix unit hardware implementation
- ❌ Memory architecture (HBM/SRAM) design
- ❌ FPGA hardware availability
- ❌ Performance targets definition

**Your Dependency:** Need ISA spec, matrix unit details, memory architecture for software implementation

### PSim Team
- ❌ Performance simulation infrastructure
- ❌ Performance validation framework
- ❌ MLPerf infrastructure setup

**Your Dependency:** Need simulation infrastructure and validation framework for benchmarking

### Hardware/System Team
- ❌ FPGA hardware procurement and setup
- ❌ Physical hardware testing
- ❌ Hardware-in-the-loop infrastructure

**Your Dependency:** Need FPGA access and hardware specs for software development

---

## ⚠️ SHARED OWNERSHIP (Collaboration Required)

### Performance Benchmarks
- **Software Stack:** Benchmark implementation, model integration, results analysis
- **PSim Team:** Simulation infrastructure, performance validation
- **Hardware Team:** Performance targets, hardware specs

### MLPerf Integration
- **Software Stack:** MLPerf benchmark suite integration, model preparation
- **PSim Team:** MLPerf infrastructure, performance simulation
- **Hardware Team:** Performance targets, hardware validation

### Tape-out Readiness
- **Software Stack:** SW stack runs on simulators/FPGA, performance validation
- **Hardware Team:** Hardware tape-out, FPGA availability
- **PSim Team:** Performance simulation and validation

---

## 🔑 KEY DEPENDENCIES (Blocking Items)

### From Compiler Team (URGENT)
1. **IREE Backend for RISC-V**
   - Needed for: vLLM integration, PyTorch/TensorFlow support
   - Timeline: Q2-Q3 2026 (critical blocker)

2. **Triton Backend for RISC-V**
   - Needed for: Custom kernel support, framework flexibility
   - Timeline: Q2 2026 (critical blocker)

3. **CUDA-equivalent SDK Core**
   - Needed for: Developer experience, migration guides
   - Timeline: Q2-Q3 2026

**Note:** Compiler Team work is critical blocker for most of your deliverables

### From Hardware Team (URGENT)
1. **Custom ISA Specification**
   - Needed for: IREE MLIR dialects, Triton backend
   - Timeline: ASAP (blocks Q2 deliverables)

2. **Matrix Unit Architecture Details**
   - Needed for: Microkernels (ukernels), Triton backend
   - Timeline: ASAP (blocks Q2 deliverables)

3. **Memory Architecture (HBM/SRAM)**
   - Needed for: PagedAttention, vLLM memory management
   - Timeline: Q1 2026

4. **FPGA Availability**
   - Needed for: Multi-core testing, production validation
   - Timeline: Q1 2026 (mentioned in VP email)

5. **Performance Targets**
   - Needed for: Benchmarking goals, customer demos
   - Timeline: Q1 2026

### From PSim Team
1. **Performance Simulation Infrastructure**
   - Needed for: MLPerf benchmarks, performance validation
   - Timeline: Q2 2026

2. **MLPerf Integration Support**
   - Needed for: Official MLPerf results
   - Timeline: Q2-Q3 2026

---

## 📊 SCOPE SUMMARY

**Your Primary Ownership:**
- Framework integration layer (LangChain, LlamaIndex, PyTorch, TensorFlow)
- Serving/orchestration layer (vLLM, llama.cpp, Ray Serve)
- Customer-facing deliverables (benchmarks, docs, migration guides)
- System integration (multi-core, multi-node, robotics)

**Your Dependencies:**
- **Compiler Team:** IREE backend, Triton backend, SDK core - **CRITICAL BLOCKER**
- Hardware Team: ISA specs, matrix units, FPGA access - **URGENT**
- PSim Team: Performance infrastructure, validation

**Your Deliverables:**
- Production-ready software stack
- Customer-ready benchmarks and demos
- Migration guides and SDK documentation

---

## 🎯 CRITICAL SUCCESS FACTORS

1. **Compiler Team Deliverables** - IREE/Triton backends (critical blocker for all your work)
2. **vLLM Plugin** - Enables enterprise adoption (depends on IREE)
3. **MLPerf Benchmarks** - Required for customer demos (depends on compiler + PSim)
4. **Multi-core Support** - Required for FPGA readiness
5. **Framework Integration** - PyTorch/TensorFlow (depends on IREE backend)

**Your work is blocked by Compiler Team deliverables (Q2-Q3) and hardware specs (Q1).**

---

**Last Updated:** January 2026  
**Status:** Quick reference for VP 1:1 preparation
