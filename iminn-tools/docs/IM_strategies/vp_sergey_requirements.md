# VP Sergey Requirements - Email Documentation
**From:** VP Sergey  
**Date:** January 2026  
**Subject:** Status & Plans for 2026 + Customer Questions  
**Action Required:** Prepare slides for CEO Mohammad meeting (next Tuesday)

---

## Email Content (Original)

Hi Team,

Per Mohammad's request, please prepare a couple of slides (each of you) with the status and plans for this year.

This is for the next meeting with Mohammad (next Tuesday), but don't delay and we can discuss your slides on 1:1's this week or next Monday (whenever you have your slides ready).

Plus, read carefully what Mohammad presented at today's meeting and prepare your feedback.

Just for the reference.

We have our tape-out scheduled for end of September (instead of end of June, not our fault), but our plans remain the same and we must be ready to run our SW stack before that on our simulators and on FPGA (we should have FPGA locally in Q1, four cores will be supported maybe six).

But even right now we can run SW in the Synopsys cloud.

And we will need to demonstrate performance results (in collaboration with the PSim team) on a number of AI benchmarks (such as MLPerf, which our potential customers would like to see) that we need to define altogether.

Additionally, we will need to demonstrate our AI SDK by, at a minimum, showing how major AI frameworks can effectively use our hardware (including CUDA-like workloads) using the IREE compiler.

Plus, show robotics pipeline (at least main components) and AI system prototype for our multicore & multi-node system (e.g. Ray, on simulators first).

Some questions from our potential customers that we have to address:

1. Would you please share the performance benchmarks and comparisons between NVIDIA and your product?
2. Standard Benchmarks: What are your official MLPerf results, and which specific models were used for your internal performance claims?
3. Software Ecosystem: How much code modification is required to run standard PyTorch or TensorFlow models on your hardware, and do you have a CUDA-equivalent SDK?
4. Compiler Performance: How efficiently does your graph compiler handle dynamic shapes and sparsity without manual optimization?
5. Diversity: Beyond LLMs, how does your architecture perform across diverse AI domains such as Computer Vision (CNNs/ViTs) and Vision-Language Models (VLMs)? Specifically, how does your hardware handle spatial data locality, 2D/3D convolutions, and the heterogeneous data processing required for multi-modal fusion?

Thanks,
-sergey

---

## Action Items Summary

### Immediate Actions (This Week)

#### 1. Prepare Slides for CEO Meeting
**Deadline:** Next Tuesday (CEO Mohammad meeting)  
**Format:** "A couple of slides" per person  
**Content:** Status and plans for 2026

**Your Deliverables:**
- [ ] Current status (what's complete)
- [ ] Plans for 2026 (what needs to be done)
- [ ] Timeline and milestones
- [ ] Dependencies and blockers

#### 2. Schedule 1:1 with VP Sergey
**Timeline:** This week or next Monday  
**Purpose:** Review and discuss your slides before CEO meeting

#### 3. Read CEO's Presentation
**Action:** Read carefully what Mohammad presented  
**Deliverable:** Prepare feedback on CEO's presentation

---

## Key Requirements Breakdown

### Timeline & Infrastructure

#### Tape-out Schedule
- **Original:** End of June 2026
- **Revised:** End of September 2026
- **Important:** "Not our fault" - external dependency
- **Impact:** Plans remain the same (no timeline relaxation)

#### FPGA Availability
- **Timeline:** Q1 2026 (local FPGA)
- **Cores:** 4 cores will be supported, maybe 6
- **Requirement:** SW stack must be ready to run on FPGA

#### Current Infrastructure
- **Available Now:** Synopsys cloud
- **Can Run:** SW stack on Synopsys cloud right now

#### Pre-Tape-out Requirements
"Must be ready to run our SW stack **before** tape-out on:"
- Simulators
- FPGA (local, Q1 2026)

---

### Required Demonstrations

#### 1. Performance Benchmarks (with PSim Team)
**Collaboration:** PSim team  
**Benchmarks:** AI benchmarks (to be defined together)  
**Key Benchmark:** MLPerf (customer requirement)  
**Purpose:** Potential customers want to see these results

**Action Items:**
- [ ] Define benchmark suite with PSim team
- [ ] Execute benchmarks
- [ ] Demonstrate performance results
- [ ] Compare with NVIDIA (customer question #1)

#### 2. AI SDK Demonstration
**Minimum Requirement:** Show how major AI frameworks use our hardware  
**Specific Examples:**
- PyTorch models
- TensorFlow models
- CUDA-like workloads

**Key Technology:** IREE compiler  
**Requirement:** Effectively demonstrate framework integration

**Action Items:**
- [ ] Demonstrate major AI framework integration
- [ ] Show CUDA-like workload support
- [ ] Use IREE compiler as enabler

#### 3. Robotics Pipeline
**Requirement:** Show at least main components  
**Purpose:** Demonstrate robotics use case

**Action Items:**
- [ ] Identify main robotics pipeline components
- [ ] Implement/demonstrate key components
- [ ] Prepare demo

#### 4. AI System Prototype
**Scope:** Multi-core & multi-node system  
**Example Technology:** Ray  
**Environment:** On simulators first

**Action Items:**
- [ ] Multi-core system prototype
- [ ] Multi-node system prototype
- [ ] Ray integration on simulators
- [ ] Demonstrate system scalability

---

## Customer Questions (Critical)

These questions are from **potential customers** and must be addressed.

### Question 1: Performance Benchmarks vs NVIDIA
**Question:** "Would you please share the performance benchmarks and comparisons between NVIDIA and your product?"

**What Customers Want:**
- Specific benchmark results
- Direct NVIDIA comparisons
- Performance metrics (throughput, latency, efficiency)

**Your Deliverables:**
- [ ] Benchmark suite execution
- [ ] NVIDIA comparison results
- [ ] Performance analysis report
- [ ] Cost/performance (TCO) analysis

**Dependencies:**
- PSim team (performance simulation)
- Hardware team (performance targets)

---

### Question 2: Official MLPerf Results
**Question:** "What are your official MLPerf results, and which specific models were used for your internal performance claims?"

**What Customers Want:**
- Official MLPerf submission results
- Specific model list used for claims
- Transparent methodology

**Your Deliverables:**
- [ ] MLPerf benchmark integration
- [ ] MLPerf results (official submission)
- [ ] Model list documentation
- [ ] Performance claims substantiation

**Dependencies:**
- PSim team (MLPerf infrastructure)
- Compiler team (IREE backend for models)

**Note:** This is a **credibility question** - customers want official, verifiable results

---

### Question 3: Software Ecosystem & CUDA SDK
**Question:** "How much code modification is required to run standard PyTorch or TensorFlow models on your hardware, and do you have a CUDA-equivalent SDK?"

**What Customers Want:**
- Migration effort estimate (minimal code changes)
- CUDA-equivalent SDK availability
- Developer experience quality

**Your Deliverables:**
- [ ] PyTorch integration (minimal code changes)
- [ ] TensorFlow integration (minimal code changes)
- [ ] CUDA-equivalent SDK documentation
- [ ] Migration guide (CUDA → RISC-V)
- [ ] Example code and tutorials

**Dependencies:**
- Compiler team (IREE backend, CUDA-equivalent SDK core)

**Key Message:** "Minimal to zero code modification" is the target answer

---

### Question 4: Compiler Performance (Dynamic Shapes & Sparsity)
**Question:** "How efficiently does your graph compiler handle dynamic shapes and sparsity without manual optimization?"

**What Customers Want:**
- Automatic optimization (no manual tuning)
- Dynamic shape handling
- Sparse tensor support

**Your Deliverables:**
- [ ] Dynamic shape support demonstration
- [ ] Sparse tensor support demonstration
- [ ] Performance comparison (automatic vs manual)
- [ ] Compiler optimization capabilities

**Dependencies:**
- Compiler team (IREE dynamic shape/sparsity support)

**Key Message:** "Automatic optimization" - customers don't want manual work

---

### Question 5: Diversity (Beyond LLMs)
**Question:** "Beyond LLMs, how does your architecture perform across diverse AI domains such as Computer Vision (CNNs/ViTs) and Vision-Language Models (VLMs)? Specifically, how does your hardware handle spatial data locality, 2D/3D convolutions, and the heterogeneous data processing required for multi-modal fusion?"

**What Customers Want:**
- CNN performance (ResNet, EfficientNet, etc.)
- Vision Transformer (ViT) performance
- Vision-Language Model (VLM) performance
- Spatial convolution optimization
- Multi-modal fusion capabilities

**Your Deliverables:**
- [ ] CNN benchmark suite (ResNet, EfficientNet, etc.)
- [ ] ViT benchmark suite
- [ ] VLM benchmark suite
- [ ] Spatial convolution performance analysis
- [ ] Multi-modal fusion performance analysis
- [ ] 2D/3D convolution optimization

**Dependencies:**
- Compiler team (convolution optimization)
- PSim team (performance validation)

**Key Message:** "Not just LLMs" - demonstrate broad AI capability

---

## Requirements Matrix: Your Scope

### ✅ Your Direct Ownership

| Requirement | Status | Timeline | Dependencies |
|------------|--------|----------|--------------|
| **FPGA Deployment Pipeline** | ⚠️ In Progress | Q1 2026 | Hardware (FPGA access) |
| **Multi-core SW Stack** | ⚠️ In Progress | Q1 2026 | Hardware (FPGA) |
| **Framework Integration (PyTorch/TensorFlow)** | ❌ Not Started | Q2-Q3 2026 | Compiler (IREE backend) |
| **MLPerf Integration** | ❌ Not Started | Q2-Q3 2026 | PSim, Compiler |
| **NVIDIA Comparisons** | ❌ Not Started | Q2-Q3 2026 | PSim |
| **Migration Guides** | ❌ Not Started | Q2-Q3 2026 | Compiler (SDK) |
| **Robotics Pipeline** | ❌ Not Started | Q2-Q3 2026 | - |
| **Ray Multi-node** | ⚠️ Partial | Q3 2026 | - |
| **Diversity Benchmarks (CNN/ViT/VLM)** | ❌ Not Started | Q2-Q3 2026 | PSim, Compiler |

### 🔴 Critical Dependencies (Compiler Team)

| Requirement | Blocker | Timeline | Impact |
|------------|---------|----------|--------|
| **Framework Integration** | IREE backend | Q2-Q3 2026 | Customer Q3 |
| **CUDA SDK** | SDK core implementation | Q2-Q3 2026 | Customer Q3 |
| **Dynamic Shapes/Sparsity** | IREE compiler features | Q2-Q3 2026 | Customer Q4 |
| **Convolution Optimization** | IREE convolution support | Q2-Q3 2026 | Customer Q5 |

### ⚠️ Collaboration Required (PSim Team)

| Requirement | Collaboration | Timeline | Purpose |
|------------|--------------|----------|---------|
| **MLPerf Benchmarks** | Infrastructure | Q2-Q3 2026 | Customer Q2 |
| **NVIDIA Comparisons** | Validation | Q2-Q3 2026 | Customer Q1 |
| **Diversity Benchmarks** | Performance | Q2-Q3 2026 | Customer Q5 |

---

## Timeline Alignment

### CEO's 4-Phase Plan vs VP's Requirements

| CEO Phase | VP Requirement | Status | Gap |
|-----------|---------------|--------|-----|
| **Phase 1: Validation (llama.cpp)** | Multi-core on FPGA | ✅ Partial | Need FPGA deployment |
| **Phase 2: Portability (Triton)** | CUDA-like workloads via IREE | ❌ Not Started | Compiler team blocker |
| **Phase 3: Scale (vLLM)** | Multi-node Ray system | ⚠️ Partial | Need disaggregation |
| **Phase 4: Economy (Ray Cluster)** | Performance benchmarks | ❌ Not Started | Need PSim collaboration |

### Critical Path to Tape-out (Sept 2026)

```
Q1 2026 (Jan-Mar) - Foundation
├── FPGA deployment pipeline
├── Multi-core support (4-6 cores)
└── Simulator integration

Q2 2026 (Apr-Jun) - Integration (CRITICAL)
├── Framework integration (blocked by Compiler)
├── MLPerf integration (collaboration with PSim)
├── Robotics pipeline components
└── CUDA SDK demonstration (blocked by Compiler)

Q3 2026 (Jul-Sep) - Validation (CRITICAL)
├── Performance benchmarks (all 5 customer questions)
├── MLPerf official results
├── NVIDIA comparisons
├── Diversity benchmarks
└── Tape-out readiness (Sept 30)
```

---

## Key Risks & Mitigation

### Risk 1: Timeline Compression
**Risk:** Tape-out moved from June → September, but plans remain same  
**Impact:** Less slack time for validation  
**Mitigation:** Prioritize critical path items, parallel execution

### Risk 2: Compiler Team Dependencies
**Risk:** 80% of your work blocked by Compiler Team (Q2-Q3)  
**Impact:** Customer questions #3, #4 cannot be answered without compiler  
**Mitigation:** 
- Establish weekly coordination with Compiler Team
- Define clear API/interface requirements early
- Create mock/stub interfaces for parallel development

### Risk 3: FPGA Availability (Q1)
**Risk:** FPGA needed in Q1, but "should have" (not confirmed)  
**Impact:** Multi-core validation delayed  
**Mitigation:**
- Confirm FPGA timeline with Hardware Team
- Use Synopsys cloud as fallback
- Prioritize simulator work

### Risk 4: Customer Questions Unanswered
**Risk:** Cannot answer 5 customer questions without deliverables  
**Impact:** Sales/business deals blocked  
**Mitigation:**
- Prioritize customer-facing deliverables (Q2-Q3)
- Establish clear ownership for each question
- Create interim status updates for customers

---

## Recommended Slide Content for CEO Meeting

### Slide 1: Status & Plans - Software Stack
**Current Status:**
- ✅ Phase 1 Complete: llama.cpp + frameworks (LangChain, LlamaIndex, Ray Serve)
- ⚠️ FPGA deployment pipeline (Q1 target)
- ⚠️ Multi-core support (Q1 target)

**2026 Plans:**
- Q1: FPGA readiness, multi-core, simulators
- Q2-Q3: Framework integration, MLPerf, robotics pipeline
- Q3: Performance validation, customer demos, tape-out readiness

### Slide 2: Dependencies & Critical Path
**Critical Dependencies:**
- 🔴 Compiler Team: IREE backend, Triton backend (Q2-Q3)
- ⚠️ Hardware Team: FPGA (Q1), ISA specs (ASAP)
- ⚠️ PSim Team: MLPerf infrastructure (Q2-Q3)

**Critical Path:**
- Hardware specs → Compiler Team → Software Stack → PSim → Customer Demos

### Slide 3: Customer Questions - Readiness
**Status by Question:**
1. Performance vs NVIDIA: ❌ Not ready (need PSim, Q2-Q3)
2. MLPerf results: ❌ Not ready (need PSim + Compiler, Q3)
3. PyTorch/TensorFlow + CUDA SDK: ❌ Not ready (need Compiler, Q2-Q3)
4. Dynamic shapes/sparsity: ❌ Not ready (need Compiler, Q2-Q3)
5. Diversity (CNN/ViT/VLM): ❌ Not ready (need Compiler + PSim, Q2-Q3)

**Timeline:** All questions answerable by Q3 2026 (before tape-out)

---

## Action Items for This Week

### High Priority (For 1:1 with VP)
- [ ] Prepare 2-3 slides as requested
- [ ] Read CEO's presentation and prepare feedback
- [ ] Review all 5 customer questions
- [ ] Identify gaps and dependencies
- [ ] Schedule 1:1 with VP Sergey (this week or Monday)

### Medium Priority (For CEO Meeting)
- [ ] Finalize slide deck
- [ ] Prepare talking points
- [ ] Coordinate with Compiler Team on timeline
- [ ] Confirm FPGA timeline with Hardware Team
- [ ] Establish PSim collaboration plan

### Long-term (Q1-Q3)
- [ ] Execute 2026 plan as outlined
- [ ] Weekly coordination with Compiler Team
- [ ] Bi-weekly updates to VP Sergey
- [ ] Track progress on customer questions

---

**Document Status:** VP Requirements Analysis  
**Last Updated:** January 2026  
**Next Steps:** Prepare slides for CEO meeting and 1:1 with VP Sergey
