# Customer Questions Deep Dive - Detailed Analysis (CORRECTED)
**Purpose:** In-depth analysis of 5 critical customer questions  
**Audience:** VP Sergey, CEO Mohammad  
**Your Role:** Director of Software Stack  
**NOTE:** This document contains NO fabricated data. All numbers marked as [TBD] or [CEO Target] are placeholders.

---

## ⚠️ Data Accuracy Notice

**What's REAL in this document:**
- CEO's strategic targets (from ceo_strategy_document.md)
- Customer questions (from VP's email)
- Methodologies and frameworks
- Your scope and dependencies

**What's NOT REAL (marked clearly):**
- Specific performance numbers → [TBD - Needs PSim measurement]
- MLPerf version numbers → [TBD - Check mlcommons.org]
- Benchmark results → [TBD - Needs execution]

---

## Overview: The 5 Customer Questions

These questions are from **real potential customers** (not hypothetical). Your answers directly impact sales.

| # | Question Category | Your Ownership | Critical Dependencies | Timeline |
|---|------------------|----------------|---------------------|----------|
| **Q1** | Performance vs NVIDIA | 🟡 Partial | PSim Team | Q2-Q3 |
| **Q2** | MLPerf Results | 🟡 Partial | Compiler + PSim | Q3 |
| **Q3** | PyTorch/TF + CUDA SDK | 🟢 Primary | Compiler Team | Q2-Q3 |
| **Q4** | Compiler Performance | 🔴 Support | Compiler Team (PRIMARY) | Q2-Q3 |
| **Q5** | Diversity (CNN/ViT/VLM) | 🟢 Primary | Compiler + PSim | Q2-Q3 |

**Legend:**
- 🟢 Primary: You own the answer and deliverables
- 🟡 Partial: You co-own with another team
- 🔴 Support: Another team owns, you provide support

---

## Question 1: Performance Benchmarks vs NVIDIA

### The Question (Exact Wording)
> "Would you please share the performance benchmarks and comparisons between NVIDIA and your product?"

### What They're Really Asking

**Surface Level:** Show me numbers  
**Deeper Level:** Prove you can compete with NVIDIA  
**Real Question:** "Why should I take the risk of switching from NVIDIA?"

**What They Want to See:**
1. **Specific Metrics:**
   - Throughput (tokens/second, inferences/second)
   - Latency (time-to-first-token, end-to-end latency)
   - Power efficiency (watts per inference)
   - Cost efficiency ($/inference, TCO over 3 years)

2. **Apples-to-Apples Comparison:**
   - Same model (e.g., Llama-3-70B)
   - Same batch size
   - Same precision (FP16 on NVIDIA vs MXFP4 on your chip)
   - Same workload characteristics

3. **TCO Analysis:**
   - Hardware cost
   - Power/cooling cost
   - Rack density (servers per rack)
   - 3-year total cost comparison

### Your Current State

**Status:** ❌ Not ready (no benchmarks executed yet)

**What You Have:**
- ✅ Software stack foundation (llama.cpp, frameworks)
- ✅ QEMU functional validation
- ❌ No performance data
- ❌ No NVIDIA baseline
- ❌ No comparison methodology

### What You Need to Deliver

#### Your Deliverables (Software Stack)

**1. Benchmark Suite Integration (Q2 2026)**
```
Task: Integrate industry-standard benchmarks
Models (Examples - TBD based on customer needs):
- Llama-3-8B, 70B (LLM)
- GPT models (if available)
- ResNet-50 (CNN)
- ViT-B/16 (Vision Transformer)

Workloads:
- Batch inference (throughput focus)
- Single request (latency focus)
- Mixed workload (realistic scenario)

Your Work:
- Model integration and quantization (MXFP4)
- Workload definition and scripting
- Input/output validation
- Results collection framework
```

**2. NVIDIA Baseline Collection (Collaboration with PSim)**
```
Task: Establish NVIDIA baseline for comparison
Hardware Targets:
- NVIDIA A100 (current generation)
- NVIDIA H100 (latest generation)
- [Specific models TBD - depends on fair comparison criteria]

Your Work:
- Define equivalent NVIDIA configuration
- Document methodology (transparent)
- Ensure apples-to-apples comparison
- Review PSim's baseline data
```

**3. Performance Results Analysis (Q3 2026)**
```
Task: Analyze and present results

Metrics to Calculate (values TBD):
- Throughput improvement: [X tokens/sec RISC-V vs Y tokens/sec NVIDIA]
- Latency comparison: [X ms vs Y ms]
- Power efficiency: [performance/watt comparison]
- TCO analysis: [3-year cost comparison]

Your Work:
- Results visualization (charts, graphs)
- Analysis report (what, why, how)
- Sales-ready materials (slide deck)
- Technical whitepaper (detailed methodology)
```

#### Critical Dependencies

**PSim Team (BLOCKER):**
```
What You Need:
1. Performance simulation infrastructure (Q2)
2. RISC-V chip simulation (cycle-accurate)
3. NVIDIA baseline simulation (or real hardware data)
4. Benchmark execution framework

Timeline: Must be ready by Q2 for Q3 results

Your Action:
- Schedule meeting with PSim team (this week)
- Define joint benchmark suite (within 2 weeks)
- Agree on methodology (within 2 weeks)
- Establish bi-weekly sync (ongoing)
```

### Your Answer Strategy

#### For VP/CEO Meeting (This Week)
```
"Question 1 asks for performance benchmarks vs NVIDIA.

CURRENT STATUS: Foundation complete, no benchmarks executed yet.

CEO'S TARGET (from strategy document):
- [CEO Target: 3-4x throughput via MXFP4]
- [CEO Target: 50% lower TCO vs NVIDIA nodes]

OUR PLAN:
- Q2: Benchmark integration (my work)
- Q2-Q3: Performance execution (PSim collaboration)
- Q3: Results ready for customers

CRITICAL DEPENDENCY: 
PSim infrastructure must be ready by Q2. I'm establishing 
coordination this week.

CONFIDENCE: Medium. 
- High confidence in my work (benchmark integration)
- Medium confidence in PSim timeline (need confirmation)

IMPORTANT: All performance numbers TBD - depend on PSim measurements.
We should NOT promise specific numbers until measured.

RISK: If PSim slips, we cannot answer this before tape-out."
```

#### For Customer (Q3 2026 - After Measurement)
```
Target Answer Template (Fill in after PSim measurements):

"Yes, we have comprehensive benchmarks comparing our RISC-V chip 
with NVIDIA A100/H100:

THROUGHPUT: [X]x higher tokens/second on Llama-3-70B 
(via MXFP4 compression) [TBD - PSim measurement]

LATENCY: [X]% difference in time-to-first-token [TBD - PSim measurement]

POWER: [X]x better performance-per-watt [TBD - PSim measurement]

TCO: [X]% lower 3-year total cost of ownership [TBD - calculation based on measured performance + power]

METHODOLOGY: We used industry-standard benchmarks with 
identical model sizes and batch configurations. Full methodology 
available in our technical whitepaper.

VALIDATION: Results validated on [FPGA/simulator] and 
cycle-accurate simulation."
```

### Action Items

**This Week:**
- [ ] Schedule meeting with PSim team lead
- [ ] Draft benchmark suite proposal (models, workloads)
- [ ] Review CEO's targets (understand what needs to be achieved)
- [ ] Prepare Q1 answer for VP meeting
- [ ] Clarify: Do NOT promise specific numbers yet

**Q1 2026 (Jan-Mar):**
- [ ] Finalize benchmark suite with PSim
- [ ] Begin model integration (on QEMU for functional validation)
- [ ] Validate functional correctness
- [ ] Set up results collection framework

**Q2 2026 (Apr-Jun):**
- [ ] Complete benchmark integration
- [ ] PSim executes benchmarks → GET REAL NUMBERS
- [ ] Collect NVIDIA baseline data
- [ ] Preliminary results analysis

**Q3 2026 (Jul-Sep):**
- [ ] Final results validation
- [ ] Complete TCO analysis (based on real measurements)
- [ ] Create sales materials (with real numbers)
- [ ] Prepare technical whitepaper
- [ ] Answer ready for customers WITH REAL DATA

---

## Question 2: Official MLPerf Results

### The Question (Exact Wording)
> "What are your official MLPerf results, and which specific models were used for your internal performance claims?"

### What They're Really Asking

**Surface Level:** Show me MLPerf scores  
**Deeper Level:** Are your claims legitimate and verifiable?  
**Real Question:** "Can I trust your performance claims, or is this marketing?"

**What They Want:**
1. **Official MLPerf Submission:**
   - Not internal benchmarks
   - Industry-recognized validation
   - Published on MLPerf.org
   - Auditable methodology

2. **Model Transparency:**
   - Specific models used for claims
   - Model sizes, architectures
   - Quantization formats
   - No cherry-picking

### Why This Question is Critical

**Trust Issue:**
- Many AI chip companies make inflated claims
- Customers have been burned before
- MLPerf is the industry standard for credibility
- "Official MLPerf" = legitimate, "internal benchmarks" = suspicious

### Your Current State

**Status:** ❌ Not ready (highest risk question)

**What You Have:**
- ❌ No MLPerf integration
- ❌ No official submission plan
- ❌ No model list documentation

**Why This is High Risk:**
- Dual dependencies (Compiler + PSim)
- MLPerf has strict rules and timelines
- First submission takes longer (learning curve)
- Results must be public (no hiding bad results)

### What You Need to Deliver

#### Step 1: Verify Current MLPerf Information (THIS WEEK)

**CRITICAL: Do NOT assume MLPerf benchmarks**
```
Action Required:
1. Visit https://mlcommons.org/benchmarks/inference/
2. Check CURRENT version number (v4.0? v4.1? v5.0?)
3. Check ACTUAL benchmark list for latest version
4. Check 2026 submission schedule
5. Download official benchmark code and rules

DO NOT PROCEED until you have verified:
- Current MLPerf version
- Official benchmark list
- Submission deadlines
- Requirements and rules
```

#### Step 2: Your Deliverables (After Verification)

**1. MLPerf Model Integration (Q2 2026)**
```
Task: Integrate MLPerf Inference benchmark suite

ACTUAL MLPerf Benchmarks (to be confirmed from mlcommons.org):
- [List TBD - check official site]
- Common examples: BERT, ResNet-50, but VERIFY

Your Work:
- Model quantization to MXFP4
- MLPerf LoadGen integration
- Accuracy validation (must meet MLPerf thresholds)
- Performance tuning
- Results submission preparation

IMPORTANT: Only plan for CONFIRMED benchmarks!
```

**2. Model Documentation (Q2 2026)**
```
Task: Document all models used for performance claims

For Each Model:
- Architecture (e.g., Llama-3-70B, ResNet-50, etc.)
- Size (parameters, weights)
- Source (HuggingFace, official repo, etc.)
- Quantization (MXFP4, precision level)
- Accuracy (vs FP32 baseline)
- Use case (why this model matters)

Your Work:
- Create model inventory spreadsheet
- Document quantization methodology
- Record baseline accuracy
- Link to internal performance claims
```

**3. MLPerf Submission (Q3 2026)**
```
Task: Submit official MLPerf results

MLPerf Submission Requirements (verify from official docs):
- Code submission (reproducible)
- Configuration files
- Measurement results
- Accuracy validation
- System description
- Compliance checker pass

Your Work:
- Coordinate with PSim for measurements
- Prepare submission package
- Run compliance checker
- Submit to MLPerf.org
- Handle any reviewer questions
```

#### Critical Dependencies

**Compiler Team (BLOCKER):**
```
What You Need:
1. IREE backend with MXFP4 support (for model compilation)
2. Model optimization (for competitive results)
3. Accuracy preservation (MXFP4 must meet MLPerf accuracy targets)

Without This: Cannot compile MLPerf models

Timeline: Q2 2026 (before your Q3 submission)

CRITICAL: MLPerf has strict accuracy requirements. MXFP4 
quantization must not degrade accuracy below thresholds.
```

**PSim Team (BLOCKER):**
```
What You Need:
1. MLPerf infrastructure setup
2. Benchmark execution environment
3. Performance measurement (cycle-accurate)
4. Results validation

Without This: Cannot generate official results

Timeline: Q2-Q3 2026
```

### Your Answer Strategy

#### For VP/CEO Meeting (This Week)
```
"Question 2 asks for official MLPerf results and model transparency.

CURRENT STATUS: Not started. This is our highest-risk question 
due to dual dependencies.

WHY THIS MATTERS:
MLPerf is the industry standard for credibility. Customers don't 
trust 'internal benchmarks' anymore. No MLPerf = red flag.

IMMEDIATE ACTION REQUIRED:
I need to verify current MLPerf information THIS WEEK:
- Current version number
- Official benchmark list
- 2026 submission schedule

I will NOT commit to specific benchmarks until verified.

OUR PLAN (after verification):
- Q2: MLPerf model integration (my work, blocked by Compiler)
- Q2-Q3: MLPerf execution (PSim infrastructure)
- Q3: Official submission to MLPerf.org

CRITICAL DEPENDENCIES:
1. Compiler Team: IREE backend for model compilation (Q2)
2. PSim Team: MLPerf infrastructure (Q2-Q3)

CONFIDENCE: Medium-Low
- Two critical dependencies
- First MLPerf submission (learning curve)
- Results will be public (no hiding)

RECOMMENDATION:
Make MLPerf a company-wide Q2-Q3 priority. This directly impacts 
sales credibility.

RISK: If either Compiler or PSim slips, we cannot submit MLPerf 
before tape-out, severely damaging credibility."
```

#### For Customer (Q3 2026 - After Official Results)
```
Target Answer Template (Fill in after MLPerf submission):

"Yes, we have official MLPerf Inference results submitted to MLPerf.org.

OFFICIAL RESULTS: [MLPerf.org link to our submission]

MODELS SUBMITTED: [List actual models submitted]

PERFORMANCE CLAIMS:
All our internal claims are based on these same models with 
identical configurations. Full model inventory and methodology 
available in our technical documentation.

TRANSPARENCY:
Our MLPerf submission includes full source code, configurations, 
and measurement methodology. Anyone can reproduce our results.

COMPETITIVE POSITION:
[Comparison to NVIDIA's MLPerf results for same benchmarks]"
```

### Action Items

**This Week (CRITICAL):**
- [ ] Visit https://mlcommons.org/benchmarks/inference/
- [ ] Verify current MLPerf version
- [ ] Download official benchmark list
- [ ] Check 2026 submission schedule
- [ ] Review submission requirements
- [ ] Update plan with REAL information

**Q1 2026 (Jan-Mar):**
- [ ] Study previous MLPerf submissions (NVIDIA, Google, etc.)
- [ ] Download MLPerf inference benchmark code
- [ ] Set up MLPerf development environment
- [ ] Test MLPerf LoadGen integration

**Q2 2026 (Apr-Jun):**
- [ ] Integrate MLPerf models (requires Compiler IREE backend)
- [ ] Quantize models to MXFP4
- [ ] Validate accuracy (meet MLPerf thresholds)
- [ ] Coordinate with PSim for execution
- [ ] Prepare submission package
- [ ] **Submit to MLPerf (check deadline)**

**Q3 2026 (Jul-Sep):**
- [ ] **MLPerf results published**
- [ ] Create customer-facing materials
- [ ] Update website with MLPerf badge (if good results)
- [ ] Prepare competitive analysis (vs NVIDIA MLPerf results)
- [ ] Answer ready for customers WITH OFFICIAL RESULTS

---

## Question 3: PyTorch/TensorFlow + CUDA SDK

### The Question (Exact Wording)
> "How much code modification is required to run standard PyTorch or TensorFlow models on your hardware, and do you have a CUDA-equivalent SDK?"

### What They're Really Asking

**Surface Level:** Is migration easy?  
**Deeper Level:** Will I need to hire specialized engineers?  
**Real Question:** "What's my risk and cost of adopting your platform?"

**What They Want:**
1. **Migration Effort:**
   - "Zero code changes" = ideal
   - "Change one line of config" = acceptable
   - "Rewrite model" = deal-breaker

2. **Developer Experience:**
   - Can my existing PyTorch engineers use your platform?
   - Or do I need RISC-V/hardware specialists?
   - What's the learning curve?

3. **CUDA Equivalence:**
   - Do you have kernel-level programmability?
   - Can I optimize custom operations?
   - What if I have proprietary CUDA kernels?

### Your Current State

**Status:** ⚠️ Partially ready (foundation exists)

**What You Have:**
- ✅ llama.cpp integration (proves RISC-V inference works)
- ✅ Framework wrappers (LangChain, LlamaIndex)
- ✅ Service layer (FastAPI, Ray Serve)
- ❌ No PyTorch `torch.compile` integration
- ❌ No TensorFlow integration
- ❌ No CUDA-equivalent SDK documentation

### What You Need to Deliver

#### Your Deliverables (Software Stack)

**1. PyTorch Integration Layer (Q2-Q3 2026)**
```
Task: Enable PyTorch models with minimal code changes

Target Integration Pattern:
# Customer's existing PyTorch code:
import torch
model = torch.load("llama3-70b.pt")
output = model(input)

# With RISC-V backend (minimal change):
import torch
torch.set_default_device("riscv")  # ← Target: ONE LINE CHANGE
model = torch.load("llama3-70b.pt")
output = model(input)

# Or using torch.compile:
model = torch.compile(model, backend="riscv")
output = model(input)

Your Deliverables:
- PyTorch backend registration
- torch.compile integration (requires Compiler Team IREE)
- Model loading and quantization (MXFP4)
- Inference execution
- Results validation
- Error handling

IMPORTANT: "One line change" is the TARGET, not guaranteed.
Actual code changes TBD based on Compiler Team's implementation.
```

**2. TensorFlow Integration Layer (Q2-Q3 2026)**
```
Task: Enable TensorFlow models with minimal code changes

Target Integration Pattern:
# Customer's existing TensorFlow code:
import tensorflow as tf
model = tf.keras.models.load_model("resnet50.h5")
output = model(input)

# With RISC-V backend (minimal change):
import tensorflow as tf
tf.config.set_visible_devices([...])  # ← Target: minimal change
model = tf.keras.models.load_model("resnet50.h5")
output = model(input)

Your Deliverables:
- TensorFlow device plugin
- Model loading and conversion
- Inference execution via IREE
- Results validation

IMPORTANT: Integration approach TBD - depends on Compiler Team.
```

**3. Migration Guide (Q2-Q3 2026)**
```
Task: Create comprehensive migration documentation

Migration Guide Contents:
1. Quick Start (10 minutes to first inference)
2. Model Conversion Guide (PyTorch → MXFP4)
3. API Reference (all functions)
4. Best Practices (optimization tips)
5. Troubleshooting (common issues)
6. FAQ (answers to typical questions)

Examples (with real models):
- Llama model migration
- ResNet migration
- Other customer-relevant models

Your Deliverables:
- Written documentation (markdown + website)
- Code examples (GitHub repo)
- Video tutorials (optional but valuable)
- Migration checklist

IMPORTANT: Cannot complete until Compiler Team delivers backends.
```

**4. CUDA-Equivalent SDK Documentation (Q2-Q3 2026)**
```
Task: Document the "CUDA-equivalent" developer experience

SDK Documentation Contents:
1. Architecture Overview (RISC-V vs CUDA comparison)
2. API Reference (function-by-function)
3. Triton Kernel Guide (custom operations via Triton)
4. Performance Optimization Guide
5. Migration from CUDA (concept mapping)

Your Deliverables:
- API documentation (coordinate with Compiler Team)
- Triton migration guide
- Example kernels (matrix multiply, etc.)
- Performance best practices

IMPORTANT: SDK core is Compiler Team's work.
You document and create examples.
```

#### Critical Dependencies

**Compiler Team (BLOCKER - CRITICAL):**
```
What You Need:
1. IREE backend (for PyTorch/TensorFlow compilation)
2. Triton backend (for custom kernel support)
3. CUDA-equivalent SDK core implementation

Without This: 
- Cannot demonstrate PyTorch integration
- Cannot show "minimal code changes"
- Cannot provide CUDA-equivalent capability

Timeline: Q2-Q3 2026

YOUR CRITICAL QUESTION FOR VP:
"When will Compiler Team deliver? This blocks customer Q3 answer."
```

### Your Answer Strategy

#### For VP/CEO Meeting (This Week)
```
"Question 3 asks about migration effort and CUDA-equivalent SDK.

CURRENT STATUS: Foundation exists (llama.cpp works), but 
PyTorch/TensorFlow integration not started.

TARGET ANSWER: 
'Minimal to zero code changes. Just change backend configuration.'

REALITY CHECK:
- "Minimal changes" is the GOAL, not guaranteed
- Actual migration effort TBD - depends on Compiler Team
- Cannot demonstrate until Compiler Team delivers

OUR APPROACH:
- Q2-Q3: PyTorch/TensorFlow integration layers (my work)
- Q2-Q3: Migration guides and documentation (my work)
- Q2-Q3: SDK documentation (my work)

CRITICAL DEPENDENCY:
Compiler Team must deliver:
1. IREE backend (for model compilation)
2. Triton backend (for custom kernels)
3. SDK core implementation

I can prepare integration layers and docs in parallel, but 
cannot complete or demonstrate until Compiler Team delivers.

CONFIDENCE: Medium
- High confidence in my work (integration, docs)
- Medium confidence in timeline (depends on Compiler Team)
- UNKNOWN: Actual code changes required (depends on implementation)

RECOMMENDATION:
Get Compiler Team timeline commitment THIS WEEK. Their delivery 
determines if we can answer this question before tape-out.

CUSTOMER IMPACT:
This is a deal-maker/deal-breaker question. Easy migration = win.
Difficult migration = lose.

DO NOT PROMISE "zero code changes" until we can demonstrate it."
```

#### For Customer (Q3 2026 - After Implementation)
```
Target Answer Template (Fill in after Compiler Team delivers):

"Migrating PyTorch or TensorFlow models to our platform requires 
[ACTUAL DESCRIPTION OF CHANGES - TBD].

PYTORCH MIGRATION:
[Real example based on actual implementation]

TENSORFLOW MIGRATION:
[Real example based on actual implementation]

CUDA-EQUIVALENT SDK:
[Description of actual SDK capabilities - TBD from Compiler Team]

CUSTOM KERNELS:
[Description of Triton support - TBD from Compiler Team]

MIGRATION SUPPORT:
- Comprehensive migration guide
- Example code for common models
- Technical support
- [Additional support based on what we can actually provide]

PROOF:
[Live demo with real model migration]"
```

### Action Items

**This Week:**
- [ ] Meet with Compiler Team lead
- [ ] Get IREE/Triton backend timeline
- [ ] Understand SDK core functionality
- [ ] Prepare Q3 answer for VP meeting
- [ ] Flag critical dependency
- [ ] DO NOT promise "zero code changes" yet

**Q1 2026 (Jan-Mar):**
- [ ] Study PyTorch backend API
- [ ] Prototype integration layer (mock implementation for planning)
- [ ] Draft migration guide structure
- [ ] Collect customer requirements

**Q2 2026 (Apr-Jun):**
- [ ] Implement PyTorch integration (when IREE backend ready)
- [ ] Implement TensorFlow integration
- [ ] Write migration guides (with real examples)
- [ ] Create code examples
- [ ] Test with real customer models

**Q3 2026 (Jul-Sep):**
- [ ] Complete SDK documentation
- [ ] Create demo videos (with real migrations)
- [ ] Customer pilot migrations (get feedback)
- [ ] Gather testimonials
- [ ] Answer ready with REAL PROOF

---

## Question 4: Compiler Performance (Dynamic Shapes/Sparsity)

### The Question (Exact Wording)
> "How efficiently does your graph compiler handle dynamic shapes and sparsity without manual optimization?"

### What They're Really Asking

**Surface Level:** Does your compiler work?  
**Deeper Level:** Will my engineers need to babysit the compiler?  
**Real Question:** "Can I trust your compiler to optimize automatically, or will I need compiler experts on staff?"

### Important Clarification

**THIS IS NOT YOUR QUESTION!**

**This question is primarily owned by: Compiler Team**

### Your Role (Support Only)

**What You Provide:**
- Test cases (realistic models with dynamic shapes/sparsity)
- Validation framework (measure correctness and performance)
- Customer examples (real-world use cases)

**What You DON'T Own:**
- Compiler implementation
- Optimization strategies
- Performance claims

### What You Need to Deliver (Supporting Role)

**1. Test Cases (Q2 2026)**
```
Task: Provide realistic test cases for compiler validation

Dynamic Shape Test Cases:
- Variable sequence lengths (NLP models)
- Variable batch sizes (serving scenarios)
- Variable image resolutions (vision models)
- Variable beam widths (generation)

Sparsity Test Cases:
- Sparse attention (long context models)
- Sparse activations (ReLU patterns)
- Sparse weights (pruned models)
- Mixed sparsity patterns

Your Work:
- Collect real customer models with these patterns
- Create test suite
- Document expected behavior
- Provide to Compiler Team for validation
```

**2. Validation Framework (Q2 2026)**
```
Task: Framework to validate compiler performance

Validation Metrics:
- Correctness: Does it produce correct answers?
- Performance: How fast vs manual optimization?
- Reliability: Does it handle edge cases?
- Consistency: Stable across different inputs?

Your Work:
- Automated test framework
- Performance comparison tools
- Results dashboard
- Report generation
```

### Your Answer Strategy

#### For VP/CEO Meeting (This Week)
```
"Question 4 asks about compiler dynamic shapes and sparsity.

IMPORTANT CLARIFICATION:
This is primarily a COMPILER TEAM question, not mine.

MY ROLE (Software Stack):
- Provide test cases (real customer models)
- Validate compiler performance
- Create examples demonstrating automatic optimization

COMPILER TEAM'S ROLE (PRIMARY):
- Implement dynamic shape support
- Implement sparsity optimization
- Ensure automatic optimization works

CURRENT STATUS: Not started (Compiler Team owns this)

RECOMMENDATION:
Compiler Team should present their plan for Q4 at this meeting.
They own this answer.

MY SUPPORT:
I'm ready to provide test cases and validation as soon as 
Compiler Team is ready to test.

CONFIDENCE: Low (not my primary scope)
This depends entirely on Compiler Team implementation."
```

#### For Customer (Q3 2026 - Compiler Team Delivers)
```
Target Answer (Compiler Team should present):

"Our compiler handles dynamic shapes and sparsity automatically.

[Compiler Team provides technical details and proof]

My validation: [Results from your test suite]"
```

### Action Items

**This Week:**
- [ ] Clarify with VP: This is Compiler Team's question
- [ ] Document your supporting role clearly
- [ ] Prepare test case collection plan

**Q1 2026 (Jan-Mar):**
- [ ] Collect customer models with dynamic shapes
- [ ] Collect sparse models
- [ ] Create test repository

**Q2 2026 (Apr-Jun):**
- [ ] Provide test cases to Compiler Team
- [ ] Validate compiler features (when ready)
- [ ] Document issues/limitations

**Q3 2026 (Jul-Sep):**
- [ ] Final validation with customer models
- [ ] Document test results
- [ ] Support Compiler Team's customer answer

---

## Question 5: Diversity (CNN/ViT/VLM Beyond LLMs)

### The Question (Exact Wording)
> "Beyond LLMs, how does your architecture perform across diverse AI domains such as Computer Vision (CNNs/ViTs) and Vision-Language Models (VLMs)? Specifically, how does your hardware handle spatial data locality, 2D/3D convolutions, and the heterogeneous data processing required for multi-modal fusion?"

### What They're Really Asking

**Surface Level:** Do you support vision models?  
**Deeper Level:** Are you LLM-only or truly general AI?  
**Real Question:** "Can I consolidate all my AI workloads on your platform, or do I need separate chips?"

### Why This Question is Critical

**Market Positioning:**
- LLM-only = niche player
- General AI = platform play
- Customers want to consolidate workloads

### Your Current State

**Status:** ❌ Not ready (LLM-focused so far)

**What You Have:**
- ✅ LLM support (llama.cpp, Llama models work)
- ❌ No CNN benchmarks
- ❌ No ViT benchmarks
- ❌ No VLM benchmarks
- ❌ No multi-modal fusion demos

### What You Need to Deliver

#### Your Deliverables (Software Stack)

**1. CNN Benchmark Suite (Q2 2026)**
```
Task: Integrate and benchmark CNN models

CNN Models (Examples - adjust based on customer needs):
- ResNet-50 (image classification)
- EfficientNet (modern CNN)
- YOLO (object detection)
- Others TBD based on customer requirements

Workloads:
- ImageNet classification
- COCO object detection (if relevant)
- Real-time inference (latency focus)
- Batch inference (throughput focus)

Your Work:
- Model integration and quantization (MXFP4)
- Dataset preparation
- Benchmark execution (with PSim)
- Results analysis and documentation

PERFORMANCE: TBD - depends on PSim measurements
```

**2. Vision Transformer Benchmark Suite (Q2 2026)**
```
Task: Integrate and benchmark ViT models

ViT Models (Examples):
- ViT-B/16 (base vision transformer)
- DeiT-Small (data-efficient ViT)
- Swin Transformer (hierarchical ViT)

Workloads:
- Image classification
- Variable image sizes (dynamic shapes)

Your Work:
- Model integration
- Performance measurement (with PSim)
- Analysis of attention patterns
- Optimization insights

PERFORMANCE: TBD - depends on PSim measurements
```

**3. Vision-Language Model Benchmark Suite (Q2 2026)**
```
Task: Integrate and benchmark VLM models

VLM Models (Examples):
- CLIP (image-text matching)
- LLaVA (visual instruction tuning)
- BLIP-2 (language-image pre-training)
- Others TBD based on availability and customer needs

Workloads:
- Zero-shot image classification
- Image-text retrieval
- Visual question answering (VQA)
- Multi-modal conversation

Your Work:
- Model integration (complex architectures)
- Multi-modal data handling
- Performance benchmarking (with PSim)
- Fusion analysis

PERFORMANCE: TBD - depends on PSim measurements
```

**4. Technical Deep-Dive Documentation (Q3 2026)**
```
Task: Answer technical sub-questions

Sub-Question 1: Spatial Data Locality
"How does your hardware handle spatial data locality?"

Your Answer: [TBD - coordinate with Hardware Team on architecture,
             Compiler Team on data layout optimizations]

Sub-Question 2: 2D/3D Convolution Optimization
"How do you optimize 2D/3D convolutions?"

Your Answer: [TBD - coordinate with Compiler Team on 
             convolution implementation strategy]

Sub-Question 3: Multi-Modal Fusion
"How do you handle heterogeneous data processing?"

Your Answer: [TBD - based on actual VLM implementation]

Your Deliverables:
- Technical whitepaper (after implementations complete)
- Architecture diagrams (coordinate with Hardware Team)
- Performance analysis (after PSim measurements)
- Code examples (real implementations)
```

#### Critical Dependencies

**Compiler Team (BLOCKER):**
```
What You Need:
1. Convolution optimization in IREE
2. Multi-modal model support
3. Variable image size handling

Timeline: Q2-Q3 2026
```

**PSim Team (BLOCKER):**
```
What You Need:
1. Benchmark infrastructure for vision models
2. Performance measurement (all model types)
3. Comparison baselines (NVIDIA on same models)

Timeline: Q2-Q3 2026
```

### Your Answer Strategy

#### For VP/CEO Meeting (This Week)
```
"Question 5 asks about AI diversity beyond LLMs.

CURRENT STATUS: LLM support proven (llama.cpp), but no 
CNN/ViT/VLM benchmarks yet.

WHY THIS MATTERS:
Customers want ONE chip for ALL AI workloads. If we're LLM-only, 
we're a niche player. If we're general AI, we're a platform.

OUR PLAN:
- Q2: CNN/ViT/VLM benchmark integration (my work)
- Q2-Q3: Convolution optimization (Compiler Team)
- Q2-Q3: Performance benchmarking (PSim)
- Q3: Technical documentation (my work)

CRITICAL DEPENDENCIES:
1. Compiler Team: Convolution optimization, multi-modal support
2. PSim Team: Benchmark infrastructure, performance measurement

TECHNICAL SUB-QUESTIONS:
Customer asks THREE specific technical questions:
1. Spatial data locality
2. 2D/3D convolution optimization
3. Multi-modal fusion

Each requires coordination with Compiler Team and Hardware Team.

PERFORMANCE CLAIMS:
DO NOT promise specific numbers (e.g., "2.8x faster on ResNet") 
until measured by PSim.

CONFIDENCE: Medium
- High confidence in model integration (my work)
- Medium confidence in Compiler optimization
- Medium confidence in PSim timeline

RECOMMENDATION:
Make 'diversity benchmarks' a Q2-Q3 priority. This positions us 
as a PLATFORM, not just an LLM chip."
```

#### For Customer (Q3 2026 - After Measurements)
```
Target Answer Template (Fill in after PSim measurements):

"Yes, our architecture performs across diverse AI domains.

LLMs (PROVEN):
[Performance data from actual measurements]

COMPUTER VISION (CNNs):
[Performance data from PSim measurements - TBD]

VISION TRANSFORMERS (ViTs):
[Performance data from PSim measurements - TBD]

VISION-LANGUAGE MODELS (VLMs):
[Performance data from PSim measurements - TBD]

TECHNICAL DETAILS:

1. Spatial Data Locality:
[Technical explanation based on actual architecture - TBD]

2. 2D/3D Convolution Optimization:
[Technical explanation based on Compiler implementation - TBD]

3. Multi-Modal Fusion:
[Technical explanation based on actual implementation - TBD]

PROOF:
[Benchmark results - real numbers after measurement]
[Architecture diagrams - real architecture]
[Demo: Running multiple model types on same chip]"
```

### Action Items

**This Week:**
- [ ] Research CNN/ViT/VLM requirements
- [ ] Identify most relevant models (customer-driven)
- [ ] Estimate integration effort
- [ ] Prepare Q5 answer for VP meeting
- [ ] Flag dependencies to Compiler/PSim teams
- [ ] DO NOT promise specific performance numbers

**Q1 2026 (Jan-Mar):**
- [ ] Download CNN models
- [ ] Download ViT models
- [ ] Download VLM models
- [ ] Set up validation environment

**Q2 2026 (Apr-Jun):**
- [ ] Integrate CNN models (requires Compiler support)
- [ ] Integrate ViT models
- [ ] Integrate VLM models
- [ ] Validate functional correctness
- [ ] Coordinate with PSim for benchmarking

**Q3 2026 (Jul-Sep):**
- [ ] Complete performance benchmarking (with PSim - GET REAL NUMBERS)
- [ ] Write technical documentation (based on real architecture)
- [ ] Answer spatial locality question (coordinate with Hardware/Compiler)
- [ ] Answer convolution question (coordinate with Compiler)
- [ ] Answer multi-modal question (based on implementation)
- [ ] Create demo materials
- [ ] Answer ready with REAL PROOF

---

## Summary: Your Action Plan for All 5 Questions

### This Week (For VP Meeting) - CRITICAL

**Key Message:**
```
"All 5 customer questions CAN BE answered by Q3 2026 IF:
1. Compiler Team delivers on Q2-Q3 timeline
2. PSim Team delivers performance infrastructure Q2-Q3
3. Cross-team coordination established THIS WEEK

However, I CANNOT promise specific numbers until measurements complete.

I'm ready to execute my work, but need cross-team coordination 
established THIS WEEK to ensure September tape-out readiness."
```

**CRITICAL: Data Accuracy**
```
VP, I want to be clear: I will NOT make up performance numbers.

All customer answers will be based on:
- Real PSim measurements (not speculation)
- Actual Compiler capabilities (not promises)
- Verified MLPerf results (not projections)

CEO's targets (3-4x throughput, 50% TCO) are GOALS, not guarantees.
We need to measure and validate before claiming to customers."
```

**Priority Order (By Risk):**
1. **🔴 Q2 (MLPerf):** Highest risk (dual dependency, credibility)
2. **🟡 Q3 (PyTorch/TF):** Deal-maker/deal-breaker
3. **🟡 Q5 (Diversity):** Market positioning
4. **🟢 Q1 (Performance):** Foundational
5. **⚪ Q4 (Compiler):** Not your question

**Immediate Actions:**
- [ ] Verify MLPerf information (mlcommons.org)
- [ ] Meet with Compiler Team lead (get realistic timeline)
- [ ] Meet with PSim Team lead (establish coordination)
- [ ] DO NOT promise specific numbers in CEO meeting
- [ ] Present realistic plan with clear dependencies

### Q1 2026 (Foundation)
- [ ] Set up all benchmark environments
- [ ] Create validation frameworks
- [ ] Establish team coordination (weekly)
- [ ] Begin parallel work (where possible)

### Q2-Q3 2026 (Execution & Measurement)
- [ ] Execute all integration work
- [ ] PSim measures real performance → GET REAL NUMBERS
- [ ] Compiler Team delivers backends
- [ ] Weekly checkpoints and risk reviews
- [ ] Prepare customer materials WITH REAL DATA

### By Sept 2026 (Tape-out)
- [ ] All 5 questions have REAL, MEASURED answers
- [ ] Customer demos with ACTUAL PROOF
- [ ] Sales materials with VERIFIED CLAIMS
- [ ] Technical documentation with REAL NUMBERS

---

**Document Status:** Corrected - All fabricated data removed  
**Next Steps:** Verify MLPerf info, establish cross-team coordination  
**Timeline:** Answers ready Q3 2026 with REAL MEASURED DATA
