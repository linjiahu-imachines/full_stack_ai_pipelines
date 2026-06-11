# IM Company Strategy Documents
**Purpose:** Strategic planning documents for VP 1:1 and company scope analysis  
**Date:** January 2026  
**Prepared for:** Director of Software Stack

---

## Document Overview

### 0. Executive Communications
- **`ceo_strategy_document.md`** - CEO Mohammad's complete strategy
  - "Switch-and-Scale" strategy
  - Three parallel execution paths
  - 4-phase migration roadmap (2026)
  - Detailed technical implementation
  - Business & technical ROI
  - 2026-2028 roadmap

- **`vp_sergey_requirements.md`** - VP Sergey's requirements email
  - Action items for CEO meeting (next Tuesday)
  - Tape-out schedule (Sept 2026)
  - Required demonstrations (benchmarks, SDK, robotics)
  - 5 critical customer questions
  - Requirements matrix and timeline
  - Dependencies and risks

- **`vp_email_action_plan.md`** - Detailed action plan for VP's requirements
  - What to prepare this week (slides, 1:1, feedback)
  - Recommended 3-slide structure for CEO meeting
  - What you CAN vs CANNOT deliver
  - Deep dive on each customer question
  - Talking points for 1:1 with VP Sergey
  - This week's action checklist

- **`customer_questions_deep_dive.md`** - In-depth analysis of 5 customer questions (CORRECTED)
  - ⚠️ **CORRECTED VERSION:** All fabricated data removed
  - Q1: Performance vs NVIDIA (benchmarks, TCO analysis) - Numbers [TBD]
  - Q2: MLPerf Results (official submission, credibility) - Verify mlcommons.org
  - Q3: PyTorch/TensorFlow + CUDA SDK (migration effort) - Implementation [TBD]
  - Q4: Compiler Performance (dynamic shapes, sparsity) - Compiler Team owns
  - Q5: Diversity - CNN/ViT/VLM (beyond LLMs) - Measurements [TBD]
  - Uses [TBD] and [CEO Target] markers for clarity
  - No speculative performance numbers
  - Answer strategies with realistic expectations

### 1. Executive Summary & Analysis
- **`vp_1on1_scope_analysis.md`** - Comprehensive scope analysis (645 lines)
  - Overall company scope vs. software stack scope
  - Gap analysis (what's done vs. what's needed)
  - Roadmap alignment with CEO's plan
  - Customer questions mapped to deliverables
  - Dependencies, risks, and action items

### 2. Presentation Materials
- **`vp_1on1_slide_outline.md`** - 9-slide presentation outline
  - Ready-to-use structure for VP Sergey 1:1
  - Executive summary, current state, scope, roadmap, dependencies
  - Key talking points and messages

- **`company_architecture_slide.md`** - Single-page architecture visual
  - One-page diagram for presentations
  - 7-layer stack from applications to hardware
  - Ownership matrix and critical dependencies

### 3. Technical Architecture
- **`company_overall_architecture.md`** - Complete technical architecture (666 lines)
  - Layer-by-layer breakdown (applications to hardware)
  - Component diagrams and data flows
  - Ownership matrix and timelines
  - Validation paths (QEMU, PSIM, FPGA)

### 4. Quick References
- **`scope_boundaries_quick_ref.md`** - Quick reference card
  - In-scope vs. out-of-scope (one-page)
  - Dependencies and critical success factors
  - What you own vs. what other teams own

- **`software_stack_scope_corrected.md`** - Corrected scope document
  - Your actual ownership (frameworks, serving, integration)
  - Compiler Team dependency (separate team)
  - Critical blockers and timeline

---

## Key Insights

### Your Scope (Software Stack Director)
**Layers You Own:**
- Framework Integration (LangChain, LlamaIndex, PyTorch, TensorFlow)
- Serving/Runtime (vLLM, llama.cpp, Ray Serve)
- SDK Integration (Documentation, migration guides)
- Benchmarking (MLPerf, NVIDIA comparisons)
- System Integration (Multi-core, multi-node, robotics)

### Critical Dependencies
1. **Compiler Team** (CRITICAL BLOCKER)
   - IREE backend, Triton backend
   - 80% of your work depends on this
   - Timeline: Q2-Q3 2026

2. **Hardware Team** (URGENT)
   - ISA specification, matrix units, FPGA
   - Timeline: Q1 2026

3. **PSim Team**
   - Performance infrastructure, MLPerf support
   - Timeline: Q2-Q3 2026

### Critical Path
```
Hardware Specs → Compiler Team → Software Stack → PSIM → Customer Demos
   (Q1)            (Q2-Q3)         (Q2-Q3)        (Q3)      (Q3-Q4)
```

---

## Document Usage

### For VP 1:1 Meeting
1. **Context:** Read `ceo_strategy_document.md` and `vp_sergey_requirements.md` first
2. **Primary:** Use `vp_1on1_slide_outline.md` for presentation structure
3. **Visual:** Use `company_architecture_slide.md` for architecture diagram
4. **Reference:** Use `scope_boundaries_quick_ref.md` for quick lookups
5. **Deep Dive:** Reference `vp_1on1_scope_analysis.md` for detailed questions

### For Daily Work
1. **Scope Clarity:** Reference `software_stack_scope_corrected.md`
2. **Architecture:** Reference `company_overall_architecture.md`
3. **Quick Checks:** Reference `scope_boundaries_quick_ref.md`

### For Team Discussions
1. **Compiler Team:** Use scope documents to clarify dependencies
2. **Hardware Team:** Use architecture documents to discuss requirements
3. **PSim Team:** Use benchmarking sections for collaboration

---

## Key Messages

### For VP Sergey
- "I own framework and serving layers, working closely with Compiler Team"
- "80% of my deliverables depend on Compiler Team's IREE/Triton backends (Q2-Q3)"
- "Critical path: Hardware specs → Compiler Team → My work → Customer demos"

### For CEO Mohammad
- "Software Stack is the critical integrator between hardware and customers"
- "Three parallel paths all depend on software stack deliverables"
- "Timeline risk: Any slip in compiler work cascades to customer demos"

---

## Timeline Summary

### Q1 2026 (Jan-Mar) - Foundation
- Multi-core support for FPGA
- Native RISC-V binary (remove QEMU)
- FPGA deployment pipeline

### Q2 2026 (Apr-Jun) - Integration (CRITICAL)
- PyTorch/TensorFlow integration (blocked by Compiler Team)
- vLLM platform plugin (blocked by Compiler Team)
- MLPerf benchmarks start
- Robotics pipeline

### Q3 2026 (Jul-Sep) - Production (CRITICAL)
- vLLM full integration
- Ray multi-node cluster
- MLPerf official results
- Diversity benchmarks
- Tape-out readiness (Sept 30)

### Q4 2026 (Oct-Dec) - Customer Demos
- Enterprise demos
- Customer-ready documentation
- Migration guides

---

## Status Summary

### ✅ Complete
- llama.cpp integration (Phase 1-5)
- LangChain integration (Phase 3)
- LlamaIndex integration (Phase 5)
- Ray Serve basic integration (Phase 4)
- FastAPI server (Phase 2)
- QEMU functional validation

### ⚠️ In Progress
- Multi-core support
- Ray Serve enhancements
- FPGA deployment

### ❌ Not Started (Critical)
- IREE backend (Compiler Team) - BLOCKER
- Triton backend (Compiler Team) - BLOCKER
- vLLM platform plugin
- PyTorch/TensorFlow integration
- MLPerf benchmarks
- Diversity benchmarks
- Robotics pipeline

---

**Last Updated:** January 2026  
**Prepared by:** AI Assistant for Director of Software Stack  
**Next Review:** After VP 1:1 meeting
