# Monthly Work Report - January 2026

Hi Team,

Here is a summary of my progress for January and the plan for the upcoming month.

## Completed & Highlights

### GLM Model Integration & Validation
Successfully configured and executed GLM-4.6V-Flash (9.4B parameters) on both QEMU system mode and PSIM with RISC-V architecture.
Created comprehensive implementation guides for single-core and multi-core configurations with 9p filesystem sharing.

### Framework Integration Planning
Completed deep-dive analysis and implementation strategy for Ray Serve integration with RISC-V llama.cpp inference.
Evaluated Ray Serve LLM, Ray Data LLM, and agentic frameworks (LangGraph, AutoGen, CrewAI) for compatibility and ROI.
Created comprehensive guides for higher-level ML/LLM frameworks integration (Ray Serve, LangChain, LlamaIndex) covering architecture patterns, networking, and deployment strategies.

### Software Stack Strategic Planning
Defined comprehensive software stack roadmap aligned with CEO's "Switch-and-Scale" strategy and company-wide architecture.
Established clear scope boundaries: Framework Integration, Serving/Runtime, SDK, and Benchmarking layers.
Identified critical dependencies: 80% of deliverables depend on Compiler Team's IREE/Triton backend completion (Q2-Q3 2026).
Created complete technical architecture with layer-by-layer breakdown, ownership matrix, and validation paths (QEMU, PSIM, FPGA).
Established implementation roadmap from Q1 2026 through tape-out (Sept 2026).

## In Progress

### Multi-Core PSIM Validation
Currently working on multi-threaded Llama.cpp execution validation on multi-core PSIM.
Status: Documentation and workflows prepared, pending multi-core simulator feature readiness.

### Native RISC-V Binary Transition
Working to remove QEMU dependency and transition to native RISC-V execution on FPGA.
Status: Q1 2026 target as per roadmap.

## Plan for Next Month (February)

**Multi-Core PSIM Validation**: Complete multi-threaded Llama.cpp execution validation on multi-core PSIM and document performance characteristics.

**Native RISC-V Binary**: Remove QEMU dependency and validate direct RISC-V execution on FPGA for improved performance.

**Framework Integration - Phase 1**: Begin hands-on implementation of Ray Serve integration and agentic frameworks (LangGraph/AutoGen/CrewAI) on IM RISC-V architecture.

**MLPerf Preparation**: Start planning for MLPerf benchmark integration and multi-node Ray cluster architecture (Q2 target).
