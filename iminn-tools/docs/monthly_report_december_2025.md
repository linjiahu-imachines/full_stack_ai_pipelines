# Monthly Work Report - December 2025

## Summary
Here is a summary of my progress for December and the plan for the upcoming month.

---

## Completed & Highlights

### Onboarding & Environment Setup
- ✅ Completed full onboarding (Account setup, VPN, GitHub, remote machine, and JAR access) with Terry
- ✅ Conducted deep-dive syncs with Sergay, Imtiaz, and Sean to understand the iminn-tool repository
- ✅ Successfully utilized the repo to run the emulator and simulator with the extended IM instruction set on a single core CPU

### Single-Core Inference Validation
- ✅ Configured and executed Llama.cpp inference with a small model on a single-core environment (multi-threaded) using both the QEMU emulator and PSIM

### Multi-Core QEMU Configuration
- ✅ Successfully configured multi-core environments (single-socket and multi-socket) within QEMU
- ✅ Validated multi-threaded Llama.cpp model execution on this multi-core QEMU emulator

---

## In Progress

### Multi-Core PSIM Exploration
- 🔄 Currently working on running multi-threaded Llama.cpp models on multi-core PSIM
- **Status**: Pending readiness of the multi-core simulator features

---

## Plan for Next Month (January 2026)

### Primary Goals

1. **PSIM Validation**
   - Resume and finalize the execution of multi-threaded Llama.cpp on multi-core PSIM once the simulator updates are ready

2. **Framework Integration**
   - Begin integrating AI inference frameworks (specifically Ray Server) and Agent frameworks (langchain/graph) to run on the IM RISC-V architecture using the extended instruction set

3. **Software Stack Planning**
   - Define and plan the comprehensive software stack roadmap for the IM CPU architecture

---

*Report prepared by: Lin Hu*  
*Date: December 2025*
