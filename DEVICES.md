# Hardware Profile: HP ZBook Power 16 G11 Mobile Workstation

This document provides a comprehensive breakdown of the host machine running **JimRouteLLM**, including the CPU, NPU, discrete and integrated GPUs, memory architecture, and operating environment.

---

## 1. Workstation Overview

- **Device**: HP ZBook Power 16 inch G11 Mobile Workstation PC
- **Hostname**: `jac-hpz`
- **Form Factor**: High-Performance Mobile Workstation / AI Development PC
- **Primary Use**: Local LLM inference, NPU-accelerated classification, autonomous coding agent hosting, and local retrieval.

---

## 2. Processor & NPU Architecture

### AMD Ryzen 9 PRO 8945HS (HawkPoint)
- **Cores / Threads**: 8 Physical Cores / 16 Concurrent Threads (Zen 4 Architecture)
- **Base / Boost Frequency**: 4.0 GHz / up to 5.2 GHz Boost
- **L3 Cache**: 16 MB shared
- **Process Technology**: TSMC 4nm FinFET

### AMD Ryzen AI XDNA 1 NPU
- **Device Node**: `/dev/accel/accel0` (`RyzenAI-npu1`)
- **Driver / Subsystem**: Linux `amdxdna` driver (`/dev/accel` standard Linux accelerator subsystem)
- **Runtime Acceleration**: ONNX Runtime via Vitis AI / XRT (`pyxrt`) execution provider
- **Compute Capability**: 16 TOPS dedicated neural processing unit
- **Role in JimRouteLLM**: Evaluates prompt complexity using `ModernBERT-large` (395M parameters INT8) in under 40ms without utilizing CPU or GPU cycles.

---

## 3. Dual GPU Subsystems

The workstation operates an asymmetrical dual-GPU architecture, allowing task separation between local model hosting, visual rendering, and headless browser automation:

### Discrete GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
- **Architecture**: Ada Lovelace (AD107GLM)
- **VRAM**: 8 GB GDDR6 (8,188 MiB dedicated)
- **Driver Version**: NVIDIA Linux Driver `595.84`
- **CUDA Cores**: 3,072
- **Role in JimRouteLLM**: Hosts primary fast inference models (`google/gemma-4-12b-qat`, 7.15 GB footprint) for sub-second TTFT and continuous high-speed generation.

### Integrated GPU: AMD Radeon 780M
- **Architecture**: RDNA 3 (HawkPoint1 iGPU)
- **Compute Units**: 12 CUs (768 stream processors)
- **Memory**: Dynamically allocated unified DDR5 system memory
- **Role in JimRouteLLM**: Offloads display rendering, Wayland compositor tasks, and Playwright Headless Chrome automation without competing for discrete GPU VRAM.

---

## 4. Memory & Storage Architecture

- **System Memory (RAM)**: 64 GB DDR5 Dual-Channel (60 GiB addressable)
  - Provides adequate headroom to load both `meta/muse-glimmer` (18.16 GB) and `google/gemma-4-12b-qat` (7.15 GB) simultaneously into system memory and discrete VRAM without swapping.
- **Swap**: 512 MB emergency swap (paging strictly avoided to prevent DDR5 memory bandwidth degradation).
- **Storage**: High-speed PCIe Gen4 x4 NVMe SSD for fast weights loading and KV cache operations.

---

## 5. Operating System & Software Stack

- **Operating System**: Ubuntu 26.04.1 LTS (`resolute`) x86_64
- **Linux Kernel**: `7.0.0-31-generic` (Preempt Dynamic SMP)
- **Local Inference Engine**: LM Studio Local Server (`http://127.0.0.1:1234/v1`)
  - Configured with `1 parallel slot` per model to eliminate memory bus contention.
- **Python Runtime**: Python 3.12+ in dedicated virtual environment (`./venv`)
- **Web Retrieval Daemon**: SearXNG (`http://127.0.0.1:8888`) with Google & Google CSE engines
- **Browser Automation**: Host Google Chrome (`/usr/local/bin/google-chrome`) via `@playwright/mcp`
