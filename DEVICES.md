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
- **Vector Extensions**: AVX-512, AVX-512 VNNI (Vector Neural Network Instructions for INT8 dot products)
- **Base / Boost Frequency**: 4.0 GHz / up to 5.2 GHz Boost
- **L3 Cache**: 16 MB shared
- **Process Technology**: TSMC 4nm FinFET
- **Role in JimRouteLLM**: Executes the dynamically quantized INT8 ModernBERT classifier (`modernbert_large_int8.onnx`, 379.4 MB) via `CPUExecutionProvider` leveraging AVX-512 VNNI, achieving 18–35ms complexity scoring with 0.0ms heuristic fast-paths.

### AMD Ryzen AI XDNA 1 NPU
- **Device Node**: `/dev/accel/accel0` (`RyzenAI-npu1`, PCI ID `0x1502`)
- **Driver / Subsystem**: Linux `amdxdna` kernel driver (`/dev/accel` Linux accelerator subsystem)
- **Compute Capability**: 16 TOPS dedicated neural processing unit
- **Status in JimRouteLLM**: Probed and opened via `pyxrt.device(0)`. Under Linux, upstream ONNX Runtime packages lack drop-in `VitisAIExecutionProvider` support for dynamic transformer graphs (which require static AIE compilation and xclbin hardware overlays). Inference execution is therefore handled by the Zen 4 CPU's AVX-512 VNNI vector units, ensuring sub-50ms routing without requiring fixed-shape graph padding.

---

## 3. Dual GPU Subsystems

The workstation operates an asymmetrical dual-GPU architecture, allowing task separation between local model hosting, visual rendering, and headless browser automation:

### Discrete GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
- **Architecture**: Ada Lovelace (AD107GLM)
- **VRAM**: 8 GB GDDR6 (8,188 MiB dedicated)
- **Driver Version**: NVIDIA Linux Driver `595.84`
- **CUDA Cores**: 3,072
- **Role in JimRouteLLM**: Hosts the primary fast model tier (`google/gemma-4-e2b` Q6_K, ~2.5 GB footprint) running at 72 tokens/sec for sub-second TTFT and fast interactive turns.

### Integrated GPU: AMD Radeon 780M
- **Architecture**: RDNA 3 (HawkPoint1 iGPU)
- **Compute Units**: 12 CUs (768 stream processors)
- **Memory**: Dynamically allocated unified DDR5 system memory
- **Role in JimRouteLLM**: Offloads display rendering, Wayland compositor tasks, and Playwright Headless Chrome automation.

---

## 4. Memory & Storage Architecture

- **System Memory (RAM)**: 64 GB DDR5 Dual-Channel (60 GiB addressable)
  - Provides adequate headroom to host `google/gemma-4-26b-a4b-qat` (~16 GB footprint) and `google/gemma-4-e2b` simultaneously without paging or DDR5 bandwidth contention.
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
