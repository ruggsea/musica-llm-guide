# Models to Benchmark -- Social Media Simulation

> **Cluster**: VSC MUSICA -- 4x NVIDIA H100 **94GB** per node (376GB total VRAM)
> **Partition**: `zen4_0768_h100x4` (112 nodes, 192 cores, 768GB RAM, 4x H100 94GB)
> **Date**: February 2026
> **Serving frameworks**: vLLM, SGLang, TGI (via SLURM jobs)

---

## Table of Contents

1. [Closed/Proprietary Models](#1-closedproprietary-models)
2. [Open-Source: Small Models (1-4B)](#2-small-models-1-4b)
3. [Open-Source: Medium Models (7-14B)](#3-medium-models-7-14b)
4. [Open-Source: Large Models (27-40B)](#4-large-models-27-40b)
5. [Open-Source: XL Models (70B class)](#5-xl-models-70b-class)
6. [Open-Source: Frontier MoE Models (100B+)](#6-frontier-moe-models-100b)
7. [Ablation Models](#7-ablation-models)
8. [GPU Memory Calculations & Precision Guide](#8-gpu-memory-calculations--precision-guide)
9. [LiteLLM Integration Reference](#9-litellm-integration-reference)
10. [Benchmark Scores Reference](#10-benchmark-scores-reference)
11. [Multi-Node Deployment (2+ Nodes)](#multi-node-deployment-2-nodes)

---

## Hardware Reference

```
GPU:    NVIDIA H100 SXM5, 94GB HBM3 each
Count:  4 GPUs per node (NVLink 4.0 interconnect)
Total:  376 GB VRAM per node
CPU:    192 cores (AMD EPYC Zen4)
RAM:    768 GB DDR5
```

### Memory Formula

```
Model Memory = Parameters (B) x Bytes_per_param x Overhead_factor(1.15)

Bytes per param:
  FP32  = 4.0 bytes    BF16/FP16 = 2.0 bytes
  FP8   = 1.0 byte     INT8      = 1.0 byte
  INT4  = 0.5 bytes

Overhead (15%) accounts for: KV cache, activations, CUDA context, framework buffers
```

---

## 1. Closed/Proprietary Models

These are API-only models accessed via LiteLLM.

| Vendor | Model | LiteLLM String | Params | Context | Input $/1M | Output $/1M | Notes |
|--------|-------|----------------|--------|---------|------------|-------------|-------|
| **OpenAI** | GPT-5 | `openai/gpt-5` | Undisclosed | ? | TBD | TBD | **VERIFY** -- expected early 2026 |
| OpenAI | GPT-5-mini | `openai/gpt-5-mini` | Undisclosed | ? | TBD | TBD | **VERIFY** |
| OpenAI | o3-mini | `openai/o3-mini` | Undisclosed | 200K | ~$1.10 | ~$4.40 | Reasoning model |
| OpenAI | GPT-4o | `openai/gpt-4o` | Undisclosed | 128K | $2.50 | $10.00 | Baseline reference |
| OpenAI | GPT-4o-mini | `openai/gpt-4o-mini` | Undisclosed | 128K | $0.15 | $0.60 | Cost-effective |
| OpenAI | GPT-4.1 | `openai/gpt-4.1` | Undisclosed | 1M | $2.00 | $8.00 | Strong instruction following |
| OpenAI | GPT-4.1-mini | `openai/gpt-4.1-mini` | Undisclosed | 1M | $0.40 | $1.60 | |
| OpenAI | GPT-4.1-nano | `openai/gpt-4.1-nano` | Undisclosed | 1M | $0.10 | $0.40 | Cheapest OpenAI |
| OpenAI | o1 | `openai/o1` | Undisclosed | 200K | $15.00 | $60.00 | Reasoning |
| OpenAI | o3 | `openai/o3` | Undisclosed | 200K | $10.00 | $40.00 | Reasoning |
| **Anthropic** | Claude Opus 4.5 | `anthropic/claude-opus-4-5-20250929` | Undisclosed | 200K | ~$15.00 | ~$75.00 | **VERIFY string** |
| Anthropic | Claude Sonnet 4.5 | `anthropic/claude-sonnet-4-5-20250929` | Undisclosed | 200K | $3.00 | $15.00 | |
| Anthropic | Claude Haiku 4.5 | `anthropic/claude-haiku-4-5-20251001` | Undisclosed | 200K | $0.80 | $4.00 | Fast, cheap |
| Anthropic | Claude Opus 4 | `anthropic/claude-opus-4-20250514` | Undisclosed | 200K | $15.00 | $75.00 | |
| Anthropic | Claude Sonnet 4 | `anthropic/claude-sonnet-4-20250514` | Undisclosed | 200K | $3.00 | $15.00 | |
| **Google** | Gemini 3 Pro | `gemini/gemini-3-pro` | Undisclosed | ? | TBD | TBD | **VERIFY** -- expected 2026 |
| Google | Gemini 2.5 Flash | `gemini/gemini-2.5-flash-preview-04-17` | Undisclosed | 1M | $0.15 | $0.60/$3.50 | Thinking mode |
| Google | Gemini 2.5 Pro | `gemini/gemini-2.5-pro-preview-03-25` | Undisclosed | 1M | $1.25 | $10.00 | Reasoning |
| Google | Gemini 2.0 Flash | `gemini/gemini-2.0-flash` | Undisclosed | 1M | $0.10 | $0.40 | Very fast |
| **xAI** | Grok 4 | `xai/grok-4` | Undisclosed | ? | TBD | TBD | **VERIFY** |
| xAI | Grok 4.1 | `xai/grok-4.1` | Undisclosed | ? | TBD | TBD | **VERIFY** |
| xAI | Grok 3 | `xai/grok-3` | Undisclosed | 128K | ~$3.00 | ~$15.00 | |
| **Mistral** | Mistral Large 3 | `mistral/mistral-large-latest` | 675B MoE (41B active) | 256K | $2.00 | $6.00 | Open weights: Apache 2.0 |
| **Cohere** | Command R+ | `cohere/command-r-plus` | 104B | 128K | $2.50 | $10.00 | RAG-optimized |
| **DeepSeek** | DeepSeek-V3.2 | `deepseek/deepseek-chat` | 685B MoE | 128K | $0.27 | $1.10 | Very cheap API |
| DeepSeek | DeepSeek-R1 | `deepseek/deepseek-reasoner` | 671B MoE | 128K | $0.55 | $2.19 | Reasoning |

---

## 2. Small Models (1-4B)

All fit on a **single GPU at FP32**. Can run **4 instances in parallel** (one per GPU) for maximum throughput.

| Model | Size | HuggingFace ID | License | Context | Training Precision | Max Precision on 1x H100 | Memory @ BF16 | Notes |
|-------|------|----------------|---------|---------|-------------------|--------------------------|---------------|-------|
| **Qwen3-0.6B** | 0.6B | `Qwen/Qwen3-0.6B` | Apache 2.0 | 32K | BF16 | **FP32** (2.8 GB) | 1.4 GB | Smallest Qwen3 |
| **Qwen3-1.7B** | 1.7B | `Qwen/Qwen3-1.7B` | Apache 2.0 | 32K | BF16 | **FP32** (7.8 GB) | 3.9 GB | |
| **Qwen3-4B** | 4B | `Qwen/Qwen3-4B` | Apache 2.0 | 32K | BF16 | **FP32** (18.4 GB) | 9.2 GB | |
| **SmolLM3-3B** | 3B | `HuggingFaceTB/SmolLM3-3B` | Apache 2.0 | 8K+ | BF16 | **FP32** (13.8 GB) | 6.9 GB | HuggingFace, fully open |
| **Phi-4-mini** | 3.8B | `microsoft/Phi-4-mini-instruct` | MIT | 128K | BF16 | **FP32** (17.5 GB) | 8.7 GB | Strong for size |
| **Gemma-3n-E2B** | 5.44B (effective 2B) | `google/gemma-3n-E2B-it` | Gemma License | 32K | BF16 | **FP32** (25.0 GB) | 12.5 GB | Multimodal, "E2B" = effective 2B active |
| **Ministral 3B** | 3.85B | `mistralai/Ministral-3-3B-Instruct-2512` | Apache 2.0 | 32K | FP8 (ships FP8) | **FP32** (17.7 GB) | 8.9 GB | Default weights are FP8 |
| **Llama 3.2 1B** | 1.24B | `meta-llama/Llama-3.2-1B-Instruct` | Llama 3.2 CL | 128K | BF16 | **FP32** (5.7 GB) | 2.9 GB | Distilled from 8B/70B |
| **Llama 3.2 3B** | 3.21B | `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 CL | 128K | BF16 | **FP32** (14.8 GB) | 7.4 GB | Distilled from 8B/70B |

### Deployment: Small Models

```bash
# Run 4 instances in parallel (1 per GPU), all at FP32 or BF16
# Example with vLLM:
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --dtype float32 \
  --tensor-parallel-size 1 \
  --port 8000 \
  --gpu-memory-utilization 0.90
```

---

## 3. Medium Models (7-14B)

All fit on a **single GPU at BF16** easily. Most fit at FP32 on 94GB H100.

| Model | Size | HuggingFace ID | License | Context | Training Precision | Max Precision on 1x H100 94GB | Memory @ BF16 | Notes |
|-------|------|----------------|---------|---------|-------------------|-------------------------------|---------------|-------|
| **Llama 3.1 8B** | 8B | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 CL | 128K | BF16 | **FP32** (36.8 GB) | 18.4 GB | Popular baseline |
| **Llama 4 Scout 8B** | 8B active | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Llama 4 CL | 10M | BF16 | See Frontier MoE (109B total) | -- | **MoE: 109B total** |
| **Qwen3-8B** | 8B | `Qwen/Qwen3-8B` | Apache 2.0 | 128K | BF16 | **FP32** (36.8 GB) | 18.4 GB | |
| **Gemma 3 9B** | 9B | `google/gemma-3-12b-it` | Gemma License | 128K | BF16 | **FP32** (41.4 GB) | 20.7 GB | Actually 12B; vision-language |
| **Mistral 7B v0.3** | 7.2B | `mistralai/Mistral-7B-Instruct-v0.3` | Apache 2.0 | 32K | BF16 | **FP32** (33.1 GB) | 16.6 GB | Classic baseline |
| **Ministral 8B** | 9B (8.4B LM + 0.4B vision) | `mistralai/Ministral-3-8B-Instruct-2512` | Apache 2.0 | 256K | FP8 (ships FP8) | **FP32** (41.4 GB) | 20.7 GB | Multimodal, ships FP8 |
| **Ministral 14B** | 14B (13.5B LM + 0.4B vision) | `mistralai/Ministral-3-14B-Instruct-2512` | Apache 2.0 | 256K | FP8 (ships FP8) | **FP32** (64.4 GB) | 32.2 GB | Multimodal, ships FP8 |
| **DeepSeek-R1-Distill-Llama-8B** | 8B | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | MIT | 128K | BF16 | **FP32** (36.8 GB) | 18.4 GB | R1 reasoning distilled |
| **Qwen-7B** | 7.6B | `Qwen/Qwen2.5-7B-Instruct` | Apache 2.0 | 128K | BF16 | **FP32** (35.0 GB) | 17.5 GB | Qwen 2.5 series |
| **Phi-4 14B** | 14B | `microsoft/phi-4` | MIT | 16K | BF16 | **FP32** (64.4 GB) | 32.2 GB | 9.8T training tokens |
| **OLMo 3 7B** | 7B | `allenai/Olmo-3-7B-Instruct` | Apache 2.0 | 4K+ | BF16 | **FP32** (32.2 GB) | 16.1 GB | Ai2, fully open |
| **Apertus 8B** | 8.05B | `swiss-ai/Apertus-8B-Instruct-2509` | Apache 2.0 | 8K+ | BF16 | **FP32** (37.0 GB) | 18.5 GB | Swiss National AI Institute |

### Note on Llama 4 Scout

Llama 4 Scout appears in the medium list by active params (8B/17B active) but has **109B total parameters** (16 MoE experts). It belongs in the **Frontier MoE** section for memory calculations. See Section 6.

### Deployment: Medium Models

```bash
# BF16 on single GPU (recommended -- plenty of headroom)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# Can run 2-4 instances for smaller models (7-8B at BF16 ≈ 18GB, fits 4x on 4 GPUs)
```

---

## 4. Large Models (27-40B)

Fit on 1-2 GPUs at BF16. TP=2 recommended for models >27B for comfortable headroom.

| Model | Size | HuggingFace ID | License | Context | Max Precision (1x 94GB GPU) | Memory @ BF16 | TP Needed | Notes |
|-------|------|----------------|---------|---------|------------------------------|---------------|-----------|-------|
| **Qwen3-32B** | 32B | `Qwen/Qwen3-32B` | Apache 2.0 | 128K | **BF16** (73.6 GB, tight) | 73.6 GB | TP=1 possible, **TP=2 recommended** | Only 20 GB headroom on 1 GPU |
| **Qwen3-30B-A3B MoE** | 30B total / 3B active | `Qwen/Qwen3-30B-A3B` | Apache 2.0 | 128K | **BF16** (69.0 GB) | 69.0 GB | TP=1 possible | MoE: 30B total must be loaded |
| **DeepSeek-R1-Distill-Qwen-32B** | 32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | MIT | 128K | **BF16** (73.6 GB, tight) | 73.6 GB | TP=1 possible, **TP=2 recommended** | Strong reasoning |
| **Gemma 2 27B** | 27B | `google/gemma-2-27b-it` | Gemma License | 8K | **BF16** (62.1 GB) | 62.1 GB | TP=1 | Comfortable on 1 GPU |
| **OLMo 3.1 32B** | 32B | `allenai/Olmo-3.1-32B-Instruct` | Apache 2.0 | 32K+ | **BF16** (73.6 GB, tight) | 73.6 GB | **TP=2 recommended** | Ai2, fully open |
| **Magistral Small** | 24B | `mistralai/Magistral-Small-2509` | Apache 2.0 | 128K | **BF16** (55.2 GB) | 55.2 GB | TP=1 | Mistral reasoning model |

### Deployment: Large Models

```bash
# 32B models: TP=2 for comfortable headroom (36.8 GB/GPU)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# 27B models: TP=1 works fine (62.1 GB on 94GB GPU)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90
```

---

## 5. XL Models (70B class)

Require **TP=4 for BF16**. All fit comfortably at BF16 across 4x H100 94GB.

| Model | Size | HuggingFace ID | License | Context | Max Precision (4x 94GB) | Memory @ BF16 | Per-GPU @ TP=4 | Notes |
|-------|------|----------------|---------|---------|--------------------------|---------------|----------------|-------|
| **Llama 3.3 70B** | 70.6B | `meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 CL | 128K | **BF16** | 162.4 GB | 40.6 GB/GPU | Latest Llama 3.x 70B |
| **Llama 4 Maverick** | 400B total / 17B active | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Llama 4 CL | 1M | See Frontier MoE | -- | -- | **MoE 128 experts, see Section 6** |
| **Qwen2.5-72B** | 72.7B | `Qwen/Qwen2.5-72B-Instruct` | Qwen License | 128K | **BF16** | 167.2 GB | 41.8 GB/GPU | Strong all-around |
| **DeepSeek-R1-Distill-Llama-70B** | 70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | MIT | 128K | **BF16** | 161.0 GB | 40.3 GB/GPU | R1 reasoning in 70B |
| **K2-V2 70B** | 70B | `LLM360/K2-V2-Instruct` | Apache 2.0 | 128K | **BF16** | 161.0 GB | 40.3 GB/GPU | LLM360/MBZUAI, fully open |
| **Apertus 70B** | 70B | `swiss-ai/Apertus-70B-Instruct-2509` | Apache 2.0 | 128K | **BF16** | 161.0 GB | 40.3 GB/GPU | Swiss AI, fully open |

### Deployment: XL Models

```bash
# All 70B models: TP=4 at BF16 -- plenty of headroom (~40 GB/GPU, ~54 GB free per GPU)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# For BF16 70B @ TP=4: ~40 GB/GPU model + ~44 GB/GPU free for KV cache
# Supports ~32K context easily, ~65K+ with careful tuning
```

---

## 6. Frontier MoE Models (100B+)

**Critical**: MoE models require ALL parameters in VRAM, not just active ones.

| Model | Total Size | Active Size | HuggingFace ID | License | Context | Max Precision (4x 94GB) | Total Mem @ Max | Per-GPU @ TP=4 | Fits? |
|-------|-----------|-------------|----------------|---------|---------|--------------------------|-----------------|----------------|-------|
| **Llama 4 Scout** | 109B | 17B | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Llama 4 CL | 10M | **BF16** | 250.7 GB | 62.7 GB/GPU | YES |
| **Llama 4 Maverick** | 400B | 17B | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Llama 4 CL | 1M | **INT4** | 230.0 GB | 57.5 GB/GPU | YES (INT4 only) |
| | | | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | | | **FP8** | 460.0 GB | -- | NO (460 > 376) |
| **DeepSeek-V3.2** | 685B | 37B | `deepseek-ai/DeepSeek-V3.2` | MIT | 128K | **DOES NOT FIT** | 394.7 GB (INT4) | -- | NO (ships FP8 natively) |
| **DeepSeek-R1** | 671B | 37B | `deepseek-ai/DeepSeek-R1` | MIT | 128K | **DOES NOT FIT** | 385.8 GB (INT4) | -- | NO (ships FP8 natively) |
| **Qwen3-235B-A22B** | 235B | 22B | `Qwen/Qwen3-235B-A22B` | Apache 2.0 | 128K | **FP8/INT8** | 270.3 GB | 67.6 GB/GPU | YES |
| **Kimi K2** | 1,032B | 32B | `moonshotai/Kimi-K2-Instruct` | Modified MIT | 128K | **DOES NOT FIT** | 593.4 GB (INT4) | -- | NO (2+ nodes) |
| **Mistral Large 3** | 675B | 41B | `mistralai/Mistral-Large-3-675B-Instruct-2512` | Apache 2.0 | 256K | **DOES NOT FIT** | 388.1 GB (INT4) | -- | NO (2+ nodes) |
| | | | `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` | | | NVFP4 variant | ~340 GB? | ~85 GB/GPU | **MAYBE** w/ NVFP4 |
| **GLM-4.5** | 355B | 32B | `zai-org/GLM-4.5` | MIT | 128K | **FP8** | 408.3 GB | -- | NO (just over) |
| | | | | | | **INT4** | 204.1 GB | 51.0 GB/GPU | YES |
| **GLM-4.5-Air** | 106B | 12B | `zai-org/GLM-4.5-Air` | MIT | 128K | **BF16** | 243.8 GB | 60.9 GB/GPU | YES |
| **MiMo-V2-Flash** | 309B | 15B | `XiaomiMiMo/MiMo-V2-Flash` | MIT | 128K | **FP8** (ships FP8) | 355.4 GB | 88.8 GB/GPU | YES (tight) |
| **MiniMax-M2.5** | 229B | ~18B (8/256 experts) | `MiniMaxAI/MiniMax-M2.5` | Modified-MIT | 196K | **FP8** (ships FP8) | 263.0 GB | 65.8 GB/GPU | YES |
| **MiniMax-M2.1** | 229B | ~18B (8/256 experts) | `MiniMaxAI/MiniMax-M2.1` | Modified-MIT | 196K | **FP8** (ships FP8) | 263.0 GB | 65.8 GB/GPU | YES |
| **MiniMax-M2** | 229B | ~18B (8/256 experts) | `MiniMaxAI/MiniMax-M2` | Modified-MIT | 196K | **FP8** (ships FP8) | 263.0 GB | 65.8 GB/GPU | YES |
| **MiniMax-M1-80k** | 456B | ~46B | `MiniMaxAI/MiniMax-M1-80k` | Modified-MIT | 80K | **BF16** | 1049.0 GB | -- | NO (PP unsupported) |
| **GLM-4.6** | 357B | 32B | `zai-org/GLM-4.6-FP8` | MIT | 128K | **FP8** | 410.6 GB | -- | NO (2 nodes FP8) |
| **LongCat-Flash-Chat** | 562B | 27B | `meituan-longcat/LongCat-Flash-Chat-FP8` | MIT | 256K | **FP8** | 646.3 GB | -- | NO (2 nodes FP8) |
| **INTELLECT-3** | 106B | 12B | `PrimeIntellect/INTELLECT-3` | Apache 2.0 | 128K | **BF16** | 243.8 GB | 60.9 GB/GPU | YES (1 node) |
| **Kimi-Linear-48B-A3B** | 48B | 3B | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | Modified MIT | 128K | **BF16** | 110.4 GB | 27.6 GB/GPU | YES (1 node, TP=2) |
| **DeepSeek-V3.1** | 685B | 37B | `deepseek-ai/DeepSeek-V3.1` | MIT | 128K | **DOES NOT FIT** | -- | -- | NO (3 nodes FP8) |
| **DeepSeek-V3.2-Exp** | 671B | 37B | `deepseek-ai/DeepSeek-V3.2-Exp` | MIT | 128K | **DOES NOT FIT** | -- | -- | NO (3 nodes FP8) |
| **Kimi-K2-Thinking** | 1,032B | 32B | `moonshotai/Kimi-K2-Thinking` | Modified MIT | 128K | **DOES NOT FIT** | -- | -- | NO (4 nodes INT4 QAT) |
| **Qwen3-Coder-480B** | 480B | 35B | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | Apache 2.0 | 128K | **FP8** | 552.0 GB | -- | NO (2 nodes FP8) |

### Detailed Memory Math for Frontier MoE Models

```
Llama 4 Scout (109B total, 16 experts):
  BF16:  109 × 2.0 × 1.15 = 250.7 GB → 62.7 GB/GPU @ TP=4  ✅ FITS
  FP8:   109 × 1.0 × 1.15 = 125.4 GB → 31.3 GB/GPU @ TP=4  ✅ FITS (very comfortable)

Llama 4 Maverick (400B total, 128 experts):
  BF16:  400 × 2.0 × 1.15 = 920.0 GB                         ❌ WAY TOO BIG
  FP8:   400 × 1.0 × 1.15 = 460.0 GB                         ❌ 460 > 376
  INT4:  400 × 0.5 × 1.15 = 230.0 GB → 57.5 GB/GPU @ TP=4   ✅ FITS

Qwen3-235B-A22B (235B total MoE):
  BF16:  235 × 2.0 × 1.15 = 540.5 GB                         ❌ TOO BIG
  FP8:   235 × 1.0 × 1.15 = 270.3 GB → 67.6 GB/GPU @ TP=4   ✅ FITS
  INT4:  235 × 0.5 × 1.15 = 135.1 GB → 33.8 GB/GPU @ TP=4   ✅ VERY COMFORTABLE

DeepSeek-V3.2 (685B total MoE):
  INT4:  685 × 0.5 × 1.15 = 393.9 GB                         ❌ 394 > 376 (misses by 18GB)
  → Needs 2 nodes (8x H100) minimum at INT4

DeepSeek-R1 (671B total MoE, ships FP8):
  INT4:  671 × 0.5 × 1.15 = 385.8 GB                         ❌ 386 > 376 (misses by 10GB)
  → Requires multi-node: 2 nodes (TP=4, PP=2) at INT4, or 3 nodes at FP8

MiMo-V2-Flash (309B total MoE, ships FP8):
  FP8:   309 × 1.0 × 1.15 = 355.4 GB → 88.8 GB/GPU @ TP=4   ✅ FITS (tight, 5GB headroom/GPU)
  INT4:  309 × 0.5 × 1.15 = 177.7 GB → 44.4 GB/GPU @ TP=4   ✅ COMFORTABLE

Kimi K2 (1,032B total MoE, 384 experts, 32B active):
  FP8:   1032 × 1.0 × 1.15 = 1186.8 GB                       ❌ WAY TOO BIG
  INT4:  1032 × 0.5 × 1.15 = 593.4 GB                         ❌ 593 > 376 (needs 2+ nodes)
  → Needs 2 nodes (8x H100) at INT4, or 4 nodes at FP8
  → Or use Moonshot API

Mistral Large 3 (675B total MoE, 41B active, Granular MoE):
  FP8:   675 × 1.0 × 1.15 = 776.3 GB                         ❌ WAY TOO BIG
  INT4:  675 × 0.5 × 1.15 = 388.1 GB                         ❌ 388 > 376 (misses by 12GB)
  NVFP4: ~675 × 0.45 × 1.15 ≈ 349 GB → 87.3 GB/GPU           ⚠️ MAYBE with NVFP4 support
  → Needs 2 nodes (8x H100) at INT4 for reliable serving

GLM-4.5 (355B total MoE, 32B active):
  BF16:  355 × 2.0 × 1.15 = 816.5 GB                         ❌ TOO BIG
  FP8:   355 × 1.0 × 1.15 = 408.3 GB                         ❌ 408 > 376
  INT4:  355 × 0.5 × 1.15 = 204.1 GB → 51.0 GB/GPU @ TP=4   ✅ FITS

GLM-4.5-Air (106B total MoE, 12B active):
  BF16:  106 × 2.0 × 1.15 = 243.8 GB → 60.9 GB/GPU @ TP=4   ✅ FITS
  FP8:   106 × 1.0 × 1.15 = 121.9 GB → 30.5 GB/GPU @ TP=4   ✅ VERY COMFORTABLE

MiniMax-M2/M2.1/M2.5 (229B total MoE, 256 experts, 8 active):
  FP8:   229 × 1.0 × 1.15 = 263.4 GB → 65.8 GB/GPU @ TP=4  ✅ FITS
  BF16:  229 × 2.0 × 1.15 = 526.7 GB                        ❌ TOO BIG
  → Ships FP8 natively, same architecture for M2, M2.1, M2.5
  → Following vLLM recipe: TP=4 single node, DP+EP for multi-node
  → --trust-remote-code required (custom minimax_m2 architecture)

MiniMax-M1-80k (456B total MoE):
  BF16:  456 × 2.0 × 1.15 = 1048.8 GB                       ❌ WAY TOO BIG
  → PP not supported (no SupportsPP), TP=8 cross-node fails (NCCL timeout)
  → Not deployable on this cluster. Use API only.

DeepSeek-V3.2 / R1 NOTE: HF weights ship natively in FP8 (F8_E4M3).
  Even at FP8 (native): 685 × 1.0 × 1.15 = 787.8 GB          ❌ WAY TOO BIG for 1 node
  INT4:  685 × 0.5 × 1.15 = 393.9 GB                          ❌ 394 > 376 (needs 2 nodes)
  → Requires multi-node deployment: 2 nodes (TP=4, PP=2) at INT4, or 3 nodes at FP8
```

### Deployment: Frontier MoE (Single Node)

```bash
# Llama 4 Scout at BF16 -- fits comfortably
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# GLM-4.5-Air at BF16 -- fits comfortably (similar to Scout)
python -m vllm.entrypoints.openai.api_server \
  --model zai-org/GLM-4.5-Air \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90

# Qwen3-235B-A22B at FP8 -- fits well
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-235B-A22B \
  --dtype float8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90

# MiMo-V2-Flash at FP8 (ships FP8) -- tight but fits
python -m vllm.entrypoints.openai.api_server \
  --model XiaomiMiMo/MiMo-V2-Flash \
  --dtype float8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192

# GLM-4.5 at INT4 (AWQ quantized)
python -m vllm.entrypoints.openai.api_server \
  --model zai-org/GLM-4.5 \
  --quantization awq \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90

# Llama 4 Maverick at INT4 (AWQ/GPTQ quantized)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --quantization awq \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```

### Models Too Large for 1 Node (need multi-node or API)

These models exceed 376GB even at lowest precision and require **multi-node deployment** (see Multi-Node Deployment section):

| Model | Total Params | Min Memory (INT4) | Nodes Needed (INT4) | TP x PP Config |
|-------|-------------|-------------------|---------------------|----------------|
| DeepSeek-V3.2 | 685B MoE | 394 GB | 2 (8x H100) | TP=4, PP=2 |
| DeepSeek-R1 | 671B MoE | 386 GB | 2 (8x H100) | TP=4, PP=2 |
| Mistral Large 3 | 675B MoE (41B active) | 388 GB | 2 (8x H100) | TP=4, PP=2 |
| Kimi K2 | 1,032B MoE | 593 GB | 2 (INT4) / 4 (FP8) | TP=4, PP=2-4 |

---

## 7. Ablation Models

Recommended 3-4 models for ablation studies spanning the quality/cost frontier:

| Model | Size | Type | Why |
|-------|------|------|-----|
| **Qwen3-32B** | 32B | Mid-size open weights | Best open model at ~30B; runs at BF16 TP=2 |
| **Llama 4 Maverick** | 400B MoE (17B active) | Large open weights | Largest open MoE that fits on 1 node (INT4) |
| **GPT-5** or **Claude Sonnet 4.5** | Frontier | Closed frontier | State-of-the-art closed model |

---

## 8. GPU Memory Calculations & Precision Guide

### Complete Summary Table (4x H100 94GB = 376GB total)

| Model | Params | FP32 | BF16 | FP8/INT8 | INT4 | Max Precision | TP | Fits? |
|-------|--------|------|------|----------|------|---------------|-----|-------|
| Qwen3-0.6B | 0.6B | 2.8 GB | 1.4 GB | 0.7 GB | 0.3 GB | **FP32** | 1 | YES |
| Llama 3.2 1B | 1.2B | 5.7 GB | 2.9 GB | 1.4 GB | 0.7 GB | **FP32** | 1 | YES |
| Qwen3-1.7B | 1.7B | 7.8 GB | 3.9 GB | 2.0 GB | 1.0 GB | **FP32** | 1 | YES |
| Gemma-3n-E2B | 5.44B | 25.0 GB | 12.5 GB | 6.3 GB | 3.1 GB | **FP32** | 1 | YES |
| SmolLM3-3B | 3B | 13.8 GB | 6.9 GB | 3.5 GB | 1.7 GB | **FP32** | 1 | YES |
| Ministral 3B | 3.85B | 17.7 GB | 8.9 GB | 4.4 GB | 2.2 GB | **FP32** | 1 | YES |
| Llama 3.2 3B | 3.2B | 14.7 GB | 7.4 GB | 3.7 GB | 1.8 GB | **FP32** | 1 | YES |
| Phi-4-mini 3.8B | 3.8B | 17.5 GB | 8.7 GB | 4.4 GB | 2.2 GB | **FP32** | 1 | YES |
| Qwen3-4B | 4B | 18.4 GB | 9.2 GB | 4.6 GB | 2.3 GB | **FP32** | 1 | YES |
| Mistral 7B | 7.2B | 33.1 GB | 16.6 GB | 8.3 GB | 4.1 GB | **FP32** | 1 | YES |
| Qwen2.5-7B | 7.6B | 35.0 GB | 17.5 GB | 8.8 GB | 4.4 GB | **FP32** | 1 | YES |
| OLMo 3 7B | 7B | 32.2 GB | 16.1 GB | 8.1 GB | 4.0 GB | **FP32** | 1 | YES |
| Llama 3.1 8B | 8B | 36.8 GB | 18.4 GB | 9.2 GB | 4.6 GB | **FP32** | 1 | YES |
| Qwen3-8B | 8B | 36.8 GB | 18.4 GB | 9.2 GB | 4.6 GB | **FP32** | 1 | YES |
| DS-R1-Distill-8B | 8B | 36.8 GB | 18.4 GB | 9.2 GB | 4.6 GB | **FP32** | 1 | YES |
| Apertus 8B | 8B | 36.8 GB | 18.4 GB | 9.2 GB | 4.6 GB | **FP32** | 1 | YES |
| Ministral 8B | 8B | 36.8 GB | 18.4 GB | 9.2 GB | 4.6 GB | **FP32** | 1 | YES |
| Gemma 3 12B | 12B | 55.2 GB | 27.6 GB | 13.8 GB | 6.9 GB | **FP32** | 1 | YES |
| Phi-4 14B | 14B | 64.4 GB | 32.2 GB | 16.1 GB | 8.1 GB | **FP32** | 1 | YES |
| Ministral 14B | 14B | 64.4 GB | 32.2 GB | 16.1 GB | 8.1 GB | **FP32** | 1 | YES |
| Magistral Small 24B | 24B | 110.4 GB | 55.2 GB | 27.6 GB | 13.8 GB | **BF16** (1 GPU) | 1 | YES |
| Gemma 2 27B | 27B | 124.2 GB | 62.1 GB | 31.1 GB | 15.5 GB | **BF16** (1 GPU) | 1 | YES |
| Qwen3-30B-A3B (MoE) | 30B | 138.0 GB | 69.0 GB | 34.5 GB | 17.3 GB | **BF16** (1 GPU) | 1 | YES |
| Qwen3-32B | 32B | 147.2 GB | 73.6 GB | 36.8 GB | 18.4 GB | **BF16** (tight) | 1-2 | YES |
| DS-R1-Distill-32B | 32B | 147.2 GB | 73.6 GB | 36.8 GB | 18.4 GB | **BF16** (tight) | 1-2 | YES |
| OLMo 3.1 32B | 32B | 147.2 GB | 73.6 GB | 36.8 GB | 18.4 GB | **BF16** (tight) | 1-2 | YES |
| Llama 3.3 70B | 70.6B | 324.8 GB | 162.4 GB | 81.2 GB | 40.6 GB | **BF16** (TP=4) | 4 | YES |
| Qwen2.5-72B | 72.7B | 334.4 GB | 167.2 GB | 83.6 GB | 41.8 GB | **BF16** (TP=4) | 4 | YES |
| DS-R1-Distill-70B | 70B | 322.0 GB | 161.0 GB | 80.5 GB | 40.3 GB | **BF16** (TP=4) | 4 | YES |
| K2-V2 70B | 70B | 322.0 GB | 161.0 GB | 80.5 GB | 40.3 GB | **BF16** (TP=4) | 4 | YES |
| Apertus 70B | 70B | 322.0 GB | 161.0 GB | 80.5 GB | 40.3 GB | **BF16** (TP=4) | 4 | YES |
| **Llama 4 Scout** | 109B | 501.4 GB | 250.7 GB | 125.4 GB | 62.7 GB | **BF16** (TP=4) | 4 | YES |
| Qwen3-235B-A22B | 235B | 1081.0 GB | 540.5 GB | 270.3 GB | 135.1 GB | **FP8** (TP=4) | 4 | YES |
| GLM-4.5-Air | 106B | -- | 243.8 GB | 121.9 GB | 60.9 GB | **BF16** (TP=4) | 4 | YES |
| MiMo-V2-Flash | 309B | -- | -- | 355.4 GB | 177.7 GB | **FP8** (tight) | 4 | YES |
| GLM-4.5 | 355B | -- | -- | 408.3 GB | 204.1 GB | **INT4** (TP=4) | 4 | YES |
| **Llama 4 Maverick** | 400B | -- | 920.0 GB | 460.0 GB | 230.0 GB | **INT4** (TP=4) | 4 | YES |
| DeepSeek-R1 | 671B | -- | -- | 771.7 GB | 385.8 GB | **NO FIT** | -- | NO |
| Mistral Large 3 | 675B MoE | -- | -- | 776.3 GB | 388.1 GB | **NO FIT** | -- | NO |
| DeepSeek-V3.2 | 685B | -- | -- | 787.8 GB | 393.9 GB | **NO FIT** | -- | NO |
| Kimi K2 | 1,032B | -- | -- | 1186.8 GB | 593.4 GB | **NO FIT** | -- | NO |

### FP8 vs INT8 on H100

H100 has **native FP8 Tensor Core support**. When a model's max precision is FP8/INT8, **always prefer FP8**:
- Same memory as INT8 (1 byte/param)
- Near-BF16 quality (better than INT8)
- 2x throughput over BF16 on H100 Tensor Cores
- Supported by vLLM (`--dtype float8`), SGLang, TGI

### KV Cache Memory (limits max context length)

```
KV cache per token (BF16) = 2 × layers × kv_heads × head_dim × 2 bytes

Llama 8B  (32 layers, 8 KV heads, dim=128): 0.125 MB/token → 4K=0.5GB, 32K=3.9GB
Llama 70B (80 layers, 8 KV heads, dim=128): 0.3125 MB/token → 4K=1.2GB, 32K=9.8GB, 131K=40GB
Llama 405B (126 layers, 8 KV heads, dim=128): 0.492 MB/token → 4K=1.9GB, 32K=15.4GB

DeepSeek MoE (MLA attention): ~93% KV cache reduction vs standard MHA
```

---

## 9. LiteLLM Integration Reference

### Self-Hosted Models (via vLLM OpenAI-compatible server)

```python
# All self-hosted models served via vLLM use the openai/ prefix pointing to local server
import litellm

# Configure local vLLM endpoint
response = litellm.completion(
    model="openai/meta-llama/Llama-3.3-70B-Instruct",  # model name from vLLM
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:8000/v1",  # vLLM server
    api_key="EMPTY"  # vLLM doesn't need a key
)
```

### API-Hosted Models

```python
# OpenAI
"openai/gpt-5"
"openai/gpt-5-mini"
"openai/o3-mini"
"openai/gpt-4o"
"openai/gpt-4o-mini"
"openai/gpt-4.1"
"openai/gpt-4.1-mini"
"openai/gpt-4.1-nano"

# Anthropic
"anthropic/claude-opus-4-5-20250929"       # VERIFY date suffix
"anthropic/claude-sonnet-4-5-20250929"     # VERIFY date suffix
"anthropic/claude-haiku-4-5-20251001"      # VERIFY date suffix
"anthropic/claude-opus-4-20250514"
"anthropic/claude-sonnet-4-20250514"

# Google Gemini
"gemini/gemini-3-pro"                      # VERIFY
"gemini/gemini-2.5-flash-preview-04-17"
"gemini/gemini-2.5-pro-preview-03-25"
"gemini/gemini-2.0-flash"

# xAI
"xai/grok-4"                               # VERIFY
"xai/grok-4.1"                             # VERIFY
"xai/grok-3"

# Mistral
"mistral/mistral-large-latest"

# Cohere
"cohere/command-r-plus"

# DeepSeek (direct API -- very cheap)
"deepseek/deepseek-chat"                   # V3.2
"deepseek/deepseek-reasoner"               # R1
```

### Third-Party Hosted Open Models (alternative to self-hosting)

```python
# Together AI
"together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
"together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
"together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo"
"together_ai/deepseek-ai/DeepSeek-V3"
"together_ai/deepseek-ai/DeepSeek-R1"

# Fireworks AI
"fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct"
"fireworks_ai/accounts/fireworks/models/qwen2p5-72b-instruct"
"fireworks_ai/accounts/fireworks/models/deepseek-v3"

# Groq (very fast inference)
"groq/llama-3.3-70b-versatile"
"groq/llama-3.1-8b-instant"
"groq/gemma2-9b-it"
"groq/deepseek-r1-distill-llama-70b"
"groq/qwen-qwq-32b"
```

### LiteLLM Verification Script

```python
# Run this to verify all model strings against your installed LiteLLM version
import litellm
for model in litellm.model_cost.keys():
    if any(x in model for x in ["gpt-5", "claude-opus-4", "gemini-3", "grok-4"]):
        print(f"{model}: ${litellm.model_cost[model]}")
```

---

## 10. Benchmark Scores Reference

Updated February 2026. Scores from official announcements, technical reports, and third-party evaluations.

### LMSYS Chatbot Arena (ELO-style ranking, Feb 2026)

| Rank | Model | Arena ELO | Type |
|------|-------|-----------|------|
| 1 | Gemini 3 Pro | 1492 | Closed |
| 2 | Grok-4.1-Thinking | 1482 | Closed |
| 3 | Gemini 3 Flash | 1470 | Closed |
| 4 | Claude Opus 4.5 (thinking) | 1466 | Closed |
| 5 | GPT-5.2-high | 1465 | Closed |
| 6 | GPT-5.1-high | 1464 | Closed |
| 7 | Grok-4.1 | 1463 | Closed |
| 8 | Claude Opus 4.5 | 1462 | Closed |
| 9 | ERNIE-5.0 | 1461 | Closed |
| 10 | Gemini 2.5 Pro | 1460 | Closed |
| ~15 | GLM-4.7 | ~1445 | Open |

*Source: [LMSYS Arena Leaderboard](https://aidevdayindia.org/blogs/lmsys-chatbot-arena-current-rankings/lmsys-chatbot-arena-leaderboard-current-top-models.html)*

### GPQA Diamond (PhD-level science QA)

| Model | GPQA Diamond |
|-------|-------------|
| GPT-5.2 | 93.2% |
| Gemini 3 Pro | 92.6% |
| Grok 4 | 87.0% |
| GPT-5 (high) | 86.2% |
| DeepSeek-R1 (V3.1) | 81.0% |
| Qwen3-235B-A22B | ~80% |
| DeepSeek-R1 | 71.5% |
| Llama 4 Maverick | 69.8% |
| Llama 4 Scout | 57.2% |

### MMLU / MMLU-Pro (Knowledge)

| Model | MMLU | MMLU-Pro |
|-------|------|----------|
| DeepSeek-R1 | 90.8% | 84.0% |
| DeepSeek-V3 | 87.1% | -- |
| Qwen3-235B-A22B | ~88% | ~82% |
| Qwen3-32B | 83.3% | 65.5% |
| Qwen2.5-72B | 86.1% | -- |
| Llama 4 Maverick | -- | 80.5% |
| Llama 4 Scout | -- | 74.3% |
| Llama 3.3 70B | 83.6% | -- |
| Llama 3.1 8B | ~69% | -- |

### MATH-500 / AIME (Competition math)

| Model | MATH-500 | AIME '24 | AIME '25 |
|-------|----------|----------|----------|
| GPT-5 (high) | 98.1% | -- | 100% |
| Claude Sonnet 4.5 | 97.7% | -- | -- |
| DeepSeek-R1 | 97.3% | -- | -- |
| Gemini 3 Pro | -- | -- | 95% |
| Qwen3-235B-A22B | -- | 85.7% | 81.5% |
| Kimi K2-Thinking | -- | ~90% | -- |

### SWE-bench Verified (Real-world coding)

| Model | SWE-bench |
|-------|-----------|
| Qwen3 (best) | 67.0% |
| Kimi K2 | ~65% |
| Claude Sonnet 4.5 | 64.8% |
| GLM-4.5 | ~64% |
| Claude Opus 4 | 62.2% |
| GPT-5.2 | 55.6% |

### Humanity's Last Exam (HLE, 2500 expert questions)

| Model | HLE |
|-------|-----|
| Gemini 3 Pro | 37.5% |
| GPT-5 | 25.3% |
| Kimi K2-Thinking | 23.9% |

### Social Media Simulation Relevance

**Best for persona consistency and creative writing** (based on Arena creative writing sub-scores):
1. Claude Opus 4.5 / Sonnet 4.5 -- strongest at roleplay, persona maintenance, nuanced writing
2. GPT-5 / GPT-4.1 -- excellent instruction following and style variation
3. Gemini 3 Pro -- top overall quality, strong creative capabilities
4. DeepSeek-V3.2 -- best open-weight model for creative tasks
5. Qwen3-32B -- strong multilingual capability for diverse personas
6. Llama 3.3 70B -- good balance of quality and self-hostability

---

## SLURM Job Template for Model Serving

```bash
#!/bin/bash
#SBATCH --job-name "vllm_serve"
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=24:00:00
#SBATCH --output=vllm_%x_%j.out
#SBATCH --error=vllm_%x_%j.err

module purge
source /path/to/venv/bin/activate

# Adjust MODEL, DTYPE, TP based on the tables above
MODEL="meta-llama/Llama-3.3-70B-Instruct"
DTYPE="bfloat16"
TP=4
MAX_LEN=32768

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --dtype $DTYPE \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_LEN \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
```

---

## Multi-Node Deployment (2+ Nodes)

For models exceeding 376GB (single-node capacity), use **multi-node serving with Ray + vLLM**.

### Architecture

```
Strategy: TP (tensor parallel) within a node, PP (pipeline parallel) across nodes
  TP = number of GPUs per node = 4
  PP = number of nodes

Example for 2 nodes (8 GPUs total):  TP=4, PP=2
Example for 4 nodes (16 GPUs total): TP=4, PP=4
```

### Multi-Node Memory Planning

| Model | Total Params | INT4 Memory | Nodes @ INT4 (TP=4, PP=N) | FP8 Memory | Nodes @ FP8 |
|-------|-------------|-------------|---------------------------|------------|-------------|
| DeepSeek-V3.2 | 685B MoE | 394 GB | 2 (8 GPUs) | 788 GB | 3 (12 GPUs) |
| DeepSeek-R1 | 671B MoE | 386 GB | 2 (8 GPUs) | 772 GB | 3 (12 GPUs) |
| Mistral Large 3 | 675B MoE (41B active) | 388 GB | 2 (8 GPUs) | 776 GB | 3 (12 GPUs) |
| Kimi K2 | 1,032B MoE | 593 GB | 2 (8 GPUs) | 1187 GB | 4 (16 GPUs) |
| Kimi-K2-Thinking | 1,032B MoE | 593 GB | 2 (8 GPUs) | 1187 GB | 4 (16 GPUs, INT4 QAT) |
| GLM-4.6 | 357B MoE | 205 GB | 1 (4 GPUs) | 411 GB | 2 (8 GPUs) |
| LongCat-Flash-Chat | 562B MoE | 323 GB | 1 (4 GPUs) | 646 GB | 2 (8 GPUs) |
| DeepSeek-V3.1 | 685B MoE | 394 GB | 2 (8 GPUs) | 788 GB | 3 (12 GPUs) |
| DeepSeek-V3.2-Exp | 671B MoE | 386 GB | 2 (8 GPUs) | 772 GB | 3 (12 GPUs) |

### SLURM Script: Multi-Node vLLM with Ray

```bash
#!/bin/bash
#SBATCH --job-name "vllm_multinode"
#SBATCH -N 2                          # Number of nodes (adjust per model)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=48:00:00
#SBATCH --output=vllm_multi_%x_%j.out
#SBATCH --error=vllm_multi_%x_%j.err

module purge
source /path/to/venv/bin/activate  # venv with vllm + ray installed

# ── Configuration ─────────────────────────────────────────────
MODEL="deepseek-ai/DeepSeek-V3.2"
DTYPE="auto"                           # auto detects FP8 natively
TP=4                                   # tensor parallel = GPUs per node
PP=${SLURM_NNODES}                     # pipeline parallel = number of nodes
MAX_LEN=131072                         # full context length

# ── Network setup ─────────────────────────────────────────────
HEAD_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_HOSTNAME} hostname --ip-address)
RAY_PORT=6379

# NCCL tuning for multi-node
export NCCL_IB_DISABLE=0              # Enable InfiniBand (if available)
export NCCL_SOCKET_IFNAME=ib0         # Use InfiniBand interface (adjust if needed)
export NCCL_DEBUG=INFO                # Debug logging (remove for production)
export VLLM_HOST_IP=${HEAD_IP}        # Ensure vLLM and Ray use same IP

# ── Start Ray cluster ────────────────────────────────────────
echo "Starting Ray head on ${HEAD_HOSTNAME} (${HEAD_IP})"
srun -J "ray-head" -N 1 -n 1 -w ${HEAD_HOSTNAME} \
  ray start --block --head --port=${RAY_PORT} &
sleep 15

echo "Starting Ray workers on remaining nodes"
srun -J "ray-workers" -N $((SLURM_NNODES - 1)) -n $((SLURM_NNODES - 1)) \
  --exclude=${HEAD_HOSTNAME} \
  ray start --block --address=${HEAD_IP}:${RAY_PORT} &
sleep 30

# ── Launch vLLM server (runs on head node, Ray distributes) ──
echo "Launching vLLM server: ${MODEL} (TP=${TP}, PP=${PP})"
vllm serve ${MODEL} \
  --dtype ${DTYPE} \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP} \
  --max-model-len ${MAX_LEN} \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

### Alternative: Ray `symmetric-run` (Simpler, requires Ray >= 2.40)

```bash
# After Ray cluster is started (head + workers), use symmetric-run:
ray symmetric-run \
  --address ${HEAD_IP}:${RAY_PORT} \
  --min-nodes ${SLURM_NNODES} \
  --num-gpus 4 \
  -- vllm serve ${MODEL} \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size ${SLURM_NNODES}
```

### Model-Specific Multi-Node Examples

```bash
# DeepSeek-V3.2 (685B MoE, ships FP8): 2 nodes
#SBATCH -N 2
MODEL="deepseek-ai/DeepSeek-V3.2"
DTYPE="auto"  # uses native FP8 weights
TP=4; PP=2
# Memory: ~788 GB FP8 / 8 GPUs = ~98.5 GB/GPU (over 94GB!)
# → Use INT4 quantization or 3 nodes at FP8
# With INT4: ~394 GB / 8 GPUs = 49.3 GB/GPU ✅

# DeepSeek-R1 (671B MoE, ships FP8): 2 nodes
#SBATCH -N 2
MODEL="deepseek-ai/DeepSeek-R1"
# Same as V3.2 -- prefer INT4 on 2 nodes or FP8 on 3 nodes

# Kimi K2 (1,032B MoE): 2 nodes at INT4, 4 nodes at FP8
#SBATCH -N 2  # or -N 4 for FP8
MODEL="moonshotai/Kimi-K2-Instruct"
# INT4: ~593 GB / 8 GPUs = 74.2 GB/GPU ✅ (on 2 nodes)
# FP8: ~1187 GB / 16 GPUs = 74.2 GB/GPU ✅ (on 4 nodes)

# Mistral Large 3 (675B MoE, 41B active): 2 nodes minimum
#SBATCH -N 2
MODEL="mistralai/Mistral-Large-3-675B-Instruct-2512"
# INT4: ~388 GB / 8 GPUs = 48.5 GB/GPU ✅
```

### Tips for Multi-Node on MUSICA

1. **NCCL over InfiniBand**: Ensure `NCCL_IB_DISABLE=0` and correct `NCCL_SOCKET_IFNAME` for high-speed inter-node communication
2. **Test on dev queue first**: Use `#SBATCH -p dev_zen4_0768_h100x4 --qos dev_zen4_0768_h100x4 --time=00:10:00` to validate setup before long runs
3. **SSH tunnel for access**: `ssh -NL 8000:<HEAD_IP>:8000 <user>@musica.vsc.ac.at`
4. **Monitor Ray cluster**: Use `ray status` to verify all nodes and GPUs are connected before launching vLLM
5. **KV cache for full context**: Multi-node gives more total memory -- use it for full context length. DeepSeek MLA reduces KV cache by ~93% compared to standard MHA

---

## Quick Decision Matrix

| Need | Recommended Model | Precision | TP | Instances/Node |
|------|-------------------|-----------|-----|----------------|
| **Max throughput, OK quality** | Qwen3-4B or Llama 3.2 3B | FP32 | 1 | 4 parallel |
| **Good quality, fast** | Llama 3.1 8B or Qwen3-8B | BF16 | 1 | 4 parallel |
| **Strong quality, medium cost** | Qwen3-32B | BF16 | 2 | 2 parallel |
| **Best open quality, 1 node** | Llama 3.3 70B | BF16 | 4 | 1 |
| **Best open reasoning** | DS-R1-Distill-70B | BF16 | 4 | 1 |
| **Largest open on 1 node** | Llama 4 Maverick (400B) | INT4 | 4 | 1 |
| **Largest MoE at full precision** | Llama 4 Scout (109B) | BF16 | 4 | 1 |
| **Best MoE at FP8** | Qwen3-235B-A22B | FP8 | 4 | 1 |
| **Best closed frontier** | GPT-5 / Claude Sonnet 4.5 | API | -- | -- |
| **Cheapest closed API** | DeepSeek V3.2 API | API | -- | -- |
| **Multi-node open** | DeepSeek-V3.2 / Kimi K2 | INT4/FP8 | 8+ | 2+ nodes |

---

## Deployment Notes (Tested on MUSICA, vLLM 0.15.1)

### Environment

```
vLLM:        0.15.1
PyTorch:     2.9.1+cu128
CUDA driver: 13.0
Python:      3.11 (NOT 3.12 beta -- breaks C extensions)
venv:        /data/fs201045/rl41113/vllm-venv
HF cache:    /data/fs201045/rl41113/hf-cache
```

### Known Environment Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `packaging>=26.0` breaks vLLM | Version incompatibility | `pip install "packaging<25"` |
| Python 3.12.0b2 breaks C extensions | Beta Python in system | Use Python 3.11 |
| Multi-node models fail silently | Missing Ray backend flag | Always pass `--distributed-executor-backend ray` |
| Multi-node SafetensorError on worker | vLLM doesn't propagate `HF_HOME` to Ray workers | Add `--download-dir $HF_HOME` to `vllm serve` |
| Multi-node Ray sees only 4 GPUs | `ray start` pynvml auto-detect fails on SLURM | Pass `--num-gpus=4` explicitly to `ray start` |
| Phi-4-mini-flash-reasoning FAIL | `Phi4FlashForCausalLM` dropped from vLLM v0.10.2+ | No fix, skip model or use older vLLM |
| gemma-2-27b-it FA3 softcap fail | FA3 build lacks softcapping; `VLLM_ATTENTION_BACKEND=FLASHINFER` ignored in V1 engine | **Fix**: `--attention-config '{"flash_attn_version": 2}'` (FA2 supports softcapping) |
| Multimodal models (gemma-3n, Ministral vision) fail | Missing `timm` library | `pip install timm` |
| GLM-4.5 OOM on 2-node | Ships BF16 (816 GB), 102 GB/GPU exceeds 94 GB | Move to 3-node (68 GB/GPU) |
| MiniMax-M1-80k OOM on 1 node | Too large for 4x H100 (90.7 GiB/GPU); PP not supported (`SupportsPP` missing); TP=8 cross-node times out (NCCL pair closure) | **Not deployable** on this cluster without PP support. Use API only |
| FlashInfer CUTLASS rebuild race | Workers importing FlashInfer trigger ninja to check .o freshness. NFS timestamp differences cause unnecessary ~60 min rebuild, blocking all concurrent jobs | Pre-compile on login node, then `chmod -R a-w ~/.cache/flashinfer/` to prevent rebuilds |
| LongCat-Flash-FP8 OOM on 2 nodes | MoE experts loaded as BF16 (unquantized), 91.96 GiB/GPU | Needs 3 nodes (~61 GiB/GPU) |
| GLM-4.7-FP8 tokenizer broken | `Can't load tokenizer for 'zai-org/GLM-4.7-FP8'` | **FIXED** in newer transformers. Use vLLM 0.17.0 venv |
| GLM-5-FP8 / GLM-4.7-Flash / Qwen3.5-397B arch unsupported | `glm_moe_dsa` / `glm4_moe_lite` / `qwen3_5_moe` not in vLLM 0.15.1 | **FIXED** in vLLM 0.17.0. Separate venv: `/data/fs201045/rl41113/vllm-017-venv` |

### Per-Model Test Results

**Status legend**: PASS = loads + generates, FAIL = error, PENDING = awaiting job results

#### Small Models (1-4B) -- Script: `01_test_small.sh`, 1 GPU

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| Qwen3-0.6B | PASS | 1 | BF16 | ~60 | ~2 GiB | -- | |
| Qwen3-1.7B | PASS | 1 | BF16 | ~65 | ~4 GiB | -- | |
| Qwen3-4B | PASS | 1 | BF16 | 81 | 7.6 GiB | -- | |
| SmolLM3-3B | PASS | 1 | BF16 | ~65 | ~6 GiB | -- | |
| Phi-4-mini-instruct | PASS | 1 | BF16 | 65 | 7.2 GiB | -- | 128K context |
| gemma-3n-E2B-it | PASS | 1 | BF16 | 157 | 10.4 GiB | `--trust-remote-code`, needs `timm` | Multimodal, 5.44B actual |
| gemma-3-1b-it | PASS | 1 | BF16 | 69 | 1.9 GiB | `--trust-remote-code` | |
| Ministral-3-3B | PASS | 1 | auto (FP8) | 75 | 4.4 GiB | -- | Ships FP8 natively |
| Llama-3.2-1B | PASS | 1 | BF16 | ~55 | ~3 GiB | -- | |
| Llama-3.2-3B | PASS | 1 | BF16 | ~65 | ~6 GiB | -- | |

#### Medium Models (7-14B) -- Script: `02_test_medium.sh`, 1 GPU

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| Llama-3.1-8B | PASS | 1 | BF16 | ~85 | ~16 GiB | -- | |
| Qwen3-8B | PASS | 1 | BF16 | 92 | 15.3 GiB | -- | |
| Qwen3-14B | PASS | 1 | BF16 | 105 | 27.5 GiB | -- | |
| gemma-3-12b-it | PASS | 1 | BF16 | 134 | 23.3 GiB | `--trust-remote-code` | Multimodal |
| Mistral-7B-v0.3 | PASS | 1 | BF16 | ~80 | ~15 GiB | -- | |
| Ministral-3-8B | PASS | 1 | auto (FP8) | 86 | 9.8 GiB | -- | Ships FP8, multimodal |
| Ministral-3-14B | PASS | 1 | auto (FP8) | ~95 | ~14 GiB | -- | Ships FP8, multimodal |
| DS-R1-Distill-Llama-8B | PASS | 1 | BF16 | ~85 | ~16 GiB | -- | |
| Qwen2.5-7B | PASS | 1 | BF16 | ~80 | ~15 GiB | -- | |
| phi-4 | PASS | 1 | BF16 | ~85 | ~15 GiB | -- | |
| OLMo-3-7B | PASS | 1 | BF16 | ~80 | ~14 GiB | -- | |
| Apertus-8B | PASS | 1 | BF16 | ~85 | ~16 GiB | -- | |
| Mistral-Nemo | PASS | 1 | BF16 | 88 | 22.8 GiB | -- | 12B |
| Nemotron-Nano-9B-v2 | PASS | 1 | BF16 | 82 | 16.6 GiB | `--trust-remote-code` | Hybrid Mamba, gen 21.8s (slow!) |
| Phi-4-reasoning-plus | PASS | 1 | BF16 | 104 | 27.4 GiB | -- | Reasoning variant |
| Phi-4-mini-flash-reasoning | **FAIL** | 1 | BF16 | -- | -- | `--trust-remote-code` | `Phi4FlashForCausalLM` dropped from vLLM after v0.10.2 |
| internlm3-8b-instruct | PASS | 1 | BF16 | 81 | 16.5 GiB | `--trust-remote-code` | |

#### Large Models (27-40B) -- Script: `03_test_large.sh`, 1-2 GPUs

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| Qwen3-32B | PASS | 2 | BF16 | 166 | 30.6 GiB | -- | |
| Qwen3-30B-A3B | PASS | 2 | BF16 | ~165 | ~28 GiB | -- | MoE |
| DS-R1-Distill-Qwen-32B | PASS | 2 | BF16 | ~165 | ~31 GiB | -- | |
| OLMo-3.1-32B | PASS | 2 | BF16 | 161 | 30.2 GiB | -- | |
| Magistral-Small | PASS | 1 | BF16 | 135 | 44.8 GiB | -- | 24B |
| Mistral-Small-3.1-24B | PASS | 1 | BF16 | 131 | 44.8 GiB | -- | Multimodal |
| gemma-2-27b-it | **PASS** | 1 | BF16 | ~300 | 50.7 GiB | `--attention-config '{"flash_attn_version": 2}'` | FA3 lacks softcapping, **FA2 works!** |
| gemma-3-27b-it | PASS | 1 | BF16 | 172 | 51.5 GiB | `--trust-remote-code`, `VLLM_ATTENTION_BACKEND=FLASHINFER` | |
| Nemotron-3-Nano-30B-A3B | PASS | 1 | BF16 | 143 | 58.9 GiB | `--trust-remote-code` | Hybrid Mamba+Transformer MoE, gen 21.6s |
| Nemotron-Nano-12B-v2 | PASS | 1 | BF16 | ~120 | ~24 GiB | `--trust-remote-code` | |
| Qwen3-Coder-30B-A3B | PASS | 2 | BF16 | 179 | 28.5 GiB | -- | Coding MoE |

#### XL Models (70B class) -- Script: `04_test_xl.sh`, 4 GPUs TP=4

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| Llama-3.3-70B | PASS | 4 | BF16 | 212 | 32.9 GiB | -- | Sweet spot: ~40 GB/GPU |
| Qwen2.5-72B | PASS | 4 | BF16 | ~220 | ~34 GiB | -- | |
| DS-R1-Distill-Llama-70B | PASS | 4 | BF16 | 219 | 32.9 GiB | -- | Reasoning distill |
| K2-V2-Instruct | PASS | 4 | BF16 | 174 | 33.9 GiB | -- | Fully open 70B |
| Apertus-70B | PASS | 4 | BF16 | ~220 | ~34 GiB | -- | |
| Qwen3-Next-80B-A3B-Instruct | PASS | 4 | BF16 | 504 | 37.2 GiB | -- | Next-gen MoE, gen 40s (slow!) |
| Qwen3-Next-80B-A3B-Thinking | PASS | 4 | BF16 | 478 | 37.2 GiB | -- | Thinking variant, gen 10.8s |

#### New Easy Models -- Script: `new_easy.conf`, 1-2 GPUs

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| GPT-OSS-20B | **PASS** | 1 | MXFP4 | 136 | 13.5 GiB | -- | 21.5B MoE, Apache 2.0 (OpenAI), 215 tok/s |
| GPT-OSS-120B | **PASS** | 1 | MXFP4 | 214 | 64.4 GiB | -- | 120B MoE, single GPU!, Apache 2.0, 165 tok/s |
| Seed-OSS-36B | **PASS** | 2 | BF16 | 208 | 33.9 GiB | -- | 36B dense, Apache 2.0 (ByteDance), 47 tok/s |
| GLM-4.7-Flash | **PASS** | 2 | BF16 | ~132 | 28.19 GiB | `--trust-remote-code` | Needs transformers 5.3.0 or vLLM 0.17.0 |

#### Frontier 1-Node -- Script: `05_test_frontier_1node.sh`, 4 GPUs TP=4

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| Llama-4-Scout | PASS | 4 | BF16 | 611 | 53.1 GiB | -- | 109B MoE, comfortable fit |
| GLM-4.5-Air | PASS | 4 | BF16 | 484 | 49.9 GiB | `--trust-remote-code` | 106B MoE |
| MiMo-V2-Flash | PASS | 4 | FP8 | 698 | 72.8 GiB | `--trust-remote-code` | 309B MoE, VERY tight (2.7 GiB free!) |

#### New Frontier 1-Node -- Config: `new_frontier.conf` / `minimax_m25.conf`, 4 GPUs TP=4

| Model | Status | TP | Precision | Load (s) | Mem/GPU | Flags | Notes |
|-------|--------|-----|-----------|----------|---------|-------|-------|
| MiniMax-M2.5 | **PASS** | 4 | FP8 | 1294 | 53.78 GiB | `--trust-remote-code` | 229B MoE, 76.6 tok/s |
| Step-3.5-Flash-FP8 | **FAIL** (TP=4) | 4 | FP8 | -- | -- | `--trust-remote-code` | FP8 block_n=128 constraint: 1280/4=320, 320%128≠0. **Moved to 2-node DP+EP** |
| Kimi-K2.5 | **FAIL** (1N) | 4 | FP8 | -- | -- | `--trust-remote-code` | 1058B MoE, too large for 1 node. **Moved to 4-node DP+EP** |

#### Multi-Node 2N -- Script: `06_test_multinode_2n.sh`, 8 GPUs TP=4 PP=2

| Model | Status | TP | PP | Precision | Load (s) | Mem/GPU | Notes |
|-------|--------|-----|-----|-----------|----------|---------|-------|
| Qwen3-235B-A22B | **PASS** | 4 | 2 | BF16 | 600 | 55.1 GiB | 235B MoE, `--download-dir` required |
| GLM-4.5 | **OOM** | 4 | 2 | BF16 | -- | 102 GiB | Ships BF16 only, 102 GB/GPU exceeds 94 GB. Moved to 3-node or use FP8 variant |
| Maverick-FP8 | **PASS** | 4 | 2 | FP8 | 351 | 49.0 GiB | 400B MoE, ships FP8 natively |
| Step-3.5-Flash-FP8 | **PASS** | -- | -- | FP8 (DP+EP 2N) | 441 | ~25 GiB | 199B MoE, 288 experts/8 GPUs. TP=4 fails (FP8 block), TP=2 OOMs |

#### Multi-Node 3N -- Script: `07_test_multinode_3n.sh`, 12 GPUs TP=4 PP=3

| Model | Status | TP | PP | Precision | Load (s) | Notes |
|-------|--------|-----|-----|-----------|----------|-------|
| DeepSeek-V3.2 | **FAIL** | 4 | 3 | FP8 | -- | `KeyError: model.layers.X.self_attn.attn` — PP bug in `DeepseekV32ForCausalLM`. **Use DP+EP mode instead** |
| DeepSeek-R1 | **PASS** | 4 | 3 | FP8 | 581 | 685B MoE, FP8 native, FlashMLA + DeepGEMM |
| Mistral-Large-3 | **FAIL** | 4 | 3 | compressed-tensors | 390 (load OK) | Server starts but generation empty. **Needs `--tokenizer-mode mistral --config-format mistral --load-format mistral`** |
| GLM-4.5 (BF16) | **FAIL** | 4 | 3 | BF16 | -- | Ray placement group timeout (cluster degraded from earlier failures) |
| GLM-4.5-FP8 | **PASS** | 4 | 2 | FP8 | 630 | 2 nodes, 51 GB/GPU. FP8 variant fits comfortably. Script: `07c_test_glm45_fp8.sh` |

#### Multi-Node 4N -- Script: `08_test_multinode_4n.sh` / `08b_test_kimi_k2_dpep.sh`

| Model | Status | Mode | Precision | Load (s) | Mem/GPU | Notes |
|-------|--------|------|-----------|----------|---------|-------|
| Kimi-K2 (TP+PP) | **FAIL** | TP=4 PP=4 | FP8 | -- | -- | Same PP bug as V3.2: `KeyError: model.layers.46.self_attn.attn` |
| Kimi-K2 (DP+EP) | **PASS** | DP=16 EP=16 | FP8 | 371 | 73.5 GiB | 1032B MoE, 384 experts → 24/GPU. **DP+EP mode works!** |
| Kimi-K2.5 (DP+EP) | **PASS** | DP=16 EP=16 | FP8 (compressed-tensors) | 391 | ~66 GiB | 1058B MoE, 384 experts → 24/GPU. Needs `blobfile` |

#### New Multi-Node Gap Models (submitted, pending)

| Model | Status | Nodes | Mode | Precision | Mem/GPU (est) | Notes |
|-------|--------|-------|------|-----------|---------------|-------|
| GLM-4.7-FP8 | **PASS** | 2 | DP+EP | FP8 | **56.57 GiB** | 358B MoE (160 experts, 20/GPU), MIT, Arena ELO ~1441. vLLM 0.17.0, 270s load |
| Qwen3.5-397B-A17B-FP8 | **PASS** | 2 | DP+EP | FP8 | **57.16 GiB** | 403B MoE (17B active), Apache 2.0, Arena ELO ~1447. vLLM 0.17.0, 320s load, `--enforce-eager` |
| GLM-5-FP8 | **PASS** | 3 | DP+EP | FP8 | **77-80 GiB** | 754B MoE, MIT, Arena ELO 1455 (#12 overall!). vLLM 0.17.0, 310s load, gpu_mem=0.95 (tight!) |
| GLM-4.6-FP8 | **PASS** | 2 | PP | FP8 | **40.3-42.4 GiB** | 357B MoE, MIT, Arena ELO 1425 (#36), 641s load |
| LongCat-Flash-Chat-FP8 | **FAIL** (OOM) | 2 | PP | FP8 | **91.96 GiB** (OOM) | 562B MoE (ScMoE), MIT, Arena ELO 1400 (#70). MoE experts loaded as BF16 (unquantized_fused_moe_method), needs 3 nodes |
| DeepSeek-V3.1 | **PASS** | 3 | DP+EP | FP8 | **47-57 GiB** | 685B MoE, Arena ELO 1418 (#47). 290s load |
| DeepSeek-V3.2-Exp | **PASS** | 3 | DP+EP | FP8 | **70-73 GiB** | 671B MoE, Arena ELO 1423 (#39). 340s load, needs DeepGEMM + FlashInfer |
| Kimi-K2-Thinking | **PASS** | 4 | DP+EP | INT4 QAT | **55.27 GiB** | 1T MoE (1032B), ships native INT4 (compressed-tensors), Arena ELO 1430 (#34). 341s load, 17.4 GiB KV cache |
| Qwen3-Coder-480B-FP8 | **PASS** | 2 | PP | FP8 | ~50 GiB | 480B MoE (35B active), Apache 2.0, Arena ELO ~1404. 721s load |

#### Gap Frontier 1-Node -- Config: `gap_frontier.conf`, 4 GPUs TP=4

| Model | Status | TP | Precision | Flags | Notes |
|-------|--------|-----|-----------|-------|-------|
| MiniMax-M2 | **FAIL** (FlashInfer) | 4 | FP8 | `--trust-remote-code` | 229B MoE, Arena ELO ~1399. Loaded (53.78 GiB, 683s) but FlashInfer ninja build failure (ptxas E8M0 CUTLASS). **Retry after fix** |
| MiniMax-M2.1 | **PASS** | 4 | FP8 | `--trust-remote-code` | 229B MoE, Arena ELO ~1424. 53.78 GiB, 662s load, 2810s total (slow due to FlashInfer interference) |
| INTELLECT-3 | **FAIL** | 4 | BF16 | `--trust-remote-code` | Repo missing `configuration_glm4_moe.py` custom code. Skip. |
| Kimi-Linear-48B-A3B | **PASS** | 2 | BF16 | `--trust-remote-code` | 48B MoE (linear attention). 45.93 GiB/GPU, 307s load |

#### Multi-Node Retries (in progress)

| Model | Script | Mode | Status | Notes |
|-------|--------|------|--------|-------|
| DeepSeek-V3.2 | `submit_multinode.sh` | DP+EP (12 ranks) | **FAIL** (FlashInfer race) | Loaded: 70.7-73.1 GiB/GPU, 168s. torch.compile OK (20s). Stuck in warmup due to FlashInfer .so rebuild. **Retry after fix** |
| Mistral-Large-3 | `submit_multinode.sh` | DP+EP (12 ranks) + mistral flags | **FAIL** (FlashInfer race) | Loaded OK. torch.compile OK (11.3s). Stuck in warmup due to FlashInfer .so rebuild. **Retry after fix** |
| GLM-4.5-FP8 | `07c_test_glm45_fp8.sh` | TP=4 PP=2 (2 nodes) | **PASS** | FP8 variant fits on 2 nodes (51 GB/GPU, 630s load) |

### Multi-Node Deployment Guide

#### Pipeline Parallel (PP) vs Data Parallel + Expert Parallel (DP+EP)

**PP mode** (TP=4 PP=N, Ray backend):
- Splits model layers across pipeline stages
- Simple: single `vllm serve` command
- **BUG**: `DeepseekV32ForCausalLM` and some MoE models hit `KeyError` in layer mapping with vLLM 0.15.1
- Works for: DeepSeek-R1, Qwen3-235B, Maverick, most non-DeepSeek-V3 models

**DP+EP mode** (data-parallel-size=N, enable-expert-parallel):
- Each GPU holds fraction of experts + shared params
- Avoids PP entirely → no layer mapping bug
- Launches separate `vllm serve` per node (no Ray)
- Higher throughput for MoE models (parallel expert dispatch)
- Works for: Kimi-K2, Kimi-K2.5, Kimi-K2-Thinking, DeepSeek-V3.2, DeepSeek-V3.2-Exp, Step-3.5-Flash-FP8

**When to use which:**
- Default to PP mode for non-DeepSeek-V3 architecture models
- Use DP+EP for DeepSeek-V3/V3.2 architecture models (including Kimi-K2)
- Use DP+EP when PP mode hits layer mapping errors

### Required Flags Quick Reference

| Flag | Models That Need It |
|------|-------------------|
| `--trust-remote-code` | gemma-3n-*, gemma-3-*, GLM-4.5*, GLM-4.6*, GLM-4.7*, GLM-5*, MiMo-V2-Flash, Nemotron-*, internlm3-*, Step-3.5-Flash, Kimi-K2*, Kimi-Linear-*, MiniMax-M*, INTELLECT-3, LongCat-*, Phi-4-mini-flash-reasoning |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | gemma-3-27b-it (softcapping workaround) |
| `--distributed-executor-backend ray` | ALL multi-node models using PP mode (not needed for DP+EP) |
| `--download-dir $HF_HOME` | ALL multi-node models (Ray doesn't propagate HF_HOME) |
| `--enable-expert-parallel` | DP+EP mode models (Kimi-K2, V3.2) |
| `--tokenizer-mode/config-format/load-format mistral` | Mistral-Large-3-675B (native Mistral format, not safetensors) |
| `pip install timm` | gemma-3n-E2B-it, gemma-3n-E4B-it (multimodal) |
| `pip install blobfile` | Kimi-K2 (tiktoken dependency) |
