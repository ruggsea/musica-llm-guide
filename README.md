# Running Open-Source LLMs on MUSICA

Practical guide to deploying open-source LLMs on [MUSICA](https://docs.asc.ac.at/systems/musica.html), the GPU partition of Austria's national supercomputing infrastructure operated by the [ASC](https://asc.ac.at) (Austrian Supercomputing Centre, formerly VSC).

Everything here has been tested with vLLM. Every model in the [Arena leaderboard](https://lmarena.ai/) that has open weights -- from 0.6B to 1T+ parameters -- loads and generates on this cluster.

> March 2026 | vLLM 0.15.1 + 0.17.0 | PyTorch 2.9/2.10 | CUDA 13.0

## The cluster

MUSICA (Multi-Site Computer Austria) spans three sites (Vienna, Innsbruck, Linz) with 440 total compute nodes on Lenovo Neptune direct liquid cooling. The Vienna site has the GPU partition used in this guide.

| | GPU nodes | CPU nodes |
|---|---|---|
| **Partition** | `zen4_0768_h100x4` | `zen4_0768` |
| **Nodes** | 112 | 72 |
| **GPUs** | 4x NVIDIA H100 SXM5 (94 GB HBM3 each) | -- |
| **CPUs** | 2x AMD EPYC 9654 (192 cores, 8 NUMA nodes) | same |
| **RAM** | 768 GB DDR5 | same |
| **Local NVMe** | 7.68 TB | 1.92 TB |
| **Interconnect** | 4x InfiniBand NDR200 | 1x InfiniBand NDR200 |
| **QoS** | `zen4_0768_h100x4` (72h) / `dev_zen4_0768_h100x4` (10min, 2 nodes) | same pattern |

**Storage** (per project):
- `$HOME`: 50 GB, NVMe-backed, backed up. Config/scripts only.
- `$DATA`: 10 TB default (extendable to 100 TB), tiered flash+HDD (IBM Spectrum Scale). Model weights go here.
- `$SCRATCH`: 5 TB, 4 PB all-NVMe WekaFS (1.8 TB/s read). Fast I/O, not backed up.
- `/local`: 7.68 TB node-local SSD, wiped between jobs.

**Quick memory math**: `memory = params (B) x bytes_per_param x 1.15`. BF16 = 2 bytes, FP8 = 1 byte, INT4 = 0.5 bytes. The 1.15 accounts for KV cache, activations, and CUDA context overhead.

## What fits where

| Size | Examples | Nodes | Precision | Per GPU |
|------|---------|-------|-----------|---------|
| 1-4B | Qwen3-4B, Phi-4-mini | 1 (1 GPU) | BF16 | <10 GB |
| 7-14B | Llama-3.1-8B, Qwen3-14B | 1 (1 GPU) | BF16 | <32 GB |
| 27-40B | Qwen3-32B, gemma-2-27b | 1 (1-2 GPUs) | BF16 | 30-50 GB |
| 70B | Llama-3.3-70B, Qwen2.5-72B | 1 (TP=4) | BF16 | ~34 GB |
| 100-300B MoE | Scout, MiMo-V2-Flash, MiniMax-M2.5 | 1 (TP=4) | BF16/FP8 | 50-73 GB |
| 300-500B MoE | Qwen3-235B, Maverick, Qwen3.5-397B | 2 | FP8 | 49-68 GB |
| 600-700B MoE | DeepSeek-V3.2, Mistral-Large-3 | 3 (DP+EP) | FP8 | 47-73 GB |
| 1T+ MoE | Kimi-K2, Kimi-K2.5 | 4 (DP+EP) | FP8 | 60-74 GB |

## Setup

```bash
# activate the venv (shared across all jobs)
source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache

# FP8 MoE models need these for FlashInfer/DeepGEMM JIT
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

# keep caches off the tiny home quota
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

Watch out:
- `packaging>=26.0` breaks vLLM -- pin `packaging<25`
- Use Python 3.11, not 3.12 beta
- Some models need extra deps: `pip install timm` (gemma-3n), `pip install blobfile` (Kimi-K2)

## Single-node serving

Minimal SLURM script:

```bash
#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH -N 1 --gres=gpu:4
#SBATCH --time=06:00:00

source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache

vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype bfloat16 --tensor-parallel-size 4 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000
```

### Tested models (single node)

Everything below loads, generates, and serves an OpenAI-compatible API.

**Small (1-4B) -- 1 GPU, BF16**

| Model | HF ID | VRAM | Load | Notes |
|-------|-------|------|------|-------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | ~2 GiB | ~60s | |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | ~4 GiB | ~65s | |
| Qwen3-4B | `Qwen/Qwen3-4B` | 7.6 GiB | 81s | |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | ~6 GiB | ~65s | |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 7.2 GiB | 65s | |
| gemma-3n-E2B | `google/gemma-3n-E2B-it` | 10.4 GiB | 157s | `--trust-remote-code`, needs `timm` |
| gemma-3-1b | `google/gemma-3-1b-it` | 1.9 GiB | 69s | `--trust-remote-code` |
| Ministral 3B | `mistralai/Ministral-3-3B-Instruct-2512` | 4.4 GiB | 75s | ships FP8 |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | ~3 GiB | ~55s | |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~6 GiB | ~65s | |

**Medium (7-14B) -- 1 GPU, BF16**

| Model | HF ID | VRAM | Load | Notes |
|-------|-------|------|------|-------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GiB | ~85s | |
| Qwen3-8B | `Qwen/Qwen3-8B` | 15.3 GiB | 92s | |
| Qwen3-14B | `Qwen/Qwen3-14B` | 27.5 GiB | 105s | |
| gemma-3-12b | `google/gemma-3-12b-it` | 23.3 GiB | 134s | `--trust-remote-code` |
| Mistral-Nemo | `mistralai/Mistral-Nemo-Instruct-2407` | 22.8 GiB | 88s | |
| Ministral-8B | `mistralai/Ministral-3-8B-Instruct-2512` | 9.8 GiB | 86s | ships FP8 |
| Nemotron-Nano-9B | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 16.6 GiB | 82s | `--trust-remote-code` |
| Phi-4-reasoning+ | `microsoft/Phi-4-reasoning-plus` | 27.4 GiB | 104s | |
| internlm3-8b | `internlm/internlm3-8b-instruct` | 16.5 GiB | 81s | `--trust-remote-code` |

**Large (27-40B) -- 1-2 GPUs, BF16**

| Model | HF ID | TP | VRAM/GPU | Load | Notes |
|-------|-------|----|----------|------|-------|
| Qwen3-32B | `Qwen/Qwen3-32B` | 2 | 30.6 GiB | 166s | |
| OLMo-3.1-32B | `allenai/Olmo-3.1-32B-Instruct` | 2 | 30.2 GiB | 161s | |
| Magistral-Small | `mistralai/Magistral-Small-2509` | 1 | 44.8 GiB | 135s | |
| gemma-2-27b | `google/gemma-2-27b-it` | 1 | 50.7 GiB | ~300s | needs `--attention-config '{"flash_attn_version": 2}'` |
| gemma-3-27b | `google/gemma-3-27b-it` | 1 | 51.5 GiB | 172s | `--trust-remote-code` + `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| Seed-OSS-36B | `ByteDance-Seed/Seed-OSS-36B-Instruct` | 2 | 33.9 GiB | 208s | |

**XL (70B) -- TP=4, BF16**

| Model | HF ID | VRAM/GPU | Load | Notes |
|-------|-------|----------|------|-------|
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 32.9 GiB | 212s | |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | ~34 GiB | ~220s | |
| DS-R1-Distill-70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 32.9 GiB | 219s | |
| K2-V2-70B | `LLM360/K2-V2-Instruct` | 33.9 GiB | 174s | |

**Frontier MoE (100B+) -- TP=4, single node**

| Model | HF ID | Precision | VRAM/GPU | Load | Notes |
|-------|-------|-----------|----------|------|-------|
| Llama-4-Scout (109B) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | BF16 | 53.1 GiB | 611s | |
| GLM-4.5-Air (106B) | `zai-org/GLM-4.5-Air` | BF16 | 49.9 GiB | 484s | `--trust-remote-code` |
| MiMo-V2-Flash (309B) | `XiaomiMiMo/MiMo-V2-Flash` | FP8 | 72.8 GiB | 698s | `--trust-remote-code`, very tight! |
| MiniMax-M2.5 (229B) | `MiniMaxAI/MiniMax-M2.5` | FP8 | 53.8 GiB | 1294s | `--trust-remote-code` |
| GPT-OSS-20B (21.5B) | `openai/gpt-oss-20b` | MXFP4 | 13.5 GiB | 136s | |
| GPT-OSS-120B (120B) | `openai/gpt-oss-120b` | MXFP4 | 64.4 GiB | 214s | |
| Kimi-Linear-48B | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | BF16 | 45.9 GiB | 307s | `--trust-remote-code` |
| GLM-4.7-Flash (30B) | `zai-org/GLM-4.7-Flash` | BF16 | ~35 GiB | ~132s | `--trust-remote-code` |

## Multi-node serving

Two modes, depending on the model:

**PP (Pipeline Parallel)** -- Ray splits layers across nodes. Use for most models.
```
vllm serve <model> --tensor-parallel-size 4 --pipeline-parallel-size N \
    --distributed-executor-backend ray
```

**DP+EP (Data Parallel + Expert Parallel)** -- each GPU gets a slice of MoE experts. No Ray. Better throughput for MoE. Use for DeepSeek, Kimi, Mistral-Large-3, and anything where PP is broken.
```
# workers (headless, one per non-head node)
vllm serve <model> --headless --data-parallel-start-rank R \
    --data-parallel-size N --data-parallel-size-local 4 \
    --data-parallel-address <HEAD_IP> --enable-expert-parallel

# master (head node)
vllm serve <model> --data-parallel-size N --data-parallel-size-local 4 \
    --data-parallel-address <HEAD_IP> --enable-expert-parallel --port 8000
```

PP is broken for DeepSeek-V3/V3.2, Mistral-Large-3 (`PixtralForConditionalGeneration`), and anything after transformers 5.3.0 that uses `DeepseekV3ForCausalLM`. Use DP+EP for all of those.

### Tested multi-node models

**2 nodes (8 GPUs)**

| Model | HF ID | Mode | VRAM/GPU | Load | Notes |
|-------|-------|------|----------|------|-------|
| Qwen3-235B | `Qwen/Qwen3-235B-A22B` | PP | 55.1 GiB | 600s | |
| Maverick (400B) | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | PP | 49.0 GiB | 351s | |
| GLM-4.5 (355B) | `zai-org/GLM-4.5-FP8` | PP | 51.0 GiB | 630s | `--trust-remote-code` |
| GLM-4.6 (357B) | `zai-org/GLM-4.6-FP8` | PP | 42.4 GiB | 641s | `--trust-remote-code` |
| Step-3.5-Flash (199B) | `stepfun-ai/Step-3.5-Flash-FP8` | DP+EP | ~25 GiB | 441s | `--trust-remote-code` |
| Qwen3-Coder-480B | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | PP | ~50 GiB | 721s | |
| Qwen3.5-397B | `Qwen/Qwen3.5-397B-A17B-FP8` | DP+EP | 57.2 GiB | 320s | vLLM 0.17.0, `--enforce-eager` |
| GLM-4.7 (358B) | `zai-org/GLM-4.7-FP8` | DP+EP | 56.6 GiB | 270s | vLLM 0.17.0, `--trust-remote-code` |

**3 nodes (12 GPUs)**

| Model | HF ID | Mode | VRAM/GPU | Load | Notes |
|-------|-------|------|----------|------|-------|
| DeepSeek-R1 (671B) | `deepseek-ai/DeepSeek-R1` | PP | ~57 GiB | 581s | |
| DeepSeek-V3.1 (685B) | `deepseek-ai/DeepSeek-V3.1` | DP+EP | 47-57 GiB | 290s | needs DeepGEMM |
| DeepSeek-V3.2 (671B) | `deepseek-ai/DeepSeek-V3.2` | DP+EP | 70-73 GiB | 300s | needs DeepGEMM |
| DeepSeek-V3.2-Exp (671B) | `deepseek-ai/DeepSeek-V3.2-Exp` | DP+EP | 70-73 GiB | 340s | needs DeepGEMM |
| Mistral-Large-3 (675B) | `mistralai/Mistral-Large-3-675B-Instruct-2512` | DP+EP | ~63 GiB | 310s | `--tokenizer-mode mistral --config-format mistral --load-format mistral` |
| GLM-5 (754B) | `zai-org/GLM-5-FP8` | DP+EP | 77-80 GiB | 310s | vLLM 0.17.0, `--trust-remote-code`, gpu_mem=0.95 |

**4 nodes (16 GPUs)**

| Model | HF ID | Mode | VRAM/GPU | Load | Notes |
|-------|-------|------|----------|------|-------|
| Kimi-K2 (1032B) | `moonshotai/Kimi-K2-Instruct` | DP+EP | 73.5 GiB | 371s | needs `blobfile` |
| Kimi-K2.5 (1058B) | `moonshotai/Kimi-K2.5` | DP+EP | ~66 GiB | 391s | needs `blobfile` |
| Kimi-K2-Thinking (1032B) | `moonshotai/Kimi-K2-Thinking` | DP+EP | 55.3 GiB | 341s | INT4 QAT, `--trust-remote-code` |

## Things that don't work

| Model | Why | Workaround |
|-------|-----|------------|
| MiniMax-M1-80k (456B) | No PP support, TP=8 cross-node NCCL timeout | API only |
| LongCat-Flash-Chat-FP8 (562B) | MoE experts loaded as BF16 despite FP8 label, OOM on 2 nodes | needs 3 nodes, or skip |
| INTELLECT-3 | Missing config file in HF repo | skip |
| Phi-4-mini-flash-reasoning | Architecture dropped from vLLM 0.10.2+ | older vLLM or skip |

## FP8 MoE: the hard part

This is where most of the pain was. FP8 MoE models (DeepSeek, Kimi, MiniMax, Mistral-Large-3) need:

1. **Pre-compiled FlashInfer CUTLASS kernels** -- 182 object files that take 30+ min to JIT compile. With multiple workers on NFS fighting over ninja locks, it's a disaster. Compile once on a compute node before running anything
2. **DeepGEMM 2.3.0** -- FP8 kernel JIT. Needs `nvcc` 12.9 (not 13.1, that breaks CUTLASS) + `cuobjdump`
3. **CUDA_HOME** and **LIBRARY_PATH** set in every srun command, including workers
4. **FlashInfer NFS race condition fix** -- patch `flashinfer/jit/core.py` to skip ninja rebuild when the .so already exists. Without this, every job wastes 30+ minutes and sometimes OOMs

## Gotchas

- **Home quota is 50 GB**. SLURM jobs die with signal 53 when it fills up. Keep `~/.cache/{uv,vllm,pip}` and `~/.local/share/mamba/pkgs` clean
- **FP8 > INT8 on H100** always. Native tensor core support, near-BF16 quality, 2x throughput
- **`--gpus-per-task=4`** not `--gres=gpu:4` in nested srun. The partition plugin overrides gres
- **ZMQ port collision in DP+EP** happens ~1/3 of the time. Just resubmit
- **Monitor disk**: the HF cache grows fast. `du -sh /data/.../hf-cache/hub/models--*/` to find the big ones
- **SSH tunnel for API**: `ssh -NL 8000:<node>:8000 user@musica.vsc.ac.at`

## Scripts

This repo includes ready-to-use SLURM scripts in `scripts/`:

| Script | Use case | Example |
|--------|----------|---------|
| `serve_single_node.sh` | 1 node, any model up to ~300B MoE | `sbatch scripts/serve_single_node.sh meta-llama/Llama-3.3-70B-Instruct bfloat16 4` |
| `serve_multinode_pp.sh` | Multi-node with Ray Pipeline Parallel | `sbatch -N 2 scripts/serve_multinode_pp.sh Qwen/Qwen3-235B-A22B auto 4` |
| `serve_multinode_dpep.sh` | Multi-node DP+EP for MoE models | `sbatch -N 3 scripts/serve_multinode_dpep.sh deepseek-ai/DeepSeek-V3.2 auto` |

All scripts expose an OpenAI-compatible API on port 8000. Access via SSH tunnel:
```bash
ssh -NL 8000:<node>:8000 user@musica.vsc.ac.at
curl http://localhost:8000/v1/models
```

## Models needing `--trust-remote-code`

gemma-3*, GLM-4.5/4.6/4.7/5*, MiMo-V2-Flash, Nemotron-*, internlm3-*, Step-3.5-Flash, Kimi-K2/K2.5, MiniMax-M*

## Special flags

| Flag | When |
|------|------|
| `--tokenizer-mode mistral --config-format mistral --load-format mistral` | Mistral-Large-3 |
| `--attention-config '{"flash_attn_version": 2}'` | gemma-2-27b (FA3 lacks softcapping) |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | gemma-3-27b |
| `--enforce-eager` | Qwen3.5-397B (vLLM 0.17.0) |
