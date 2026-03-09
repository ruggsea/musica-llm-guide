# Running Open-Source LLMs on VSC MUSICA

A practical guide for deploying every open-source LLM from the [LMSYS Chatbot Arena](https://lmarena.ai/) leaderboard on the **VSC MUSICA cluster** (Vienna Scientific Cluster).

> **Last updated**: March 2026 | **vLLM**: 0.15.1 | **PyTorch**: 2.9.1+cu128 | **CUDA driver**: 13.0

---

## Hardware

```
Node:   4x NVIDIA H100 SXM5, 94 GB HBM3 each (376 GB total VRAM)
CPU:    192 cores AMD EPYC Zen4
RAM:    768 GB DDR5
Interconnect: NVLink 4.0 (intra-node), InfiniBand (inter-node)
Partition: zen4_0768_h100x4 (112 nodes)
```

### Memory Formula

```
Model Memory (GB) = Parameters (B) x Bytes_per_param x 1.15 (overhead)

Bytes per param:  BF16 = 2.0  |  FP8 = 1.0  |  INT4 = 0.5
Overhead (15%):   KV cache, activations, CUDA context, framework buffers
```

---

## Quick Reference: What Fits Where

| Category | Examples | Nodes | GPUs | Precision | Per-GPU |
|----------|---------|-------|------|-----------|---------|
| Small (1-4B) | Qwen3-4B, Llama-3.2-3B | 1 | 1 | BF16 | <10 GB |
| Medium (7-14B) | Llama-3.1-8B, Qwen3-14B | 1 | 1 | BF16 | <32 GB |
| Large (27-40B) | Qwen3-32B, gemma-2-27b | 1 | 1-2 | BF16 | 30-50 GB |
| XL (70B) | Llama-3.3-70B, Qwen2.5-72B | 1 | 4 (TP=4) | BF16 | ~34 GB |
| Frontier MoE | Scout (109B), GLM-4.5-Air (106B) | 1 | 4 (TP=4) | BF16 | 50-63 GB |
| Frontier FP8 | MiMo-V2-Flash (309B), MiniMax-M2.5 (229B) | 1 | 4 (TP=4) | FP8 | 55-73 GB |
| Multi-node 2N | Qwen3-235B, Maverick (400B), Qwen3.5-397B | 2 | 8 | FP8/BF16 | 49-68 GB |
| Multi-node 3N | DeepSeek-R1/V3.1/V3.2, Mistral-Large-3 | 3 | 12 (DP+EP) | FP8 | 47-73 GB |
| Multi-node 4N | Kimi-K2 (1032B), Kimi-K2.5 (1058B) | 4 | 16 (DP+EP) | FP8 | 60-74 GB |

---

## Environment Setup

```bash
# venv location (shared across all jobs)
VENV="/data/fs201045/rl41113/vllm-venv/bin/activate"
HF_HOME="/data/fs201045/rl41113/hf-cache"

# CUDA for FP8 MoE models (FlashInfer CUTLASS + DeepGEMM JIT)
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

# vLLM cache (keep off home quota)
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache

# Extended timeout for large MoE models
export VLLM_ENGINE_READY_TIMEOUT_S=1800
```

### Critical Environment Fixes

| Issue | Fix |
|-------|-----|
| `packaging>=26.0` breaks vLLM | `pip install "packaging<25"` |
| Python 3.12 beta breaks C extensions | Use Python 3.11 |
| Multimodal models (gemma-3n) need `timm` | `pip install timm` |
| Kimi-K2/K2.5 needs `blobfile` | `pip install blobfile` |

---

## Single-Node Models

### SLURM Template

```bash
#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache

vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000
```

### All Tested Single-Node Models

#### Small (1-4B) -- 1 GPU, BF16

| Model | HuggingFace ID | Mem/GPU | Load Time | Flags |
|-------|---------------|---------|-----------|-------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | ~2 GiB | ~60s | -- |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | ~4 GiB | ~65s | -- |
| Qwen3-4B | `Qwen/Qwen3-4B` | 7.6 GiB | 81s | -- |
| SmolLM3-3B | `HuggingFaceTB/SmolLM3-3B` | ~6 GiB | ~65s | -- |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | 7.2 GiB | 65s | -- |
| gemma-3n-E2B | `google/gemma-3n-E2B-it` | 10.4 GiB | 157s | `--trust-remote-code` (needs `timm`) |
| gemma-3-1b | `google/gemma-3-1b-it` | 1.9 GiB | 69s | `--trust-remote-code` |
| Ministral 3B | `mistralai/Ministral-3-3B-Instruct-2512` | 4.4 GiB | 75s | Ships FP8 |
| Llama 3.2 1B | `meta-llama/Llama-3.2-1B-Instruct` | ~3 GiB | ~55s | -- |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~6 GiB | ~65s | -- |

#### Medium (7-14B) -- 1 GPU, BF16

| Model | HuggingFace ID | Mem/GPU | Load Time | Flags |
|-------|---------------|---------|-----------|-------|
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GiB | ~85s | -- |
| Qwen3-8B | `Qwen/Qwen3-8B` | 15.3 GiB | 92s | -- |
| Qwen3-14B | `Qwen/Qwen3-14B` | 27.5 GiB | 105s | -- |
| gemma-3-12b | `google/gemma-3-12b-it` | 23.3 GiB | 134s | `--trust-remote-code` |
| Mistral-Nemo | `mistralai/Mistral-Nemo-Instruct-2407` | 22.8 GiB | 88s | -- |
| Ministral-8B | `mistralai/Ministral-3-8B-Instruct-2512` | 9.8 GiB | 86s | Ships FP8 |
| Nemotron-Nano-9B | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 16.6 GiB | 82s | `--trust-remote-code` (hybrid Mamba) |
| Phi-4-reasoning+ | `microsoft/Phi-4-reasoning-plus` | 27.4 GiB | 104s | -- |
| internlm3-8b | `internlm/internlm3-8b-instruct` | 16.5 GiB | 81s | `--trust-remote-code` |

#### Large (27-40B) -- 1-2 GPUs, BF16

| Model | HuggingFace ID | TP | Mem/GPU | Load Time | Flags |
|-------|---------------|-----|---------|-----------|-------|
| Qwen3-32B | `Qwen/Qwen3-32B` | 2 | 30.6 GiB | 166s | -- |
| OLMo-3.1-32B | `allenai/Olmo-3.1-32B-Instruct` | 2 | 30.2 GiB | 161s | -- |
| Magistral-Small | `mistralai/Magistral-Small-2509` | 1 | 44.8 GiB | 135s | -- |
| gemma-2-27b | `google/gemma-2-27b-it` | 1 | 50.7 GiB | ~300s | `--attention-config '{"flash_attn_version": 2}'` |
| gemma-3-27b | `google/gemma-3-27b-it` | 1 | 51.5 GiB | 172s | `--trust-remote-code` + `VLLM_ATTENTION_BACKEND=FLASHINFER` |
| Nemotron-30B-A3B | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | 1 | 58.9 GiB | 143s | `--trust-remote-code` |
| Qwen3-Coder-30B-A3B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 2 | 28.5 GiB | 179s | -- |
| Seed-OSS-36B | `bytedance-seed/Seed-OSS-36B-Instruct` | 2 | 33.9 GiB | 208s | -- |

#### XL (70B) -- 4 GPUs, TP=4, BF16

| Model | HuggingFace ID | Mem/GPU | Load Time | Flags |
|-------|---------------|---------|-----------|-------|
| Llama-3.3-70B | `meta-llama/Llama-3.3-70B-Instruct` | 32.9 GiB | 212s | -- |
| Qwen2.5-72B | `Qwen/Qwen2.5-72B-Instruct` | ~34 GiB | ~220s | -- |
| DS-R1-Distill-70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 32.9 GiB | 219s | -- |
| K2-V2-70B | `LLM360/K2-V2-Instruct` | 33.9 GiB | 174s | -- |

#### Frontier (100B+ MoE) -- 4 GPUs, TP=4

| Model | HuggingFace ID | Precision | Mem/GPU | Load Time | Flags |
|-------|---------------|-----------|---------|-----------|-------|
| Llama-4-Scout (109B) | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | BF16 | 53.1 GiB | 611s | -- |
| GLM-4.5-Air (106B) | `zai-org/GLM-4.5-Air` | BF16 | 49.9 GiB | 484s | `--trust-remote-code` |
| MiMo-V2-Flash (309B) | `XiaomiMiMo/MiMo-V2-Flash` | FP8 | 72.8 GiB | 698s | `--trust-remote-code` (VERY tight!) |
| MiniMax-M2 (229B) | `MiniMaxAI/MiniMax-M2` | FP8 | 53.8 GiB | 683s | `--trust-remote-code` |
| MiniMax-M2.1 (229B) | `MiniMaxAI/MiniMax-M2.1` | FP8 | 53.8 GiB | 662s | `--trust-remote-code` |
| GPT-OSS-20B (21.5B) | `openai/gpt-oss-20b` | MXFP4 | 13.5 GiB | 136s | -- |
| GPT-OSS-120B (120B) | `openai/gpt-oss-120b` | MXFP4 | 64.4 GiB | 214s | -- |
| MiniMax-M2.5 (229B) | `MiniMaxAI/MiniMax-M2.5` | FP8 | 53.8 GiB | 1294s | `--trust-remote-code` |
| Kimi-Linear-48B | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | BF16 | 45.9 GiB | 307s | `--trust-remote-code` |
| GLM-4.7-Flash (30B) | `zai-org/GLM-4.7-Flash` | BF16 | ~35 GiB | ~132s | `--trust-remote-code` |

---

## Multi-Node Models

Multi-node deployment uses two modes:

### PP Mode (Pipeline Parallel)

Uses Ray to split model layers across nodes. Simpler setup.

```
TP = 4 (GPUs per node)
PP = N (number of nodes)
Requires: --distributed-executor-backend ray
```

### DP+EP Mode (Data Parallel + Expert Parallel)

Each GPU holds a fraction of MoE experts. No Ray needed. Higher MoE throughput.

```
--data-parallel-size N    (total GPUs)
--data-parallel-size-local 4  (GPUs per node)
--enable-expert-parallel
```

**When to use which:**
- **PP**: Most non-DeepSeek models (Qwen3-235B, Maverick, GLM-4.5/4.6)
- **DP+EP**: All DeepSeek-V3 architecture models, Kimi-K2/K2.5, Mistral-Large-3, Step-3.5-Flash-FP8 (FP8 block constraint prevents TP=4)

> **Bug**: PP mode broken for `DeepseekV3ForCausalLM` and `DeepseekV32ForCausalLM` after transformers 5.3.0 (`KeyError: model.layers.X.self_attn.attn`). Also broken for `PixtralForConditionalGeneration` (Mistral-Large-3).

### Multi-Node SLURM Template (DP+EP)

```bash
#!/bin/bash
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --gres=gpu:4
#SBATCH -N 3
#SBATCH --time=06:00:00

source /data/fs201045/rl41113/vllm-venv/bin/activate
export HF_HOME=/data/fs201045/rl41113/hf-cache
export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH:-}

MODEL="deepseek-ai/DeepSeek-V3.2"
DP_SIZE=12          # total GPUs (3 nodes x 4)
DP_LOCAL=4          # GPUs per node
PORT=8000
RPC_PORT=29600

ALL_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)
HEAD_NODE=$(echo "$ALL_NODES" | head -n 1)
HEAD_IP=$(srun -N 1 -n 1 -w ${HEAD_NODE} hostname --ip-address | head -1)

# Launch headless workers (all nodes except head)
RANK=0
for NODE in $ALL_NODES; do
    [ "$NODE" = "$HEAD_NODE" ] && { RANK=$((RANK + DP_LOCAL)); continue; }
    srun -N 1 -n 1 -w ${NODE} --gpus-per-task=4 bash -c "
        source /data/fs201045/rl41113/vllm-venv/bin/activate
        export HF_HOME=/data/fs201045/rl41113/hf-cache
        export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
        export VLLM_ENGINE_READY_TIMEOUT_S=1800
        export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
        export PATH=\$CUDA_HOME/bin:\$PATH
        vllm serve $MODEL --headless \
            --data-parallel-start-rank $RANK \
            --data-parallel-size $DP_SIZE \
            --data-parallel-size-local $DP_LOCAL \
            --data-parallel-address ${HEAD_IP} \
            --data-parallel-rpc-port $RPC_PORT \
            --enable-expert-parallel \
            --download-dir /data/fs201045/rl41113/hf-cache \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.90
    " &
    RANK=$((RANK + DP_LOCAL))
done
sleep 10

# Launch master on head node
srun -N 1 -n 1 -w ${HEAD_NODE} --gpus-per-task=4 bash -c "
    source /data/fs201045/rl41113/vllm-venv/bin/activate
    export HF_HOME=/data/fs201045/rl41113/hf-cache
    export VLLM_CACHE_ROOT=/data/fs201045/rl41113/vllm-cache
    export VLLM_ENGINE_READY_TIMEOUT_S=1800
    export CUDA_HOME=/data/fs201045/rl41113/cuda-nvcc-env
    export PATH=\$CUDA_HOME/bin:\$PATH
    vllm serve $MODEL \
        --port $PORT \
        --served-model-name deepseek-v32 \
        --data-parallel-size $DP_SIZE \
        --data-parallel-size-local $DP_LOCAL \
        --data-parallel-address ${HEAD_IP} \
        --data-parallel-rpc-port $RPC_PORT \
        --enable-expert-parallel \
        --download-dir /data/fs201045/rl41113/hf-cache \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 \
        --host 0.0.0.0
" &

wait
```

### All Tested Multi-Node Models

#### 2 Nodes (8 GPUs)

| Model | HuggingFace ID | Mode | Precision | Mem/GPU | Load Time | Flags |
|-------|---------------|------|-----------|---------|-----------|-------|
| Qwen3-235B (235B MoE) | `Qwen/Qwen3-235B-A22B` | PP | BF16 | 55.1 GiB | 600s | `--download-dir` |
| Maverick (400B MoE) | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | PP | FP8 | 49.0 GiB | 290s | -- |
| GLM-4.5 (355B MoE) | `zai-org/GLM-4.5-FP8` | PP | FP8 | 51.0 GiB | 630s | `--trust-remote-code` |
| GLM-4.6 (357B MoE) | `zai-org/GLM-4.6-FP8` | PP | FP8 | 42.9 GiB | 642s | `--trust-remote-code` |
| Step-3.5-Flash (199B MoE) | `stepfun-ai/Step-3.5-Flash-FP8` | DP+EP | FP8 | ~25 GiB | 441s | `--trust-remote-code` (TP=4 fails: FP8 block constraint) |
| Qwen3-Coder-480B (480B MoE) | `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` | PP | FP8 | ~50 GiB | 721s | -- |
| Qwen3.5-397B (403B MoE) | `Qwen/Qwen3.5-397B-A17B-FP8` | DP+EP | FP8 | 57.16 GiB | 320s | Requires vLLM 0.17.0, `--enforce-eager` |

#### 3 Nodes (12 GPUs)

| Model | HuggingFace ID | Mode | Precision | Mem/GPU | Load Time | Special Flags |
|-------|---------------|------|-----------|---------|-----------|--------------|
| DeepSeek-R1 (671B MoE) | `deepseek-ai/DeepSeek-R1` | PP | FP8 | ~57 GiB | 581s | -- |
| DeepSeek-V3.1 (685B MoE) | `deepseek-ai/DeepSeek-V3.1` | DP+EP | FP8 | 47-57 GiB | 290s | Needs DeepGEMM |
| DeepSeek-V3.2 (671B MoE) | `deepseek-ai/DeepSeek-V3.2` | DP+EP | FP8 | 70-73 GiB | 300s | Needs DeepGEMM |
| Mistral-Large-3 (675B MoE) | `mistralai/Mistral-Large-3-675B-Instruct-2512` | DP+EP | FP8 | ~63 GiB | 310s | `--tokenizer-mode mistral --config-format mistral --load-format mistral` |

#### 4 Nodes (16 GPUs)

| Model | HuggingFace ID | Mode | Precision | Mem/GPU | Load Time | Flags |
|-------|---------------|------|-----------|---------|-----------|-------|
| Kimi-K2 (1032B MoE) | `moonshotai/Kimi-K2-Instruct` | DP+EP | FP8 | 73.5 GiB | 371s | `--trust-remote-code`, needs `blobfile` |
| Kimi-K2.5 (1058B MoE) | `moonshotai/Kimi-K2.5` | DP+EP | FP8 (compressed-tensors) | ~66 GiB | 391s | `--trust-remote-code`, needs `blobfile` |

---

## Known Issues & Workarounds

### Models That Don't Work

| Model | Issue | Workaround |
|-------|-------|------------|
| Phi-4-mini-flash-reasoning | `Phi4FlashForCausalLM` dropped from vLLM 0.10.2+ | Skip or use older vLLM |
| MiniMax-M1-80k (456B) | PP unsupported, TP=8 cross-node NCCL timeout | Not deployable. Use API only |
| INTELLECT-3 | Missing `configuration_glm4_moe.py` in HF repo | Skip |
| LongCat-Flash-Chat-FP8 (562B) | MoE experts loaded as BF16 despite FP8 label (OOM on 2N) | Needs 3 nodes |

### Models Needing vLLM Upgrade (0.17.0+)

| Model | Architecture | Issue |
|-------|-------------|-------|
| Qwen3.5-397B-A17B | `qwen3_5_moe` | **FIXED** in vLLM 0.17.0 -- PASS on 2 nodes DP+EP |
| GLM-5 (754B) | `glm_moe_dsa` | Needs vLLM 0.17.0 (testing pending) |
| GLM-4.7-FP8 | -- | Tokenizer **FIXED** in transformers 5.3.0, needs vLLM 0.17.0 (testing pending) |

### Special Flags Reference

| Flag | Models |
|------|--------|
| `--trust-remote-code` | gemma-3*, GLM-4.5/4.6/4.7*, MiMo-V2-Flash, Nemotron-*, internlm3-*, Step-3.5-Flash, Kimi-K2/K2.5, MiniMax-M* |
| `--attention-config '{"flash_attn_version": 2}'` | gemma-2-27b-it (FA3 lacks softcapping) |
| `VLLM_ATTENTION_BACKEND=FLASHINFER` | gemma-3-27b-it |
| `--tokenizer-mode mistral --config-format mistral --load-format mistral` | Mistral-Large-3 |
| `--download-dir $HF_HOME` | All multi-node models |
| `--distributed-executor-backend ray` | All multi-node PP mode models |

---

## FP8 MoE Prerequisites

FP8 MoE models (DeepSeek-V3.x, Mistral-Large-3, MiniMax-M2.x, Kimi-K2) need:

1. **Pre-compiled FlashInfer CUTLASS kernels** -- 182 .o files, compile on compute node via `sbatch scripts/compile_flashinfer_slurm.sh`
2. **DeepGEMM 2.3.0** -- FP8 kernel JIT. Needs `nvcc` 12.9 + `cuobjdump`
3. **CUDA_HOME** set in ALL srun commands (even workers)
4. **LIBRARY_PATH** with libcuda stubs for FlashInfer linking

### FlashInfer NFS Race Condition Fix

When multiple workers on different nodes import FlashInfer, ninja checks .o timestamps against NFS. Metadata inconsistencies trigger 60+ min rebuilds. **Critical fix**: patch `flashinfer/jit/core.py` to skip ninja when `.so` exists:

```python
# In build_and_load():
with FileLock(self.lock_path, thread_local=False):
    so_path = self.jit_library_path
    if so_path.exists() and so_path.stat().st_size > 0:
        result = self.load(so_path)
    else:
        self.build(verbose, need_lock=False)
        result = self.load(so_path)
```

---

## Tips

1. **Always prefer FP8 over INT8** on H100 -- native tensor core support, near-BF16 quality, 2x throughput
2. **SLURM GPU binding**: Use `--gpus-per-task=4` not `--gres=gpu:4` in nested srun (partition plugin overrides gres)
3. **Test on dev queue first**: `-p dev_zen4_0768_h100x4 --qos dev_zen4_0768_h100x4 --time=00:10:00`
4. **SSH tunnel for API access**: `ssh -NL 8000:<HEAD_IP>:8000 user@musica.vsc.ac.at`
5. **Monitor disk**: HF cache fills up fast. `du -sh /data/fs201045/rl41113/hf-cache/models--*/` to find large caches
6. **ZMQ port collision in DP+EP**: Random ~1/3 chance. Just resubmit the job

---

## Repository Structure

```
configs/
  tiers/           # Single-node tier configs (small/medium/large/xl/frontier)
  multinode/       # Multi-node model configs (one per model)
scripts/
  run_tier.sh      # Universal single-node runner
  submit_tier.sh   # Submit helper for tiers
  run_multinode.sh # Universal multi-node runner (PP + DP+EP)
  submit_multinode.sh  # Submit helper for multi-node
  compile_flashinfer_slurm.sh  # FlashInfer CUTLASS pre-compilation
  manage_cache.sh  # Cache management (status, cleanup)
```
