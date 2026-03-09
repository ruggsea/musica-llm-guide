#!/usr/bin/env python3
"""Test that a model can be loaded by vLLM and generate one token.

Usage:
    python test_model_load.py --model <hf_id> [--dtype auto] [--tp 1] [--max-model-len 4096]

Prints:
    - Model load success/failure
    - Per-GPU memory usage
    - One-token generation latency
    - Total wall time
"""

import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Test vLLM model loading")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dtype", default="auto", help="Data type: auto, bfloat16, float16, float8, etc.")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max context length for test")
    parser.add_argument("--quantization", default=None, help="Quantization method: awq, gptq, etc.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"MODEL LOAD TEST: {args.model}")
    print(f"  dtype={args.dtype}, tp={args.tp}, max_model_len={args.max_model_len}")
    if args.quantization:
        print(f"  quantization={args.quantization}")
    print(f"=" * 70)

    # Print GPU info
    try:
        import torch
        gpu_count = torch.cuda.device_count()
        print(f"\nGPUs available: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({total:.1f} GB)")
    except Exception as e:
        print(f"GPU info error: {e}")

    # Load model
    t0 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading model...")
    try:
        from vllm import LLM, SamplingParams

        kwargs = dict(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.90,
            trust_remote_code=args.trust_remote_code,
        )
        if args.quantization:
            kwargs["quantization"] = args.quantization

        llm = LLM(**kwargs)
        t_load = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] Model loaded in {t_load:.1f}s")
    except Exception as e:
        t_load = time.time() - t0
        print(f"\n{'!' * 70}")
        print(f"LOAD FAILED after {t_load:.1f}s: {type(e).__name__}: {e}")
        print(f"{'!' * 70}")
        sys.exit(1)

    # Print memory after load
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {alloc:.1f} GB allocated, {reserved:.1f} GB reserved / {total:.1f} GB total ({alloc/total*100:.0f}% used)")
    except Exception as e:
        print(f"Memory info error: {e}")

    # Generate one response
    t1 = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] Generating test output...")
    try:
        sampling_params = SamplingParams(max_tokens=32, temperature=0.7)
        outputs = llm.generate(["Hello, my name is"], sampling_params)
        t_gen = time.time() - t1
        text = outputs[0].outputs[0].text
        tokens = len(outputs[0].outputs[0].token_ids)
        print(f"[{time.strftime('%H:%M:%S')}] Generated {tokens} tokens in {t_gen:.2f}s ({tokens/t_gen:.1f} tok/s)")
        print(f"  Output: {text[:200]}")
    except Exception as e:
        t_gen = time.time() - t1
        print(f"Generation FAILED after {t_gen:.1f}s: {type(e).__name__}: {e}")
        sys.exit(2)

    # Summary
    t_total = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"RESULT: PASS")
    print(f"  Model:      {args.model}")
    print(f"  Load time:  {t_load:.1f}s")
    print(f"  Gen time:   {t_gen:.2f}s")
    print(f"  Total:      {t_total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
