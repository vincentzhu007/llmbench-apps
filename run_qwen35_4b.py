#!/usr/bin/env python3
"""Run the converted Qwen3.5-4B Core AI model.

Uses the stateless forward model: re-runs the full prompt+generated
sequence each step. Simple but functional for short demos.
"""

import asyncio
import time
import sys
import os

import numpy as np
import coreai.runtime as rt
from transformers import AutoTokenizer


async def load_model(path: str):
    model = await rt.AIModel.load(path, rt.SpecializationOptions.default())
    return model.load_function("main")


async def generate(
    bundle_path: str,
    tokenizer_path: str,
    prompt: str,
    max_new_tokens: int = 16,
    max_seq_len: int = 64,
):
    print(f"Loading model: {bundle_path}")
    fn = await load_model(bundle_path)
    desc = fn.desc
    model_seq_len = desc.input_descriptor("input_ids").shape[1]
    vocab_size = desc.output_descriptor("logits").shape[-1]
    print(f"  Model seq_len={model_seq_len}, vocab={vocab_size}")
    if max_seq_len > model_seq_len:
        print(f"  Warning: requested seq_len {max_seq_len} > model seq_len {model_seq_len}")

    print(f"Loading tokenizer: {tokenizer_path}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Simple tokenization (avoid complex VLM chat template)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)

    print(f"  Prompt ({len(prompt_ids)} tokens): {prompt!r}")

    generated_ids = []
    eos_id = tok.eos_token_id or 248046

    t0 = time.time()
    for step in range(max_new_tokens):
        # Build full sequence: prompt + generated so far
        full_ids = prompt_ids + generated_ids
        if len(full_ids) > model_seq_len:
            full_ids = full_ids[-model_seq_len:]

        # Pad to model seq_len
        padded = np.zeros((1, model_seq_len), dtype=np.int32)
        padded[0, :len(full_ids)] = full_ids

        result = await fn({"input_ids": rt.NDArray(padded)})
        logits = result["logits"].numpy()

        # Get logits at the last real token position
        last_pos = min(len(full_ids) - 1, model_seq_len - 1)
        next_logits = logits[0, last_pos, :]
        next_id = int(np.argmax(next_logits))

        if next_id == eos_id:
            break
        generated_ids.append(next_id)

    elapsed = time.time() - t0
    generated_text = tok.decode(generated_ids)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Output: {generated_text}")
    print(f"{'='*60}")
    n = len(generated_ids)
    if n > 0 and elapsed > 0:
        print(f"Stats: {n} tokens in {elapsed:.1f}s ({n/elapsed:.1f} tok/s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="*", default=["The capital of France is"])
    ap.add_argument("--mode", default="int4lin",
                    choices=["fp16", "int8lin", "int4lin", "int4mix"])
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(__file__) or ".", "exports")
    dirname = f"qwen3_5_4b_forward_{args.mode}"
    bundle = os.path.join(base, dirname, f"{dirname}.aimodel")
    tokenizer = os.path.join(base, dirname, "tokenizer/")

    prompt = " ".join(args.prompt)
    asyncio.run(generate(bundle, tokenizer, prompt, max_new_tokens=args.max_tokens))
