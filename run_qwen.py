#!/usr/bin/env python3
"""Run Qwen3.5-0.8B Core AI model on macOS 27 with greedy decoding.

Uses the ios-gpu host-cache variant (fp16) to avoid the MPSGraph KV-write
beta bug. The host manages KV cache + SSM states as regular I/O tensors.

Performance: ~30 tok/s decode on M3, ~17 tok/s prefill (per-token S=1).
"""

import asyncio
import time
import sys
import os

import numpy as np
import coreai.runtime as rt
from transformers import AutoTokenizer

# ---- Model paths -----------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "qwen3.5-0.8B-CoreAI")
IOS_GPU_BUNDLE = os.path.join(MODEL_DIR, "ios-gpu/qwen3_5_0_8b_ios_hc0.aimodel")
TOKENIZER_DIR = os.path.join(
    MODEL_DIR,
    "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym/tokenizer/",
)


# ---- Core inference --------------------------------------------------------


async def load_model(path: str):
    """Load an .aimodel bundle, return its main InferenceFunction."""
    model = await rt.AIModel.load(path, rt.SpecializationOptions.default())
    return model.load_function("main")


async def run_step(fn, past_k, past_v, conv_state, rec_state,
                   causal_mask, token_id, position, ctx):
    """Single forward pass. All state arrays are mutated in-place."""
    result = await fn({
        "input_ids": rt.NDArray(np.array([[token_id]], dtype=np.int32)),
        "position_ids": rt.NDArray(np.array([[position]], dtype=np.int32)),
        "causal_mask": rt.NDArray(causal_mask),
        "past_k": rt.NDArray(past_k),
        "past_v": rt.NDArray(past_v),
        "conv_state": rt.NDArray(conv_state),
        "rec_state": rt.NDArray(rec_state),
    })

    logits = result["logits"].numpy().flatten()

    # Place current-token KV into the full cache
    k_cur = result["k_cur"].numpy()
    v_cur = result["v_cur"].numpy()
    past_k[:, :, :, position, :] = k_cur[:, :, :, 0, :]
    past_v[:, :, :, position, :] = v_cur[:, :, :, 0, :]

    np.copyto(conv_state, result["conv_cur"].numpy())
    np.copyto(rec_state, result["rec_cur"].numpy())

    return logits


def sample_greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


# ---- Public API ------------------------------------------------------------


async def generate(
    prompt: str,
    bundle_path: str = IOS_GPU_BUNDLE,
    tokenizer_path: str = TOKENIZER_DIR,
    max_new_tokens: int = 128,
    ctx: int = 2048,
    verbose: bool = True,
) -> str:
    """Generate text from a prompt using the Core AI model.

    Returns the generated text (not including the prompt).
    """
    fn = await load_model(bundle_path)
    desc = fn.desc
    vocab_size = desc.output_descriptor("logits").shape[-1]

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    except Exception:
        prompt_ids = tok.encode(prompt, add_special_tokens=False)

    if verbose:
        print(f"Model loaded: vocab={vocab_size}, ctx={ctx}")
        print(f"Prompt ({len(prompt_ids)} tokens): {prompt!r}")

    # Host-managed state buffers
    past_k = np.zeros((6, 1, 2, ctx, 256), dtype=np.float16)
    past_v = np.zeros((6, 1, 2, ctx, 256), dtype=np.float16)
    conv_state = np.zeros((18, 1, 6144, 3), dtype=np.float16)
    rec_state = np.zeros((18, 1, 16, 128, 128), dtype=np.float16)

    t0 = time.time()

    # Prefill (S=1 tokens)
    for i, tid in enumerate(prompt_ids):
        mask = np.zeros((1, 1, 1, ctx + 1), dtype=np.float16)
        mask[0, 0, 0, : i + 1] = 1.0
        logits = await run_step(fn, past_k, past_v, conv_state, rec_state,
                                mask, int(tid), i, ctx)

    prefill_time = time.time() - t0
    if verbose:
        print(f"Prefill: {len(prompt_ids)} tokens in {prefill_time:.1f}s "
              f"({len(prompt_ids)/prefill_time:.1f} tok/s)")

    # Decode
    eos_id = tok.eos_token_id or 248046
    generated_ids = []
    current_id = sample_greedy(logits)
    decode_start = time.time()

    for step in range(max_new_tokens):
        if current_id == eos_id:
            break
        generated_ids.append(current_id)
        position = len(prompt_ids) + step
        if position >= ctx:
            if verbose:
                print("Context limit reached.")
            break

        mask = np.zeros((1, 1, 1, ctx + 1), dtype=np.float16)
        mask[0, 0, 0, : position + 1] = 1.0
        logits = await run_step(fn, past_k, past_v, conv_state, rec_state,
                                mask, int(current_id), position, ctx)
        current_id = sample_greedy(logits)

    elapsed = time.time() - t0
    decode_time = time.time() - decode_start
    generated_text = tok.decode(generated_ids)

    if verbose:
        n = len(generated_ids)
        print(f"Output: {generated_text}")
        print(f"Decode: {n} tokens, "
              f"{n/decode_time:.1f} tok/s" if decode_time > 0 else "",
              f"| Total: {(len(prompt_ids)+n)/elapsed:.1f} tok/s")

    return generated_text


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the capital of France?"
    asyncio.run(generate(prompt, max_new_tokens=64))
