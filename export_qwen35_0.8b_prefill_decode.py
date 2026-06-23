#!/usr/bin/env python3
"""
Re-export Qwen3.5-0.8B with prefill (chunk=64) + decode (S=1) functions.

Produces a single .aimodel with two functions:
  - prefill: input_ids [1, -1] → hidden_states (up to 64 tokens)
  - decode:  input_ids [1, 1]  → hidden_states

The inference engine switches between them: prefill for the first chunk,
decode for each subsequent token. States are managed by the host.

Usage:
  python export_qwen35_0.8b_prefill_decode.py --out-dir exports
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


class SimpleWrapper(nn.Module):
    """Simple forward wrapper: input_ids → hidden_states, no cache."""

    def __init__(self, hf_model):
        super().__init__()
        self.lm = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, use_cache=False)
        return self.lm_head(out.last_hidden_state)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-path", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--prefill-len", type=int, default=64)
    args = ap.parse_args()

    hf_path = args.hf_path
    model_name = f"qwen3_5_0_8b_prefill{args.prefill_len}_decode1"

    print(f"Loading HF model from {hf_path}...")
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path, trust_remote_code=True, dtype=torch.float16, device_map="cpu"
    )
    hf_model.eval()
    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    tc = config.text_config if hasattr(config, "text_config") else config
    vocab_size = tc.vocab_size

    print(f"  Layers: {tc.num_hidden_layers}, Hidden: {tc.hidden_size}, Vocab: {vocab_size}")

    wrapper = SimpleWrapper(hf_model)
    wrapper.eval()

    from coreai_torch import TorchConverter, get_decomp_table

    converter = TorchConverter()

    # ---- Export prefill function ----
    S_prefill = args.prefill_len
    input_ids_prefill = torch.randint(0, vocab_size, (1, S_prefill), dtype=torch.long)
    seq_dim = torch.export.Dim("seq_len", min=1, max=S_prefill)
    dynamic_shapes_prefill = {"input_ids": {1: seq_dim}}

    print(f"\nTracing prefill (seq_len up to {S_prefill})...")
    with torch.no_grad():
        ep_prefill = torch.export.export(
            wrapper,
            args=(input_ids_prefill,),
            dynamic_shapes=dynamic_shapes_prefill,
        )
    print(f"  {len(ep_prefill.graph.nodes)} nodes")

    print("  Decompositions...")
    ep_prefill_decomp = ep_prefill.run_decompositions(get_decomp_table())
    print(f"  {len(ep_prefill_decomp.graph.nodes)} nodes")

    print("  Adding prefill to converter...")
    converter.add_exported_program(
        exported_program=ep_prefill_decomp,
        input_names=["input_ids"],
        output_names=["logits"],
        function_name="prefill",
    )

    # ---- Export decode function ----
    input_ids_decode = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)

    print(f"\nTracing decode (seq_len=1)...")
    with torch.no_grad():
        ep_decode = torch.export.export(
            wrapper,
            args=(input_ids_decode,),
        )
    print(f"  {len(ep_decode.graph.nodes)} nodes")

    print("  Decompositions...")
    ep_decode_decomp = ep_decode.run_decompositions(get_decomp_table())
    print(f"  {len(ep_decode_decomp.graph.nodes)} nodes")

    print("  Adding decode to converter...")
    converter.add_exported_program(
        exported_program=ep_decode_decomp,
        input_names=["input_ids"],
        output_names=["logits"],
        function_name="decode",
    )

    # ---- Convert ----
    print("\nConverting to Core AI IR...")
    prog = converter.to_coreai()
    print("  Converter OK")

    print("Optimizing...")
    prog.optimize()
    print("  Optimize OK")

    # ---- Save ----
    out_dir = Path(args.out_dir) / model_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{model_name}.aimodel"
    print(f"Saving {aimodel}...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    # Metadata
    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": model_name,
        "assets": {"main": f"{model_name}.aimodel"},
        "language": {
            "tokenizer": "Qwen/Qwen3.5-0.8B",
            "vocab_size": vocab_size,
            "max_context_length": 4096,
            "embedded_tokenizer": True,
            "function_map": {
                "main": ["prefill", "decode"]
            },
        },
        "source": {"model_definition": "torch", "hf_model_id": "Qwen/Qwen3.5-0.8B"},
        "compression": None,
        "compilation": {
            "date": datetime.now(timezone.utc).isoformat(),
            "targets": [],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy tokenizer from the existing bundle
    src_tok = "/Users/zgd/Code/llm/core-ai/qwen3.5-0.8B-CoreAI/gpu-pipelined/qwen3_5_0_8b_decode_int8hu_perchan_sym/tokenizer/"
    dst_tok = out_dir / "tokenizer"
    if os.path.exists(src_tok) and not dst_tok.exists():
        shutil.copytree(src_tok, dst_tok)

    size_gb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(out_dir)
        for fn in fns
    ) / 1e9
    print(f"\nDone! Bundle: {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
