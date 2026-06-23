#!/usr/bin/env python3
"""
Convert Qwen3.5-4B to Core AI .aimodel with optional weight quantization.

Modes:
  fp16    - no quantization (~16 GB)
  int8lin - int8 linear per-block-32 (~8 GB, ship config, gate 16/16)
  int4lin - int4 linear per-block-32 (~4 GB, may flip gate)
  int4mix - mixed: SSM projections int8 + FFN/head int4 (~5 GB, best int4)

Usage:
  python convert_qwen35_4b.py --mode int4lin --seq-len 64 --out-dir exports
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


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

class SimpleWrapper(nn.Module):
    """input_ids [1,S] → logits [1,S,vocab]."""

    def __init__(self, hf_model):
        super().__init__()
        self.lm = hf_model.model
        self.lm_head = hf_model.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, use_cache=False)
        return self.lm_head(out.last_hidden_state)


# ---------------------------------------------------------------------------
# Quantization configs
# ---------------------------------------------------------------------------

def _base_exclusions() -> dict:
    """Module type/name exclusions to keep at fp16."""
    return {
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {
            # Skip ALL norm modules (1D weights break per-block axis-1)
            r".*norm$": None,
            r".*_norm$": None,
            r".*input_layernorm$": None,
            r".*post_attention_layernorm$": None,
        },
    }


def config_int8lin() -> dict:
    """int8 linear per-block-32 — ship config, gate 16/16."""
    cfg = {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int8",
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        **_base_exclusions(),
    }
    cfg["module_name_configs"][r".*lm_head$"] = None  # tied head → fp16
    return cfg


def config_int4lin() -> dict:
    """int4 linear per-block-32 — smallest, may flip gate on SSM layers."""
    cfg = {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int4",
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        **_base_exclusions(),
    }
    cfg["module_name_configs"][r".*lm_head$"] = None  # tied head → fp16
    return cfg


def config_int4mix() -> dict:
    """
    Mixed precision: SSM projections → int8 (sensitive),
    FFN + everything else → int4.
    """
    def _layer_cfg(dtype: str, qscheme: str) -> dict:
        return {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": qscheme,
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        }

    cfg = {
        "execution_mode": "eager",
        "global_config": _layer_cfg("int4", "symmetric_with_clipping"),
        **_base_exclusions(),
    }
    # SSM-sensitive projections → int8 (no clipping — symmetric absmax)
    cfg["module_name_configs"].update({
        r".*linear_attn\.in_proj_qkv$": _layer_cfg("int8", "symmetric"),
        r".*linear_attn\.in_proj_z$": _layer_cfg("int8", "symmetric"),
        r".*linear_attn\.in_proj_a$": _layer_cfg("int8", "symmetric"),
        r".*linear_attn\.in_proj_b$": _layer_cfg("int8", "symmetric"),
        r".*lm_head$": None,
    })
    return cfg


QUANT_CONFIGS = {
    "fp16": None,
    "int8lin": config_int8lin,
    "int4lin": config_int4lin,
    "int4mix": config_int4mix,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="fp16",
                    choices=["fp16", "int8lin", "int4lin", "int4mix"])
    ap.add_argument("--hf-path", default="/Users/zgd/Code/llm/model/Qwen/Qwen3.5-4B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=2048)
    args = ap.parse_args()

    model_name = f"qwen3_5_4b_forward_{args.mode}"
    hf_path = args.hf_path
    quant_cfg = QUANT_CONFIGS[args.mode]

    print(f"Mode: {args.mode}")

    # ---- Load ----
    print(f"Loading HF model from {hf_path}...")
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_path, trust_remote_code=True, dtype=torch.float16, device_map="cpu"
    )
    hf_model.eval()

    config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
    tc = config.text_config if hasattr(config, "text_config") else config
    print(f"  Layers: {tc.num_hidden_layers}, Hidden: {tc.hidden_size}, Vocab: {tc.vocab_size}")

    # ---- Quantize (before tracing) ----
    if quant_cfg is not None:
        print(f"Applying {args.mode} quantization...")
        from coreai_models.export.compression import quantize_pytorch_model

        S = args.seq_len
        # Reference inputs for the inner language model
        ref_input_ids = torch.randint(0, tc.vocab_size, (1, S), dtype=torch.long)
        ref_inputs = (ref_input_ids,)
        dyn_shapes = None  # static shapes

        hf_model.model = quantize_pytorch_model(
            hf_model.model,
            ref_inputs,
            dyn_shapes,
            quant_cfg(),
        )
        print("  Quantization done")

    # ---- Wrap ----
    wrapper = SimpleWrapper(hf_model)
    wrapper.eval()

    # ---- Trace ----
    S = args.seq_len
    input_ids = torch.randint(0, tc.vocab_size, (1, S), dtype=torch.long)

    print(f"Tracing (seq_len={S})...")
    with torch.no_grad():
        ep = torch.export.export(wrapper, args=(input_ids,))
    print(f"  {len(ep.graph.nodes)} nodes")

    # ---- Decompose ----
    print("Decompositions...")
    from coreai_torch import TorchConverter, get_decomp_table

    ep_decomp = ep.run_decompositions(get_decomp_table())
    print(f"  {len(ep_decomp.graph.nodes)} nodes")

    # ---- Convert ----
    print("Converting to Core AI IR...")
    converter = TorchConverter()
    converter.add_exported_program(
        exported_program=ep_decomp,
        input_names=["input_ids"],
        output_names=["logits"],
    )
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

    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": model_name,
        "assets": {"main": f"{model_name}.aimodel"},
        "language": {
            "tokenizer": hf_path,
            "vocab_size": tc.vocab_size,
            "max_context_length": args.max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_path},
        "compression": {"mode": args.mode} if args.mode != "fp16" else None,
        "compilation": {
            "date": datetime.now(timezone.utc).isoformat(),
            "targets": [],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True).save_pretrained(
        out_dir / "tokenizer"
    )

    size_gb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(out_dir)
        for fn in fns
    ) / 1e9
    print(f"\nDone! Bundle: {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
