#!/usr/bin/env python3
"""
Convert Qwen3.5-4B using the coreai_models overlay (qwen3_5.py).

Produces a dynamic-shape stateful .aimodel with 4 states
(KV cache + SSM conv/rec). Supports chunked prefill natively.

Usage:
  python convert_qwen35_4b_v2.py --mode fp16 --out-dir exports
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

DTYPE = torch.float16


def palettization_config(n_bits: int = 8, group: int = 32) -> dict:
    spec = {
        "n_bits": n_bits,
        "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": group},
        "enable_per_channel_scale": False,
    }
    return {
        "global_config": {"op_state_spec": {"weight": spec}},
        "module_name_configs": {r".*lm_head$": None, r".*conv1d$": None},
    }


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="fp16", choices=["fp16", "int8lin"])
    ap.add_argument("--hf-path", default="/Users/zgd/Code/llm/model/Qwen/Qwen3.5-4B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--num-layers", type=int, default=None)
    args = ap.parse_args()

    model_name = f"qwen3_5_4b_stateful_{args.mode}"
    hf_path = args.hf_path

    # ---- Load model via overlay ----
    print(f"Loading Qwen3.5-4B from {hf_path}...")
    from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    from coreai_models.models.macos.qwen3_5 import (
        Qwen35StatefulForCausalLM,
        build_decode_state,
        DECODE_STATE_NAMES,
    )

    # Monkey-patch HF downloads to use local path
    import os, huggingface_hub
    _orig_snap = huggingface_hub.snapshot_download
    _orig_hf = huggingface_hub.hf_hub_download

    def _patched_snap(repo_id, **kw):
        return repo_id if os.path.isdir(repo_id) else _orig_snap(repo_id, **kw)
    def _patched_hf(repo_id, filename, **kw):
        if os.path.isdir(repo_id):
            p = os.path.join(repo_id, filename)
            if os.path.exists(p): return p
        return _orig_hf(repo_id, filename, **kw)

    huggingface_hub.snapshot_download = _patched_snap
    huggingface_hub.hf_hub_download = _patched_hf

    model = Qwen35StatefulForCausalLM.from_hf(
        hf_path, max_context_length=args.max_ctx, target_dtype=DTYPE,
        num_layers=args.num_layers,
    )
    model.eval()
    cfg = model.config
    print(f"  Layers: {cfg.num_hidden_layers}, Hidden: {cfg.hidden_size}, Vocab: {cfg.vocab_size}")

    # ---- Quantize ----
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        # Reference inputs for quantization
        trace_past = 64
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
        state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN)
        ref = {
            "input_ids": input_ids, "position_ids": position_ids,
            "k_cache": state["k_cache"], "v_cache": state["v_cache"],
            "conv_state": state["conv_state"], "rec_state": state["rec_state"],
        }
        # Dynamic shapes for quantization
        seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
        kv_seq = torch.export.Dim("kv_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
        dyn = {
            "input_ids": None, "position_ids": {1: seq_pos},
            "k_cache": {3: kv_seq}, "v_cache": {3: kv_seq},
            "conv_state": None, "rec_state": None,
        }
        print("Applying int8 linear quantization...")
        model = quantize_pytorch_model(
            model, tuple(ref.values()), dyn, linear_quant_config("int8"),
        )
        print("  Quantization done")

    # ---- Export ----
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN)

    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"], "rec_state": state["rec_state"],
    }

    # Dynamic shapes: position seq + KV cache seq are dynamic
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    kv_seq = torch.export.Dim("kv_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1] for decode, but let the engine prefill chunk
        "position_ids": {1: seq_pos},
        "k_cache": {3: kv_seq},
        "v_cache": {3: kv_seq},
        "conv_state": None,
        "rec_state": None,
    }

    # Externalize specs (use all except gated_delta — it's called directly)
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]

    print("Exporting to Core AI IR...")
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits", "k_out", "v_out", "conv_out", "rec_out"),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
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
            "tokenizer": hf_path,
            "vocab_size": cfg.vocab_size,
            "max_context_length": args.max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_path},
        "compression": None if args.mode == "fp16" else {"mode": args.mode},
        "compilation": {
            "date": datetime.now(timezone.utc).isoformat(),
            "targets": [],
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Copy tokenizer
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(
        hf_path, trust_remote_code=True, local_files_only=os.path.isdir(hf_path),
    ).save_pretrained(out_dir / "tokenizer")

    size_gb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(out_dir) for fn in fns
    ) / 1e9
    print(f"\nDone! Bundle: {out_dir} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
