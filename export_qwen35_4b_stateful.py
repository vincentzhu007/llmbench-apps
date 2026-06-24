#!/usr/bin/env python3
"""Export Qwen3.5-4B with stateful overlay to .aimodel (dynamic shape, 4 states)."""

from __future__ import annotations
import argparse, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path
import torch, huggingface_hub as hh

# ---------------------------------------------------------------------------
# Fix torch._check compatibility (torch 2.9 vs 2.11)
# ---------------------------------------------------------------------------
import torch as _torch
_orig_check = _torch._check
def _patched_check(cond, message=None):
    # Accept both positional and keyword: torch._check(cond, msg) AND torch._check(cond, message=msg)
    msg = message if callable(message) else (lambda m=message: str(m) if m else '')
    try:
        return _orig_check(cond, msg)
    except TypeError:
        return _orig_check(cond, message=msg)
_torch._check = _patched_check

# ---------------------------------------------------------------------------
# Patch HF downloads for local paths
# ---------------------------------------------------------------------------
_os_snap = hh.snapshot_download
_os_hf = hh.hf_hub_download
def _p_snap(r, **kw): return r if os.path.isdir(r) else _os_snap(r, **kw)
def _p_hf(r, f, **kw):
    if os.path.isdir(r):
        p = os.path.join(r, f)
        if os.path.exists(p): return p
    return _os_hf(r, f, **kw)
hh.snapshot_download = _p_snap
hh.hf_hub_download = _p_hf

DTYPE = torch.float16

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="fp16", choices=["fp16", "int8lin"])
    ap.add_argument("--hf-path", default="/Users/zgd/Code/llm/model/Qwen/Qwen3.5-4B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--num-layers", type=int, default=None, help="Truncate to N layers (debug)")
    ap.add_argument("--full", action="store_true", help="Export all 32 layers")
    args = ap.parse_args()

    if args.full:
        args.num_layers = None

    model_name = f"qwen3_5_4b_stateful_{args.mode}"
    if args.num_layers:
        model_name += f"_l{args.num_layers}"

    hf_path = args.hf_path
    print(f"Exporting Qwen3.5-4B stateful from {hf_path}")
    print(f"  Mode: {args.mode}, Layers: {args.num_layers or 'all'}")

    # ---- Load via overlay ----
    from coreai_models.models.macos.qwen3_5 import (
        Qwen35StatefulForCausalLM, build_decode_state, DECODE_STATE_NAMES,
    )
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN

    print("Loading model...")
    model = Qwen35StatefulForCausalLM.from_hf(
        hf_path, max_context_length=args.max_ctx, target_dtype=DTYPE,
        num_layers=args.num_layers,
    )
    model.eval()
    cfg = model.config
    n_full = sum(1 for lt in cfg.layer_types if lt == "full_attention")
    n_ssm = sum(1 for lt in cfg.layer_types if lt == "linear_attention")
    print(f"  Layers: {cfg.num_hidden_layers} ({n_full} full + {n_ssm} SSM)")

    # ---- Quantize ----
    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        trace_past = 64
        ref_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        ref_pos = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
        st = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
        ref = {
            "input_ids": ref_ids, "position_ids": ref_pos,
            "k_cache": st["k_cache"], "v_cache": st["v_cache"],
            "conv_state": st["conv_state"], "rec_state": st["rec_state"],
        }
        dyn = {
            "input_ids": None, "position_ids": {1: torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)},
            "k_cache": {3: torch.export.Dim("kv_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)},
            "v_cache": {3: torch.export.Dim("kv_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)},
            "conv_state": None, "rec_state": None,
        }
        cfg_q = {
            "execution_mode": "eager",
            "global_config": {
                "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
                "op_input_spec": None, "op_output_spec": None,
            },
            "module_type_configs": {
                "coreai_models.primitives.macos.sdpa.SDPA": None,
                "coreai_models.primitives.macos.rope.RoPE": None,
                "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
                "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
                "torch.nn.modules.sparse.Embedding": None,
                "torch.nn.modules.conv.Conv1d": None,
            },
            "module_name_configs": {r".*lm_head$": None, r".*_key_to_value_proj$": None},
        }
        print("Quantizing int8...")
        model = quantize_pytorch_model(model, tuple(ref.values()), dyn, cfg_q)
        print("  Done")

    # ---- Export ----
    trace_past = 64
    ref_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    ref_pos = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    st = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    ref = {
        "input_ids": ref_ids, "position_ids": ref_pos,
        "k_cache": st["k_cache"], "v_cache": st["v_cache"],
        "conv_state": st["conv_state"], "rec_state": st["rec_state"],
    }

    seq_dim = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    kv_dim = torch.export.Dim("kv_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dyn_shapes = {
        "input_ids": None,
        "position_ids": {1: seq_dim},
        "k_cache": {3: kv_dim},
        "v_cache": {3: kv_dim},
        "conv_state": None,
        "rec_state": None,
    }

    print("Exporting to Core AI IR...")
    # conv_state/rec_state: regular I/O (not CoreAI states — no slice_update).
    # Only k_cache/v_cache are true states.
    prog = export_to_coreai(
        model, ref, dynamic_shapes=dyn_shapes,
        input_names=("input_ids", "position_ids", "conv_state", "rec_state"),
        output_names=("logits", "conv_out", "rec_out"),
        state_names=("k_cache", "v_cache"),
    )
    print("  Converter OK")
    print("Optimizing...")
    prog.optimize()
    print("  Optimize OK")

    # ---- Save ----
    out_dir = Path(args.out_dir) / model_name
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt
    aimodel = out_dir / f"{model_name}.aimodel"
    print(f"Saving {aimodel}...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    meta = {
        "metadata_version": "0.2", "kind": "llm", "name": model_name,
        "assets": {"main": f"{model_name}.aimodel"},
        "language": {"tokenizer": hf_path, "vocab_size": cfg.vocab_size,
                      "max_context_length": args.max_ctx, "embedded_tokenizer": True,
                      "function_map": {"main": ["main"]}},
        "source": {"model_definition": "torch", "hf_model_id": hf_path},
        "compression": None if args.mode == "fp16" else {"mode": args.mode},
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(
        hf_path, trust_remote_code=True, local_files_only=os.path.isdir(hf_path),
    ).save_pretrained(out_dir / "tokenizer")

    sz = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out_dir) for fn in fns) / 1e9
    print(f"\nDone! Bundle: {out_dir} ({sz:.1f} GB)")

if __name__ == "__main__":
    main()
