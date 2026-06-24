#!/usr/bin/env python3
"""Export Qwen3-0.6B to Core AI .aimodel using Apple's built-in pipeline."""

import os
import shutil
from pathlib import Path

def main():
    hf_path = "/Users/zgd/Code/llm/model/Qwen/Qwen3-0.6B"
    out_dir = Path("exports/qwen3_0.6b")

    print(f"Exporting Qwen3-0.6B from {hf_path}")

    # Monkey-patch HF downloads to use local path (offline mode).
    # The export pipeline calls snapshot_download + hf_hub_download internally.
    import huggingface_hub
    _orig_snapshot = huggingface_hub.snapshot_download
    _orig_hf_hub = huggingface_hub.hf_hub_download

    def _patched_snapshot(repo_id, **kw):
        if os.path.isdir(repo_id):
            return repo_id
        return _orig_snapshot(repo_id, **kw)

    def _patched_hf_hub(repo_id, filename, **kw):
        if os.path.isdir(repo_id):
            p = os.path.join(repo_id, filename)
            if os.path.exists(p):
                return p
        # Also check if the repo_id is a HF name like "Qwen/Qwen3-0.6B"
        # and we have a local copy
        local_dir = os.path.join("/Users/zgd/Code/llm/model/Qwen", repo_id.split("/")[-1])
        if os.path.isdir(local_dir):
            p = os.path.join(local_dir, filename)
            if os.path.exists(p):
                return p
        return _orig_hf_hub(repo_id, filename, **kw)

    huggingface_hub.snapshot_download = _patched_snapshot
    huggingface_hub.hf_hub_download = _patched_hf_hub

    from coreai_models.export.pipeline import ExportConfig, export_model

    # Use HF model ID — patched downloads will resolve to local path
    config = ExportConfig(
        hf_model_id="Qwen/Qwen3-0.6B",
        variant="macOS",
        compute_precision="float16",
        compression="4bit",
        output_dir=str(out_dir),
    )

    result = export_model(config)
    print(f"\nExport done: {result}")

    # Copy tokenizer
    from transformers import AutoTokenizer
    tok_dir = out_dir / "tokenizer"
    if not tok_dir.exists():
        tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        tok.save_pretrained(str(tok_dir))

    size_gb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(out_dir)
        for fn in fns
    ) / 1e9
    print(f"Size: {size_gb:.1f} GB")
    print(f"Bundle ready: {out_dir}")

if __name__ == "__main__":
    main()
