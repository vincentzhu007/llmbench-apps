#!/usr/bin/env python3
"""Wrapper that exports a LOCAL HuggingFace model directory to Core AI via the
coreai_models pipeline. Patches huggingface_hub so a local dir is accepted
as the model path (instead of being treated as a repo id) and applies a
torch._check compat shim for torch 2.9 vs 2.11. Then delegates to
coreai_models.llm.export:main, so all its CLI flags work unchanged.

Usage:
  python export_local_coreai.py /path/to/local/model --platform macOS \
      --compute-precision float16 --experimental --output-dir ... --output-name ...
"""
import os
import sys

import huggingface_hub as hh
import torch as _torch

# 1) Accept a local directory as the model path.
_snap = hh.snapshot_download
_hf = hh.hf_hub_download


def _p_snap(repo_id, **kw):
    return repo_id if os.path.isdir(repo_id) else _snap(repo_id, **kw)


def _p_hf(repo_id, filename, **kw):
    if os.path.isdir(repo_id):
        p = os.path.join(repo_id, filename)
        if os.path.exists(p):
            return p
    return _hf(repo_id, filename, **kw)


hh.snapshot_download = _p_snap
hh.hf_hub_download = _p_hf

# 2) torch._check signature compat (torch 2.9 vs 2.11).
_orig_check = _torch._check


def _patched_check(cond, message=None):
    msg = message if callable(message) else (lambda m=message: str(m) if m else "")
    try:
        return _orig_check(cond, msg)
    except TypeError:
        return _orig_check(cond, message=message)


_torch._check = _patched_check

# 3) Delegate to coreai's export CLI (must be imported AFTER the patches above).
from coreai_models.llm.export import main  # noqa: E402

main()
