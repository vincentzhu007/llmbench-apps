# Copyright 2026.
# Qwen3.5 (hybrid SSM + full-attention) model overlay for coreai_models.
#
# Uses coreai_models primitives + coreai_torch GatedDeltaUpdate composite.
# Stateful forward with 4 states: k_cache, v_cache, conv_state, rec_state.

import torch
import torch.nn as nn
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling, resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm, RMSNormGated
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_torch.composite_ops import GatedDeltaUpdate

USE_FUSED_KV = True
DTYPE = torch.float16

# ---------------------------------------------------------------------------
# Full Attention layer (every 4th layer)
# ---------------------------------------------------------------------------

class FullAttention(nn.Module):
    """Gated full attention with GQA, QK-norm, RoPE, and KV cache."""

    def __init__(self, config, layer_idx: int, cache_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_idx = cache_idx  # KV cache slot index (0, 1, 2, ...)
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", dim // self.n_heads)

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        if USE_FUSED_KV:
            self.qk_norm = RMSNorm(
                self.head_dim, eps=config.rms_norm_eps,
                n_heads=self.n_heads + self.n_kv_heads,
            )
        else:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=True)
        assert is_default_rope_scaling(config), f"unsupported rope_scaling: {config.rope_scaling}"
        self.rope = initialize_rope(base=resolve_rope_theta(config))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        q = self.q_proj(x).reshape(batch_size, query_len, n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        gate = self.gate_proj(x).reshape(batch_size, query_len, n_heads, self.head_dim).permute(0, 2, 1, 3)

        if USE_FUSED_KV:
            query_key = torch.cat([q, k], dim=1)
            query_key = self.qk_norm(query_key)
            q = query_key.narrow(1, 0, n_heads)
            k = query_key.narrow(1, n_heads, n_kv_heads)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)

        seq_len = position_ids.shape[-1]
        offset = seq_len - query_len
        rope_positions = position_ids.narrow(-1, offset, query_len)

        q = self.rope(q, position_ids=rope_positions)
        k = self.rope(k, position_ids=rope_positions)

        if cache is not None:
            k, v = cache.update_and_fetch(
                self.cache_idx, offset, k, v, seq_len=seq_len, query_len=query_len,
            )

        out = (
            self.sdpa(q, k, v)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        gate = gate.permute(0, 2, 1, 3).reshape(batch_size, query_len, self.n_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# GatedDeltaNet (SSM / linear attention) layer
# ---------------------------------------------------------------------------

class GatedDeltaNetLayer(nn.Module):
    """Single SSM layer: causal conv + GatedDeltaUpdate composite.

    The GatedDeltaUpdate is Apple's Core-AI primitive for the gated-delta
    recurrence. Handles both prefill (scan) and decode (single step).
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_delta = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        # b uses num_v_heads (matches delta/dt), c uses num_k_heads
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Causal conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Apple's GatedDeltaUpdate composite
        self.gated_delta = GatedDeltaUpdate()

        self.use_loopfree_step = False

    def forward(
        self,
        h: torch.Tensor,
        position_ids: torch.IntTensor | None = None,
        conv_state: torch.Tensor | None = None,
        rec_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = h.shape

        qkv = self.in_proj_qkv(h)
        z = self.in_proj_z(h)
        delta = self.in_proj_delta(h)
        b = self.in_proj_b(h)

        # Causal conv
        qkv_t = qkv.transpose(1, 2)  # [B, C, S]

        # conv_state is per-layer [1, 1, C, K-1] or [1, C, K-1]; squeeze to [C, K-1]
        if conv_state is not None:
            cs = conv_state.squeeze(0).squeeze(0)  # → [C, K-1]
            cs = cs.unsqueeze(0)  # → [1, C, K-1] (add batch dim back)

        if seq_len == 1 and conv_state is not None:
            qkv_t = torch.cat([cs.expand(qkv_t.shape[0], -1, -1), qkv_t], dim=-1)
            new_conv_state = qkv_t[:, :, -(self.conv_kernel_size - 1):]
        else:
            qkv_t = nn.functional.pad(qkv_t, (self.conv_kernel_size - 1, 0))
            new_conv_state = qkv_t[:, :, -(self.conv_kernel_size - 1):]

        qkv_conv = nn.functional.silu(self.conv1d(qkv_t))

        if seq_len == 1:
            qkv_conv = qkv_conv[:, :, -1:]
        else:
            qkv_conv = qkv_conv[:, :, :seq_len]

        qkv_conv = qkv_conv.transpose(1, 2)

        q = qkv_conv[:, :, :self.key_dim]
        k = qkv_conv[:, :, self.key_dim:self.key_dim * 2]
        v = qkv_conv[:, :, self.key_dim * 2:]

        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # HF model repeats q/k to match value head count
        if self.num_v_heads // self.num_k_heads > 1:
            rpt = self.num_v_heads // self.num_k_heads
            q = q.repeat_interleave(rpt, dim=2)  # [B,S,Nk,Hk] → [B,S,Nv,Hk]
            k = k.repeat_interleave(rpt, dim=2)

        # GatedDeltaUpdate: (query, key, value, g, beta, initial_state) -> (out, new_state)
        # g = -exp(A_log) * softplus(delta + dt_bias) — matches HF Qwen3.5 formula
        if rec_state is not None:
            rs = rec_state.squeeze(0)
        else:
            rs = None

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(delta.float() + self.dt_bias)
        g = g.to(q.dtype)

        out, new_rec_state = self.gated_delta(
            query=q, key=k, value=v, g=g, beta=beta, initial_state=rs,
        )

        # Output is [B, S, Nv, Hv] — matches value shape after q/k repeat
        out = out.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(batch_size, seq_len, self.value_dim)

        # Reshape states back to per-layer format [1, 1, ...]
        if new_conv_state.dim() == 3:
            new_conv_state = new_conv_state.unsqueeze(0)  # → [1, 1, C, K-1]
        if new_rec_state is not None and new_rec_state.dim() == 4:
            new_rec_state = new_rec_state.unsqueeze(0)  # → [1, 1, H, K, V]

        return self.out_proj(out), new_conv_state, new_rec_state


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int, cache_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_full = self.layer_type == "full_attention"

        if self.is_full:
            self.attn = FullAttention(config, layer_idx, cache_idx)
        else:
            self.linear_attn = GatedDeltaNetLayer(config, layer_idx)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: KVCache | None = None,
        conv_state: torch.Tensor | None = None,
        rec_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        r = self.input_layernorm(x)

        new_conv = conv_state
        new_rec = rec_state

        if self.is_full:
            r = self.attn(r, position_ids, k_cache)
        else:
            r, new_conv, new_rec = self.linear_attn(r, position_ids, conv_state, rec_state)

        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, new_conv, new_rec


# ---------------------------------------------------------------------------
# Qwen3.5 Model
# ---------------------------------------------------------------------------

class Qwen35Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Count full-attention layers to assign correct KV cache slots
        cache_idx = 0
        blocks = []
        for i in range(config.num_hidden_layers):
            lt = config.layer_types[i]
            if lt == "full_attention":
                blocks.append(TransformerBlock(config, i, cache_idx))
                cache_idx += 1
            else:
                blocks.append(TransformerBlock(config, i))
        self.layers = nn.ModuleList(blocks)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.embed_tokens(input_ids)
        cache = KVCache(k_cache, v_cache)

        new_conv = conv_state
        new_rec = rec_state

        for i, layer in enumerate(self.layers):
            if layer.is_full:
                h, _, _ = layer(h, position_ids, cache)
            else:
                layer_conv = conv_state[i:i + 1]
                layer_rec = rec_state[i:i + 1]
                h, lc, lr = layer(h, position_ids, cache, layer_conv, layer_rec)
                new_conv = torch.cat([new_conv[:i], lc, new_conv[i + 1:]])
                new_rec = torch.cat([new_rec[:i], lr, new_rec[i + 1:]])

        h = self.norm(h)
        return h, k_cache, v_cache, new_conv, new_rec


# ---------------------------------------------------------------------------
# Stateful Causal LM wrapper
# ---------------------------------------------------------------------------

DECODE_STATE_NAMES = ("k_cache", "v_cache", "conv_state", "rec_state")


class Qwen35StatefulForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = None  # Custom loading path

    @override
    def _init_model(self, config) -> None:
        self.model = Qwen35Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k, v, conv, rec = self.model(
            input_ids, position_ids, k_cache, v_cache, conv_state, rec_state,
        )
        return self.lm_head(h), k, v, conv, rec

    @classmethod
    def from_hf(cls, hf_id: str, max_context_length: int = 4096,
                target_dtype=DTYPE, hf_config_attr: str = "text_config",
                num_layers: int | None = None) -> Self:
        """Load model weights from HuggingFace checkpoint."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        tc = getattr(config, hf_config_attr, config) if hf_config_attr else config
        if num_layers:
            tc.num_hidden_layers = num_layers
            tc.layer_types = tc.layer_types[:num_layers]

        # Ensure config has required attributes
        for attr in ["rms_norm_eps", "num_attention_heads", "num_key_value_heads", "head_dim"]:
            if not hasattr(tc, attr):
                setattr(tc, attr, getattr(tc, attr.replace("rms_norm_eps", "layer_norm_epsilon"),
                                         tc.hidden_size // tc.num_attention_heads if "head_dim" in attr else None))

        model = cls(tc)
        # Cast newly-created layers (key→value proj) to target dtype
        if target_dtype in (torch.float16, torch.bfloat16):
            model = model.to(target_dtype)
        model.eval()

        # Load HF weights
        from transformers import AutoModelForCausalLM
        import os

        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_id, trust_remote_code=True, dtype=target_dtype, device_map="cpu",
            local_files_only=os.path.isdir(hf_id),
        )
        state = hf_model.state_dict()

        # Map weights
        mapped = cls._map_weights(state, tc)
        model.load_state_dict(mapped, strict=False, assign=True)
        del hf_model
        return model

    @staticmethod
    def _map_weights(hf_state: dict, config) -> dict:
        """Map HF Qwen3.5 weight names to re-authored model names."""
        import re
        mapped = {}
        n_layers = config.num_hidden_layers

        # Embedding
        for key, target in [
            ("model.embed_tokens.weight", "model.embed_tokens.weight"),
            ("language_model.embed_tokens.weight", "model.embed_tokens.weight"),
        ]:
            if key in hf_state:
                mapped[target] = hf_state[key]
                break

        # Final norm
        for key in ["model.norm.weight", "language_model.norm.weight"]:
            if key in hf_state:
                mapped["model.norm.weight"] = hf_state[key]
                break

        # LM head
        for key in ["lm_head.weight", "language_model.lm_head.weight"]:
            if key in hf_state:
                mapped["lm_head.weight"] = hf_state[key]
                break
        else:
            # Tied: use embedding weight
            mapped["lm_head.weight"] = mapped["model.embed_tokens.weight"]

        for i in range(n_layers):
            # Determine HF prefix (model.layers or language_model.layers)
            hf_pfx0 = f"model.layers.{i}."
            hf_pfx1 = f"language_model.layers.{i}."
            hf_pfx = hf_pfx0 if any(k.startswith(hf_pfx0) for k in hf_state) else hf_pfx1
            npfx = f"model.layers.{i}."
            lt = config.layer_types[i]

            # Layer norms
            for norm_name in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                mapped[npfx + norm_name] = hf_state[hf_pfx + norm_name]

            if lt == "full_attention":
                # k_proj, v_proj, o_proj map directly
                for proj in ["k_proj", "v_proj", "o_proj"]:
                    mapped[npfx + f"attn.{proj}.weight"] = hf_state[hf_pfx + f"self_attn.{proj}.weight"]

                # HF q_proj outputs query+gate combined (2*head_dim per head); split
                qp = hf_state[hf_pfx + "self_attn.q_proj.weight"]
                half = qp.shape[0] // 2
                mapped[npfx + "attn.q_proj.weight"] = qp[:half]
                mapped[npfx + "attn.gate_proj.weight"] = qp[half:]

                # QK norm (fuse q_norm + k_norm into qk_norm)
                if USE_FUSED_KV:
                    qn = hf_state[hf_pfx + "self_attn.q_norm.weight"]
                    kn = hf_state[hf_pfx + "self_attn.k_norm.weight"]
                    n_h = config.num_attention_heads
                    n_kv = config.num_key_value_heads
                    hd = getattr(config, "head_dim", config.hidden_size // n_h)
                    qr = qn.unsqueeze(0).unsqueeze(0).expand(n_h, 1, hd)
                    kr = kn.unsqueeze(0).unsqueeze(0).expand(n_kv, 1, hd)
                    mapped[npfx + "attn.qk_norm.weight"] = torch.cat([qr, kr], dim=0)
            else:
                # SSM layer weight mapping
                la = hf_pfx + "linear_attn."
                nla = npfx + "linear_attn."

                ssm_map = {
                    "in_proj_qkv.weight": "in_proj_qkv.weight",
                    "in_proj_z.weight": "in_proj_z.weight",
                    "in_proj_b.weight": "in_proj_b.weight",
                    "conv1d.weight": "conv1d.weight",
                    "out_proj.weight": "out_proj.weight",
                    "norm.weight": "norm.weight",
                    "dt_bias": "dt_bias",
                    "A_log": "A_log",
                }
                for hf_k, our_k in ssm_map.items():
                    if la + hf_k in hf_state:
                        mapped[nla + our_k] = hf_state[la + hf_k]

                # Delta/dt projection (called "in_proj_a" in HF)
                for dt_k in ["in_proj_a.weight", "in_proj_delta.weight", "dt_proj.weight"]:
                    if la + dt_k in hf_state:
                        mapped[nla + "in_proj_delta.weight"] = hf_state[la + dt_k]
                        break

            # MLP
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                mapped[npfx + f"mlp.{proj}.weight"] = hf_state[hf_pfx + f"mlp.{proj}.weight"]

        return mapped

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        """No-op — weights are already mapped in from_hf."""
        pass

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Decode state builder (for the conversion pipeline)
# ---------------------------------------------------------------------------

def build_decode_state(config, max_seq_len: int, dtype=DTYPE) -> dict:
    """Build initial decode states (zero-filled)."""
    n_full = sum(1 for lt in config.layer_types if lt == "full_attention")
    n_lin = sum(1 for lt in config.layer_types if lt == "linear_attention")
    n_kv = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    c_dim = config.linear_key_head_dim * config.linear_num_key_heads * 2 + config.linear_value_head_dim * config.linear_num_value_heads

    return {
        "k_cache": torch.zeros(n_full, 1, n_kv, max_seq_len, head_dim, dtype=dtype),
        "v_cache": torch.zeros(n_full, 1, n_kv, max_seq_len, head_dim, dtype=dtype),
        "conv_state": torch.zeros(n_lin, 1, c_dim, config.linear_conv_kernel_dim - 1, dtype=dtype),
        "rec_state": torch.zeros(n_lin, 1, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim, dtype=dtype),
    }
