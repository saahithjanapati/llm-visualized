"""Utility to sample GPT-2 activations for the llm-visualized project.

This script runs an instrumented forward pass through GPT-2 (124M) and
records compact activation "vector states" alongside lightweight
metadata (flop/parameter counts, sampling logits, etc.).

Usage (interactive example):

    python scripts/extract_gpt2_data.py \
        --max-new-tokens 32 --num-completions 5 --output out.json

The script will prompt for a seed prompt, generate completions, allow the
user to pick or edit a completion, and then emit a JSON payload that can
be consumed by the three.js visualiser.

Notes
-----
* Vector states are sampled every 32 dims by default (configurable).
* Values are quantised to float16 by default; int8 quantisation is
  available via ``--quantisation int8``.
* FLOP counts are approximate and only cover generation tokens (prompt
  tokens are ignored as described in the project brief).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "transformers is required. Install with `pip install transformers`."
    ) from exc


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------


class BaseQuantiser:
    """Interface for emitting compact numeric payloads."""

    name: str = "base"

    def __init__(self, round_decimals: Optional[int] = None) -> None:
        self.round_decimals = round_decimals

    def _round_values(self, values: List[float]) -> List[float]:
        if self.round_decimals is None:
            return values
        return [round(float(val), self.round_decimals) for val in values]

    def encode(self, tensor: torch.Tensor) -> Dict[str, object]:  # pragma: no cover - interface
        raise NotImplementedError


class Float16Quantiser(BaseQuantiser):
    name = "float16"

    def encode(self, tensor: torch.Tensor) -> Dict[str, object]:
        return {
            "v": self._round_values(tensor.detach().cpu().to(torch.float16).tolist()),
        }


class Float32Quantiser(BaseQuantiser):
    name = "float32"

    def encode(self, tensor: torch.Tensor) -> Dict[str, object]:
        return {"v": self._round_values(tensor.detach().cpu().to(torch.float32).tolist())}


class Int8SymmetricQuantiser(BaseQuantiser):
    """Simple symmetric int8 quantiser with a single scale per vector."""

    name = "int8_sym"

    def encode(self, tensor: torch.Tensor) -> Dict[str, object]:
        cpu = tensor.detach().cpu().to(torch.float32)
        max_abs = float(cpu.abs().max())
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        if self.round_decimals is not None:
            scale = round(scale, self.round_decimals)
        quantised = torch.clamp(torch.round(cpu / scale), -127, 127).to(torch.int8)
        return {"s": scale, "v": quantised.tolist()}


def build_quantiser(name: str, round_decimals: Optional[int]) -> BaseQuantiser:
    if name == "float16":
        return Float16Quantiser(round_decimals)
    if name == "float32":
        return Float32Quantiser(round_decimals)
    if name in {"int8", "int8_sym"}:
        return Int8SymmetricQuantiser(round_decimals)
    raise ValueError(f"Unknown quantisation mode: {name}")


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def sample_tensor(tensor: torch.Tensor, stride: int) -> torch.Tensor:
    """Sample `tensor` along the last dimension every `stride` steps."""

    if stride <= 0:
        raise ValueError("stride must be > 0")
    dim = tensor.size(-1)
    indices = torch.arange(0, dim, stride, device=tensor.device)
    return tensor.index_select(-1, indices)


def encode_vector_states(
    tensor: torch.Tensor,
    stride: int,
    quantiser: BaseQuantiser,
) -> List[Dict[str, object]]:
    """Return quantised samples for each token in a sequence tensor."""

    if tensor.dim() != 3:
        raise ValueError("Expected tensor with shape (batch, seq, dim)")
    batch, seq_len, _ = tensor.shape
    if batch != 1:
        raise ValueError("Only batch size 1 is supported for capture")
    samples: List[Dict[str, object]] = []
    for t in range(seq_len):
        vec = tensor[0, t]
        sampled = sample_tensor(vec, stride)
        samples.append(quantiser.encode(sampled))
    return samples


def encode_head_vector_states(
    tensor: torch.Tensor,
    stride: int,
    quantiser: BaseQuantiser,
) -> List[List[Dict[str, object]]]:
    """Return quantised samples per head/per token for attention tensors."""

    if tensor.dim() != 4:
        raise ValueError("Expected tensor with shape (batch, heads, seq, dim)")
    batch, num_heads, seq_len, _ = tensor.shape
    if batch != 1:
        raise ValueError("Only batch size 1 is supported for capture")
    head_samples: List[List[Dict[str, object]]] = []
    for h in range(num_heads):
        token_samples: List[Dict[str, object]] = []
        for t in range(seq_len):
            vec = tensor[0, h, t]
            sampled = sample_tensor(vec, stride)
            token_samples.append(quantiser.encode(sampled))
        head_samples.append(token_samples)
    return head_samples


def encode_triangular(
    tensor: torch.Tensor,
    quantiser: BaseQuantiser,
) -> List[List[Dict[str, object]]]:
    """Encode lower-triangular attention matrices per head/per token."""

    if tensor.dim() != 4:
        raise ValueError("Expected tensor with shape (batch, heads, query, key)")
    batch, num_heads, seq_len, _ = tensor.shape
    if batch != 1:
        raise ValueError("Only batch size 1 is supported for capture")
    results: List[List[Dict[str, object]]] = []
    for h in range(num_heads):
        head_entries: List[Dict[str, object]] = []
        for q in range(seq_len):
            allowed = tensor[0, h, q, : q + 1]
            head_entries.append(quantiser.encode(allowed))
        results.append(head_entries)
    return results


# ---------------------------------------------------------------------------
# FLOP accounting (approximate)
# ---------------------------------------------------------------------------


@dataclass
class StageFlops:
    stage: str
    layer: Optional[int]
    flops: float
    cumulative: float


def approximate_layer_flops(
    hidden_size: int,
    intermediate_size: int,
    num_heads: int,
    num_layers: int,
    prompt_len: int,
    generation_len: int,
) -> List[StageFlops]:
    """Return approximate FLOP counts for the requested checkpoints."""

    head_dim = hidden_size // num_heads
    entries: List[StageFlops] = []
    cumulative = 0.0

    def emit(stage: str, layer: Optional[int], flops: float) -> None:
        nonlocal cumulative
        cumulative += flops
        entries.append(StageFlops(stage=stage, layer=layer, flops=flops, cumulative=cumulative))

    def per_token_embed_flops() -> float:
        # Embedding lookup is mostly memory, but treat as hidden_size ops.
        return float(hidden_size)

    def per_token_residual_add() -> float:
        return float(hidden_size)

    def per_token_layernorm_norm() -> float:
        # Mean + variance + normalisation (approximate).
        return float(5 * hidden_size)

    def per_token_layernorm_scale_shift() -> float:
        return float(hidden_size)

    def per_token_gelu() -> float:
        # Rough estimate for GELU (tanh-based approx).
        return float(6 * intermediate_size)

    for _ in range(generation_len):
        emit("token_embedding", None, per_token_embed_flops())
        emit("position_embedding", None, per_token_embed_flops())
        emit("embedding_sum", None, per_token_residual_add())

    for layer_idx in range(num_layers):
        for step in range(generation_len):
            context_len = prompt_len + step + 1
            emit("ln1_norm", layer_idx, per_token_layernorm_norm())
            emit("ln1_scale", layer_idx, per_token_layernorm_scale_shift())
            emit("ln1_shift", layer_idx, per_token_layernorm_scale_shift())

            qkv_flops = 6.0 * hidden_size * hidden_size
            emit("qkv_projection", layer_idx, qkv_flops)

            attn_matmul = 2.0 * num_heads * head_dim * context_len
            softmax_flops = float(num_heads * 5 * context_len)
            emit("attention_scores", layer_idx, attn_matmul + softmax_flops)
            emit("attention_weighted_values", layer_idx, attn_matmul)
            emit("concat_heads", layer_idx, 0.0)

            proj_flops = 2.0 * hidden_size * hidden_size
            emit("attn_output_projection", layer_idx, proj_flops)
            emit("attn_residual_add", layer_idx, per_token_residual_add())

            emit("ln2_norm", layer_idx, per_token_layernorm_norm())
            emit("ln2_scale", layer_idx, per_token_layernorm_scale_shift())
            emit("ln2_shift", layer_idx, per_token_layernorm_scale_shift())

            up_flops = 2.0 * hidden_size * intermediate_size
            down_flops = 2.0 * hidden_size * intermediate_size
            emit("mlp_up_projection", layer_idx, up_flops)
            emit("mlp_activation", layer_idx, per_token_gelu())
            emit("mlp_down_projection", layer_idx, down_flops)
            emit("mlp_residual_add", layer_idx, per_token_residual_add())

    for _ in range(generation_len):
        emit("final_ln_norm", None, per_token_layernorm_norm())
        emit("final_ln_scale", None, per_token_layernorm_scale_shift())
        emit("final_ln_shift", None, per_token_layernorm_scale_shift())

    return entries


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


@dataclass
class StageParameters:
    stage: str
    layer: Optional[int]
    delta: int
    cumulative: int


def parameter_accounting(model: GPT2LMHeadModel) -> List[StageParameters]:
    entries: List[StageParameters] = []
    cumulative = 0

    def emit(stage: str, layer: Optional[int], delta: int) -> None:
        nonlocal cumulative
        cumulative += delta
        entries.append(StageParameters(stage=stage, layer=layer, delta=delta, cumulative=cumulative))

    wte_params = model.transformer.wte.weight.numel()
    wpe_params = model.transformer.wpe.weight.numel()
    emit("token_embedding", None, wte_params)
    emit("position_embedding", None, wpe_params)
    emit("embedding_sum", None, 0)

    for layer_idx, block in enumerate(model.transformer.h):
        hidden_size = block.ln_1.weight.numel()
        emit("ln1_norm", layer_idx, 0)
        emit("ln1_scale", layer_idx, hidden_size)
        emit("ln1_shift", layer_idx, hidden_size)

        c_attn_weight = block.attn.c_attn.weight.numel()
        c_attn_bias = block.attn.c_attn.bias.numel()
        emit("qkv_projection", layer_idx, c_attn_weight + c_attn_bias)
        emit("attention_scores", layer_idx, 0)
        emit("attention_weighted_values", layer_idx, 0)
        emit("concat_heads", layer_idx, 0)

        c_proj_weight = block.attn.c_proj.weight.numel()
        c_proj_bias = block.attn.c_proj.bias.numel()
        emit("attn_output_projection", layer_idx, c_proj_weight + c_proj_bias)
        emit("attn_residual_add", layer_idx, 0)

        emit("ln2_norm", layer_idx, 0)
        emit("ln2_scale", layer_idx, hidden_size)
        emit("ln2_shift", layer_idx, hidden_size)

        mlp_up_weight = block.mlp.c_fc.weight.numel()
        mlp_up_bias = block.mlp.c_fc.bias.numel()
        emit("mlp_up_projection", layer_idx, mlp_up_weight + mlp_up_bias)
        emit("mlp_activation", layer_idx, 0)

        mlp_down_weight = block.mlp.c_proj.weight.numel()
        mlp_down_bias = block.mlp.c_proj.bias.numel()
        emit("mlp_down_projection", layer_idx, mlp_down_weight + mlp_down_bias)
        emit("mlp_residual_add", layer_idx, 0)

    final_ln_size = model.transformer.ln_f.weight.numel()
    emit("final_ln_norm", None, 0)
    emit("final_ln_scale", None, final_ln_size)
    emit("final_ln_shift", None, final_ln_size)

    return entries


# ---------------------------------------------------------------------------
# Sampling logits helpers
# ---------------------------------------------------------------------------


def extract_sampling_logits(
    logits: torch.Tensor,
    top_k: Optional[int],
    top_p: Optional[float],
    round_decimals: Optional[int],
) -> List[List[Dict[str, object]]]:
    """Return token/logit metadata matching the sampler configuration."""

    if logits.dim() != 2:
        raise ValueError("Expected logits with shape (seq, vocab)")

    seq_len, vocab_size = logits.shape
    probs = torch.softmax(logits, dim=-1)
    entries: List[List[Dict[str, object]]] = []

    for t in range(seq_len):
        token_entries: List[Dict[str, object]] = []
        logit_row = logits[t]
        prob_row = probs[t]

        candidate_indices: Optional[torch.Tensor] = None
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(prob_row, min(top_k, vocab_size))
            candidate_indices = indices
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(prob_row, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            mask = cumulative <= top_p
            # ensure at least one token is kept
            if not mask.any():
                mask[0] = True
            indices = sorted_indices[mask]
            candidate_indices = indices if candidate_indices is None else torch.unique(torch.cat([candidate_indices, indices]))

        if candidate_indices is None:
            # default to argmax if no sampling constraints specified
            candidate_indices = prob_row.argmax(dim=-1, keepdim=True)

        for idx in candidate_indices.tolist():
            logit = float(logit_row[idx])
            if round_decimals is not None:
                logit = round(logit, round_decimals)
            token_entries.append({"token_id": int(idx), "logit": logit})
        entries.append(token_entries)

    return entries


# ---------------------------------------------------------------------------
# Completion verification helpers
# ---------------------------------------------------------------------------


def token_allowed_by_sampler(
    logits: torch.Tensor,
    token_id: int,
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: float,
) -> Tuple[bool, Optional[str]]:
    """Return whether token_id survives top-k/top-p sampling filters."""

    scores = logits.detach().to(torch.float32)
    if temperature and temperature != 1.0:
        temperature = max(1e-6, float(temperature))
        scores = scores / temperature

    vocab_size = scores.numel()
    topk_indices = None
    if top_k is not None and top_k > 0:
        top_k = min(top_k, vocab_size)
        _, topk_indices = torch.topk(scores, top_k)
        if not bool((topk_indices == token_id).any()):
            return False, f"outside top-k ({top_k})"
        mask = torch.full_like(scores, float("-inf"))
        mask[topk_indices] = scores[topk_indices]
        scores = mask

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        sorted_mask = cumulative <= top_p
        if not bool(sorted_mask.any()):
            sorted_mask[0] = True
        allowed = sorted_indices[sorted_mask]
        if not bool((allowed == token_id).any()):
            return False, f"outside top-p ({top_p})"

    return True, None


def check_completion_feasibility(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_ids: List[int],
    completion_ids: List[int],
    top_k: Optional[int],
    top_p: Optional[float],
    temperature: float,
    device: str,
) -> Tuple[bool, str]:
    """Check if completion tokens are allowed by the sampling filters."""

    if not prompt_ids:
        return False, "Prompt is empty; cannot verify completion."
    if not completion_ids:
        return False, "Completion is empty."

    all_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(all_ids).logits[0]

    prompt_len = len(prompt_ids)
    for idx, token_id in enumerate(completion_ids):
        logit_row = logits[prompt_len - 1 + idx]
        allowed, reason = token_allowed_by_sampler(logit_row, token_id, top_k, top_p, temperature)
        if not allowed:
            token_str = tokenizer.convert_ids_to_tokens([token_id])[0]
            return (
                False,
                f"Token {idx + 1} ('{token_str}', id {token_id}) is {reason}.",
            )

    return True, f"Completion is compatible with top-k/top-p at temperature {temperature}."


def print_next_token_topk(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_ids: torch.Tensor,
    top_k: int,
    temperature: float,
) -> None:
    """Print the top-k next-token candidates for the prompt."""

    if top_k <= 0:
        print("Top-k preview skipped (k <= 0).")
        return

    with torch.no_grad():
        logits = model(prompt_ids).logits[0, -1]

    scores = logits.detach().to(torch.float32)
    if temperature and temperature != 1.0:
        temperature = max(1e-6, float(temperature))
        scores = scores / temperature

    probs = torch.softmax(scores, dim=-1)
    k = min(int(top_k), probs.numel())
    values, indices = torch.topk(probs, k)

    print(f"\nTop-{k} next-token candidates (temperature={temperature}):\n")
    for rank, (prob, token_id) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        token_display = token.replace("\n", "\\n").replace("\t", "\\t")
        print(f"{rank:>2}. id={token_id:<5} p={prob:.4f} token={token_display}")
# ---------------------------------------------------------------------------
# GPT-2 forward instrumentation
# ---------------------------------------------------------------------------


@dataclass
class CaptureConfig:
    residual_stride: int = 32
    attention_stride: int = 32
    mlp_stride: int = 32
    quantisation: str = "float16"
    round_decimals: Optional[int] = None


def layer_norm_states(x: torch.Tensor, ln: torch.nn.LayerNorm) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    normed = (x - mean) / torch.sqrt(variance + ln.eps)
    scaled = normed * ln.weight
    shifted = scaled + ln.bias
    return normed, scaled, shifted


@dataclass
class CaptureResult:
    payload: Dict[str, object]
    logits: torch.Tensor


def run_instrumented_pass(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    config: CaptureConfig,
) -> CaptureResult:
    device = next(model.parameters()).device
    quantiser = build_quantiser(config.quantisation, config.round_decimals)
    attention_quantiser = quantiser
    mlp_quantiser = quantiser

    model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("Only batch size 1 is supported for capture")

        transformer = model.transformer
        token_embeddings = transformer.wte(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = transformer.wpe(position_ids)

        hidden_states = token_embeddings + position_embeddings
        hidden_states = transformer.drop(hidden_states)

        data: Dict[str, object] = {
            "embeddings": {
                "token": encode_vector_states(token_embeddings, config.residual_stride, quantiser),
                "position": encode_vector_states(position_embeddings, config.residual_stride, quantiser),
                "sum": encode_vector_states(hidden_states, config.residual_stride, quantiser),
            },
            "layers": [],
        }

        residual = hidden_states
        for layer_idx, block in enumerate(transformer.h):
            layer_entry: Dict[str, object] = {}

            layer_entry["incoming"] = encode_vector_states(residual, config.residual_stride, quantiser)

            ln1_norm, ln1_scaled, ln1_shifted = layer_norm_states(residual, block.ln_1)
            layer_entry["ln1"] = {
                "norm": encode_vector_states(ln1_norm, config.residual_stride, quantiser),
                "scale": encode_vector_states(ln1_scaled, config.residual_stride, quantiser),
                "shift": encode_vector_states(ln1_shifted, config.residual_stride, quantiser),
            }

            ln1_out = ln1_shifted

            qkv = block.attn.c_attn(ln1_out)
            q, k, v = qkv.split(transformer.wte.embedding_dim, dim=2)

            num_heads = block.attn.num_heads
            head_dim = block.attn.head_dim

            def reshape_heads(t: torch.Tensor) -> torch.Tensor:
                return t.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

            q_heads = reshape_heads(q)
            k_heads = reshape_heads(k)
            v_heads = reshape_heads(v)

            layer_entry["qkv"] = {
                "q": encode_head_vector_states(q_heads, config.attention_stride, attention_quantiser),
                "k": encode_head_vector_states(k_heads, config.attention_stride, attention_quantiser),
                "v": encode_head_vector_states(v_heads, config.attention_stride, attention_quantiser),
            }

            attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(head_dim)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            attn_weights = attn_weights.masked_fill(mask == 1, float("-inf"))
            attn_probs = F.softmax(attn_weights, dim=-1)

            layer_entry["attention_scores"] = {
                "pre": encode_triangular(attn_weights, quantiser),
                "post": encode_triangular(attn_probs, quantiser),
            }

            context = torch.matmul(attn_probs, v_heads)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(batch_size, seq_len, transformer.wte.embedding_dim)

            attn_output = block.attn.c_proj(context)
            attn_output = block.attn.resid_dropout(attn_output)
            layer_entry["attn_output_proj"] = encode_vector_states(attn_output, config.residual_stride, quantiser)

            residual = residual + attn_output
            layer_entry["post_attn_residual"] = encode_vector_states(residual, config.residual_stride, quantiser)

            ln2_norm, ln2_scaled, ln2_shifted = layer_norm_states(residual, block.ln_2)
            layer_entry["ln2"] = {
                "norm": encode_vector_states(ln2_norm, config.residual_stride, quantiser),
                "scale": encode_vector_states(ln2_scaled, config.residual_stride, quantiser),
                "shift": encode_vector_states(ln2_shifted, config.residual_stride, quantiser),
            }

            mlp_in = ln2_shifted
            mlp_up = block.mlp.c_fc(mlp_in)
            layer_entry["mlp_up"] = encode_vector_states(mlp_up, config.mlp_stride, mlp_quantiser)

            mlp_act = block.mlp.act(mlp_up)
            layer_entry["mlp_act"] = encode_vector_states(mlp_act, config.mlp_stride, mlp_quantiser)

            mlp_down = block.mlp.c_proj(mlp_act)
            mlp_down = block.mlp.dropout(mlp_down)
            layer_entry["mlp_down"] = encode_vector_states(mlp_down, config.residual_stride, quantiser)

            residual = residual + mlp_down
            layer_entry["post_mlp_residual"] = encode_vector_states(residual, config.residual_stride, quantiser)

            data["layers"].append(layer_entry)

        final_norm, final_scaled, final_shifted = layer_norm_states(residual, transformer.ln_f)
        data["final_layernorm"] = {
            "norm": encode_vector_states(final_norm, config.residual_stride, quantiser),
            "scale": encode_vector_states(final_scaled, config.residual_stride, quantiser),
            "shift": encode_vector_states(final_shifted, config.residual_stride, quantiser),
        }

        logits = model.lm_head(final_shifted)
        return CaptureResult(payload=data, logits=logits[0])


# ---------------------------------------------------------------------------
# Prompt/CLI helpers
# ---------------------------------------------------------------------------


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:  # pragma: no cover - interactive fallback
        return ""


def pick_completion(options: Sequence[str]) -> str:
    if not options:
        return ""
    print("\nGenerated completions:\n")
    for idx, text in enumerate(options, start=1):
        print(f"[{idx}] {text}\n")

    while True:
        choice = prompt_user("Select completion [number], 'c' to craft your own, 'v' to verify a completion, or 'r' to re-run sampling: ")
        choice = choice.strip().lower()
        if choice == "c":
            return prompt_user("Enter custom completion text: ")
        if choice == "v":
            return "__verify__"
        if choice == "r":
            return "__rerun__"
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid selection, please try again.\n")


def maybe_truncate_completion(tokenizer: GPT2TokenizerFast, prompt_text: str, completion: str) -> Tuple[str, List[int], List[int]]:
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    print(f"\nSelected completion has {len(completion_ids)} tokens.")
    trunc = prompt_user("Enter max completion tokens to keep (or press Enter for full completion): ")
    trunc = trunc.strip()
    if trunc:
        try:
            keep = max(0, min(len(completion_ids), int(trunc)))
        except ValueError:
            print("Invalid number, keeping full completion.")
            keep = len(completion_ids)
        completion_ids = completion_ids[:keep]
        completion = tokenizer.decode(completion_ids, clean_up_tokenization_spaces=False)
    return completion, prompt_ids, completion_ids


def parse_token_list(value: Optional[str], label: str) -> List[int]:
    if value is None:
        return []
    raw = value.strip()
    if not raw:
        return []
    try:
        if raw.startswith("["):
            items = json.loads(raw)
            if not isinstance(items, list):
                raise ValueError
            return [int(item) for item in items]
        parts = [part for part in re.split(r"[,\s]+", raw) if part]
        return [int(part) for part in parts]
    except Exception as exc:
        raise ValueError(f"Invalid {label}: expected JSON list or comma-separated ints.") from exc




# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=None, help="Seed prompt text. If omitted, the user is prompted interactively.")
    parser.add_argument("--prompt-tokens", type=str, default=None, help="Prompt token ids (JSON list or comma-separated).")
    parser.add_argument("--completion-tokens", type=str, default=None, help="Completion token ids (JSON list or comma-separated).")
    parser.add_argument("--num-completions", type=int, default=5, help="Number of candidate completions to sample.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum number of tokens to generate for each completion.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible sampling.")
    parser.add_argument("--output", type=str, default="gpt2_capture.json", help="Output JSON path.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu or cuda).")
    parser.add_argument("--residual-stride", type=int, default=64, help="Stride for 768-d residual vectors.")
    parser.add_argument("--attention-stride", type=int, default=64, help="Stride for 64-d head vectors.")
    parser.add_argument("--mlp-stride", type=int, default=64, help="Stride for 3072-d MLP activations.")
    parser.add_argument("--quantisation", type=str, default="float16", choices=["float16", "float32", "int8"], help="Quantisation mode for stored activations.")
    parser.add_argument("--round-decimals", type=int, default=2, help="Round stored floats to this many decimal places.")
    parser.add_argument("--inspect-next-top-k", type=int, default=None, help="Print top-k next-token candidates for the prompt.")

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(args.device)
    model.eval()

    prompt_text = args.prompt
    prompt_id_list: List[int] = []
    completion_id_list: List[int] = []
    completion_text = ""

    use_token_override = args.prompt_tokens is not None or args.completion_tokens is not None
    if use_token_override:
        if args.prompt_tokens is None or args.completion_tokens is None:
            print("Both --prompt-tokens and --completion-tokens are required.")
            return 1
        try:
            prompt_id_list = parse_token_list(args.prompt_tokens, "prompt tokens")
            completion_id_list = parse_token_list(args.completion_tokens, "completion tokens")
        except ValueError as exc:
            print(str(exc))
            return 1
        if not prompt_id_list:
            print("Prompt tokens are empty. Exiting.")
            return 1
        if prompt_text is None:
            prompt_text = tokenizer.decode(prompt_id_list, clean_up_tokenization_spaces=False)
        completion_text = tokenizer.decode(completion_id_list, clean_up_tokenization_spaces=False)
    else:
        prompt_text = prompt_text or prompt_user("Enter a seed prompt: ")
        if not prompt_text:
            print("No prompt provided. Exiting.")
            return 1

        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
        )
        if args.top_k is not None:
            generate_kwargs["top_k"] = args.top_k
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p

        completions: List[str] = []
        while True:
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt")
            prompt_ids = prompt_ids.to(args.device)

            if args.inspect_next_top_k is not None:
                print_next_token_topk(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    top_k=args.inspect_next_top_k,
                    temperature=args.temperature,
                )

            outputs = model.generate(
                prompt_ids,
                num_return_sequences=args.num_completions,
                pad_token_id=tokenizer.eos_token_id,
                **generate_kwargs,
            )

            completions = []
            for seq in outputs:
                completion_ids = seq[len(prompt_ids[0]) :]
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completions.append(text.strip())

            completion_text = ""
            prompt_id_list = []
            completion_id_list = []
            while True:
                selected = pick_completion(completions)
                if selected == "__rerun__":
                    break
                if selected == "__verify__":
                    candidate = prompt_user("Enter completion to verify: ").strip()
                    if not candidate:
                        print("No completion entered.")
                        continue
                    prompt_id_list = tokenizer.encode(prompt_text, add_special_tokens=False)
                    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
                    ok, message = check_completion_feasibility(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_ids=prompt_id_list,
                        completion_ids=candidate_ids,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        device=args.device,
                    )
                    print(message)
                    if ok:
                        use = prompt_user("Use this completion? [y/N]: ").strip().lower()
                        if use in {"y", "yes"}:
                            completion_text, prompt_id_list, completion_id_list = maybe_truncate_completion(
                                tokenizer, prompt_text, candidate
                            )
                            break
                    continue
                completion_text, prompt_id_list, completion_id_list = maybe_truncate_completion(
                    tokenizer, prompt_text, selected
                )
                break

            if selected == "__rerun__":
                continue
            if completion_text:
                break

    all_token_ids = torch.tensor([prompt_id_list + completion_id_list], dtype=torch.long)

    capture_config = CaptureConfig(
        residual_stride=args.residual_stride,
        attention_stride=args.attention_stride,
        mlp_stride=args.mlp_stride,
        quantisation=args.quantisation,
        round_decimals=args.round_decimals,
    )

    capture = run_instrumented_pass(model, all_token_ids.to(args.device), capture_config)

    top_logits = extract_sampling_logits(capture.logits, args.top_k, args.top_p, args.round_decimals)

    prompt_len = len(prompt_id_list)
    generation_len = len(completion_id_list)

    flops_entries = approximate_layer_flops(
        hidden_size=model.config.n_embd,
        intermediate_size=model.config.n_inner or model.config.n_embd * 4,
        num_heads=model.config.n_head,
        num_layers=model.config.n_layer,
        prompt_len=prompt_len,
        generation_len=generation_len,
    )

    params_entries = parameter_accounting(model)

    payload = {
        "meta": {
            "prompt": prompt_text,
            "completion": completion_text,
            "prompt_tokens": prompt_id_list,
            "completion_tokens": completion_id_list,
            "token_strings": tokenizer.convert_ids_to_tokens(prompt_id_list + completion_id_list),
            "config": dataclasses.asdict(capture_config),
        },
        "activations": capture.payload,
        "logits": top_logits,
        "flops": [dataclasses.asdict(entry) for entry in flops_entries],
        "parameters": [dataclasses.asdict(entry) for entry in params_entries],
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    print(f"\nSaved capture to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
