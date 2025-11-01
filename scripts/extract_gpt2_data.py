"""Utility for tracing GPT-2 forward passes for the llm-visualized project.

This script provides an interactive prompt for sampling completions from the
124M parameter GPT-2 model and then replays the chosen completion while
recording a compact summary of the activations required by the Three.js
visualisation.

The collector focuses on the following signals:

* Residual stream vector states (sampled every ``stride`` dimensions)
* LayerNorm intermediary vectors (normalised, scaled and shifted forms)
* Multi-head attention query/key/value samples for every head and token
* Pre- and post-softmax attention probabilities (causal mask respected)
* MLP activations (up projection, activation, down projection)
* Residual additions between the attention/MLP blocks
* Final layer norm intermediary vectors

Alongside activations we track:

* Parameter counts activated up to each visualisation checkpoint
* Floating point operation (FLOP) estimates for sequential generation
* Sampling metadata and the top-k / top-p logits that influenced sampling

The resulting payload is intentionally compact and web-friendly.  Vector
states can be optionally quantised (float16 or int8 with a per-vector scale)
and only regularly spaced samples are stored.

Usage example::

    python scripts/extract_gpt2_data.py \
        --prompt "Far out in the uncharted backwaters" \
        --num-completions 4 --max-new-tokens 32 \
        --quantization float16 --output out.json

The script will display the sampled continuations, allow you to select or
edit a completion, and finally emit a structured JSON file together with a
companion format description in ``docs/GPT2TraceFormat.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import torch
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ---------------------------------------------------------------------------
# Quantisation helpers


QuantizationMode = Literal["none", "float16", "int8"]


@dataclass
class QuantizedVector:
    """Container for a quantised vector sample."""

    data: List[float | int]
    scale: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        if self.scale is None:
            return {"data": self.data}
        return {"data": self.data, "scale": self.scale}


class VectorQuantizer:
    """Applies optional quantisation to vector samples."""

    def __init__(self, mode: QuantizationMode) -> None:
        self.mode = mode

    def quantize(self, values: Iterable[float]) -> QuantizedVector:
        if self.mode == "none":
            return QuantizedVector(list(float(v) for v in values))
        tensor = torch.tensor(list(values), dtype=torch.float32)
        if self.mode == "float16":
            return QuantizedVector(tensor.to(torch.float16).tolist())
        if self.mode == "int8":
            max_abs = torch.max(torch.abs(tensor)).item()
            scale = max(max_abs / 127.0, 1e-8)
            quant = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
            return QuantizedVector([int(v) for v in quant.tolist()], scale=scale)
        raise ValueError(f"Unsupported quantisation mode: {self.mode}")


# ---------------------------------------------------------------------------
# Sampling utilities


def sample_vector(tensor: Tensor, stride: int) -> List[float]:
    """Sample evenly spaced values from a 1D tensor using ``stride`` steps."""

    if tensor.dim() != 1:
        raise ValueError("sample_vector expects a 1D tensor")
    stride = max(1, stride)
    indices = torch.arange(0, tensor.shape[0], stride, device=tensor.device)
    return tensor.index_select(0, indices).tolist()


def sample_lower_triangular(matrix: Tensor) -> List[List[float]]:
    """Return the lower triangular (causal) rows of an attention matrix."""

    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square attention matrix")
    out: List[List[float]] = []
    for t in range(matrix.shape[0]):
        row = matrix[t, : t + 1].tolist()
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# GPT-2 tracing logic


@dataclass
class LayerTrace:
    residual_in: List[QuantizedVector]
    ln1_norm: List[QuantizedVector]
    ln1_scale: List[QuantizedVector]
    ln1_shift: List[QuantizedVector]
    q_vectors: List[List[QuantizedVector]]
    k_vectors: List[List[QuantizedVector]]
    v_vectors: List[List[QuantizedVector]]
    attention_logits: List[List[List[float]]]
    attention_probs: List[List[List[float]]]
    attn_proj: List[QuantizedVector]
    post_attn_residual: List[QuantizedVector]
    ln2_norm: List[QuantizedVector]
    ln2_scale: List[QuantizedVector]
    ln2_shift: List[QuantizedVector]
    mlp_up: List[QuantizedVector]
    mlp_act: List[QuantizedVector]
    mlp_down: List[QuantizedVector]
    post_mlp_residual: List[QuantizedVector]


@dataclass
class FinalLayerNormTrace:
    norm: List[QuantizedVector]
    scale: List[QuantizedVector]
    shift: List[QuantizedVector]


def layernorm_intermediates(x: Tensor, layernorm: torch.nn.LayerNorm) -> Tuple[Tensor, Tensor, Tensor]:
    """Return vectors after norm, scale (gamma) and shift (beta)."""

    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    norm = (x - mean) / torch.sqrt(variance + layernorm.eps)
    scale = norm * layernorm.weight
    shift = scale + layernorm.bias
    return norm, scale, shift


def trace_gpt2_forward(
    model: GPT2LMHeadModel,
    input_ids: Tensor,
    vector_stride: int,
    quantizer: VectorQuantizer,
) -> Tuple[List[LayerTrace], FinalLayerNormTrace, Dict[str, List[QuantizedVector]], Tensor]:
    """Run a manual forward pass and capture intermediate activations."""

    transformer = model.transformer
    device = model.device
    batch_size, seq_len = input_ids.shape
    if batch_size != 1:
        raise ValueError("trace_gpt2_forward currently supports batch size 1")

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        token_embeddings = transformer.wte(input_ids)
        position_embeddings = transformer.wpe(position_ids)
        hidden_states = token_embeddings + position_embeddings

        embedding_trace = {
            "token": [
                quantizer.quantize(sample_vector(token_embeddings[0, t], vector_stride))
                for t in range(seq_len)
            ],
            "position": [
                quantizer.quantize(sample_vector(position_embeddings[0, t], vector_stride))
                for t in range(seq_len)
            ],
            "sum": [
                quantizer.quantize(sample_vector(hidden_states[0, t], vector_stride))
                for t in range(seq_len)
            ],
        }

        layer_traces: List[LayerTrace] = []

        residual = hidden_states
        for layer_index, block in enumerate(transformer.h):
            layer_data: Dict[str, List[QuantizedVector]] = {}

            # Residual input (shared with previous layer output)
            residual_samples = [
                quantizer.quantize(sample_vector(residual[0, t], vector_stride))
                for t in range(seq_len)
            ]

            norm1, scale1, shift1 = layernorm_intermediates(residual, block.ln_1)
            ln1_norm_samples = [
                quantizer.quantize(sample_vector(norm1[0, t], vector_stride))
                for t in range(seq_len)
            ]
            ln1_scale_samples = [
                quantizer.quantize(sample_vector(scale1[0, t], vector_stride))
                for t in range(seq_len)
            ]
            ln1_shift_samples = [
                quantizer.quantize(sample_vector(shift1[0, t], vector_stride))
                for t in range(seq_len)
            ]

            # Attention projections
            attn = block.attn
            qkv = torch.addmm(attn.c_attn.bias, shift1.view(seq_len, -1), attn.c_attn.weight.t())
            qkv = qkv.view(seq_len, 3, attn.n_head, attn.n_embd // attn.n_head)
            q = qkv[:, 0]
            k = qkv[:, 1]
            v = qkv[:, 2]

            q_samples: List[List[QuantizedVector]] = []
            k_samples: List[List[QuantizedVector]] = []
            v_samples: List[List[QuantizedVector]] = []
            for head in range(attn.n_head):
                q_head = q[:, head, :]
                k_head = k[:, head, :]
                v_head = v[:, head, :]
                q_samples.append(
                    [quantizer.quantize(sample_vector(q_head[t], vector_stride)) for t in range(seq_len)]
                )
                k_samples.append(
                    [quantizer.quantize(sample_vector(k_head[t], vector_stride)) for t in range(seq_len)]
                )
                v_samples.append(
                    [quantizer.quantize(sample_vector(v_head[t], vector_stride)) for t in range(seq_len)]
                )

            # Attention scores
            scale = 1.0 / math.sqrt(attn.n_embd // attn.n_head)
            q_ = q.permute(1, 0, 2)  # head, seq, dim
            k_ = k.permute(1, 0, 2)
            v_ = v.permute(1, 0, 2)
            attn_logits = torch.matmul(q_, k_.transpose(-1, -2)) * scale
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
            attn_probs = torch.softmax(attn_logits, dim=-1)

            logits_samples = [
                sample_lower_triangular(attn_logits[head]) for head in range(attn.n_head)
            ]
            prob_samples = [
                sample_lower_triangular(attn_probs[head]) for head in range(attn.n_head)
            ]

            # Attention output
            context = torch.matmul(attn_probs, v_)
            context = context.permute(1, 0, 2).reshape(seq_len, -1)
            attn_output = torch.addmm(attn.c_proj.bias, context, attn.c_proj.weight.t())

            attn_output_samples = [
                quantizer.quantize(sample_vector(attn_output[t], vector_stride))
                for t in range(seq_len)
            ]

            residual_after_attn = residual + attn_output.view(1, seq_len, -1)
            residual_after_attn_samples = [
                quantizer.quantize(sample_vector(residual_after_attn[0, t], vector_stride))
                for t in range(seq_len)
            ]

            # LayerNorm 2
            norm2, scale2, shift2 = layernorm_intermediates(residual_after_attn, block.ln_2)
            ln2_norm_samples = [
                quantizer.quantize(sample_vector(norm2[0, t], vector_stride))
                for t in range(seq_len)
            ]
            ln2_scale_samples = [
                quantizer.quantize(sample_vector(scale2[0, t], vector_stride))
                for t in range(seq_len)
            ]
            ln2_shift_samples = [
                quantizer.quantize(sample_vector(shift2[0, t], vector_stride))
                for t in range(seq_len)
            ]

            mlp = block.mlp
            up = torch.addmm(mlp.c_fc.bias, shift2.view(seq_len, -1), mlp.c_fc.weight.t())
            up_reshaped = up.view(seq_len, -1)
            act = mlp.act(up_reshaped)
            down = torch.addmm(mlp.c_proj.bias, act, mlp.c_proj.weight.t())

            mlp_up_samples = [
                quantizer.quantize(sample_vector(up_reshaped[t], vector_stride))
                for t in range(seq_len)
            ]
            mlp_act_samples = [
                quantizer.quantize(sample_vector(act[t], vector_stride))
                for t in range(seq_len)
            ]
            mlp_down_samples = [
                quantizer.quantize(sample_vector(down[t], vector_stride))
                for t in range(seq_len)
            ]

            residual = residual_after_attn + down.view(1, seq_len, -1)
            residual_after_mlp_samples = [
                quantizer.quantize(sample_vector(residual[0, t], vector_stride))
                for t in range(seq_len)
            ]

            layer_traces.append(
                LayerTrace(
                    residual_in=residual_samples,
                    ln1_norm=ln1_norm_samples,
                    ln1_scale=ln1_scale_samples,
                    ln1_shift=ln1_shift_samples,
                    q_vectors=q_samples,
                    k_vectors=k_samples,
                    v_vectors=v_samples,
                    attention_logits=logits_samples,
                    attention_probs=prob_samples,
                    attn_proj=attn_output_samples,
                    post_attn_residual=residual_after_attn_samples,
                    ln2_norm=ln2_norm_samples,
                    ln2_scale=ln2_scale_samples,
                    ln2_shift=ln2_shift_samples,
                    mlp_up=mlp_up_samples,
                    mlp_act=mlp_act_samples,
                    mlp_down=mlp_down_samples,
                    post_mlp_residual=residual_after_mlp_samples,
                )
            )

        norm_f, scale_f, shift_f = layernorm_intermediates(residual, transformer.ln_f)
        final_trace = FinalLayerNormTrace(
            norm=[
                quantizer.quantize(sample_vector(norm_f[0, t], vector_stride))
                for t in range(seq_len)
            ],
            scale=[
                quantizer.quantize(sample_vector(scale_f[0, t], vector_stride))
                for t in range(seq_len)
            ],
            shift=[
                quantizer.quantize(sample_vector(shift_f[0, t], vector_stride))
                for t in range(seq_len)
            ],
        )

        logits = model.lm_head(shift_f)

    return layer_traces, final_trace, embedding_trace, logits


# ---------------------------------------------------------------------------
# Parameter and flop accounting


def count_parameters(model: GPT2LMHeadModel) -> Dict[str, int]:
    """Return cumulative parameter usage for all checkpoints."""

    transformer = model.transformer
    counts: Dict[str, int] = {}
    total = 0

    def add(name: str, *parameters: Tensor) -> None:
        nonlocal total
        subtotal = sum(p.numel() for p in parameters)
        total += subtotal
        counts[name] = total

    add("embeddings/token", transformer.wte.weight)
    add("embeddings/position", transformer.wpe.weight)

    for idx, block in enumerate(transformer.h):
        prefix = f"layer_{idx:02d}"
        add(f"{prefix}/ln1", block.ln_1.weight, block.ln_1.bias)
        add(f"{prefix}/attn/qkv", block.attn.c_attn.weight, block.attn.c_attn.bias)
        add(f"{prefix}/attn/proj", block.attn.c_proj.weight, block.attn.c_proj.bias)
        add(f"{prefix}/ln2", block.ln_2.weight, block.ln_2.bias)
        add(f"{prefix}/mlp/up", block.mlp.c_fc.weight, block.mlp.c_fc.bias)
        add(f"{prefix}/mlp/down", block.mlp.c_proj.weight, block.mlp.c_proj.bias)

    add("final_ln", transformer.ln_f.weight, transformer.ln_f.bias)
    add("lm_head", model.lm_head.weight)
    return counts


def estimate_generation_flops(
    seq_lengths: List[int],
    n_layer: int,
    n_head: int,
    n_embd: int,
) -> Dict[str, List[int]]:
    """Estimate FLOPs consumed at each checkpoint for autoregressive tokens."""

    head_dim = n_embd // n_head
    checkpoints: Dict[str, List[int]] = {}

    def ensure_list(key: str) -> List[int]:
        return checkpoints.setdefault(key, [])

    for t, context_length in enumerate(seq_lengths, start=1):
        # Each entry corresponds to a new generated token.
        # Embedding addition (token + position)
        ensure_list("embeddings/add").append(n_embd)

        for layer_idx in range(n_layer):
            prefix = f"layer_{layer_idx:02d}"
            # LayerNorm (mean + variance + normalise)
            ln_norm_cost = 5 * n_embd
            ensure_list(f"{prefix}/ln1/norm").append(ln_norm_cost)
            ensure_list(f"{prefix}/ln1/scale").append(n_embd)
            ensure_list(f"{prefix}/ln1/shift").append(n_embd)

            # QKV projections: three matrix-vector multiplies
            proj_cost = 6 * n_embd * n_embd
            ensure_list(f"{prefix}/attn/qkv").append(proj_cost)

            # Attention logits and softmax (for single new token)
            attn_scores = 2 * n_head * head_dim * context_length
            softmax_cost = 5 * context_length * n_head
            ensure_list(f"{prefix}/attn/scores").append(attn_scores + softmax_cost)

            # Context vector (probabilities @ values)
            context_cost = 2 * n_head * head_dim * context_length
            ensure_list(f"{prefix}/attn/context").append(context_cost)

            # Output projection
            ensure_list(f"{prefix}/attn/proj").append(2 * n_embd * n_embd)

            # Residual add
            ensure_list(f"{prefix}/attn/residual").append(n_embd)

            # LayerNorm 2
            ensure_list(f"{prefix}/ln2/norm").append(ln_norm_cost)
            ensure_list(f"{prefix}/ln2/scale").append(n_embd)
            ensure_list(f"{prefix}/ln2/shift").append(n_embd)

            # MLP up projection
            ensure_list(f"{prefix}/mlp/up").append(2 * n_embd * (4 * n_embd))
            # Activation
            ensure_list(f"{prefix}/mlp/act").append(8 * 4 * n_embd)
            # Down projection
            ensure_list(f"{prefix}/mlp/down").append(2 * (4 * n_embd) * n_embd)
            # Residual add
            ensure_list(f"{prefix}/mlp/residual").append(n_embd)

        # Final layer norm
        ensure_list("final_ln/norm").append(5 * n_embd)
        ensure_list("final_ln/scale").append(n_embd)
        ensure_list("final_ln/shift").append(n_embd)

    return checkpoints


# ---------------------------------------------------------------------------
# Sampling utilities and CLI helpers


def pretty_completion(text: str) -> str:
    return text.replace("\n", "\\n")


def generate_completions(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    prompt_ids: Tensor,
    max_new_tokens: int,
    num_completions: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    seed: Optional[int],
) -> List[Tensor]:
    completions: List[Tensor] = []
    generator = torch.Generator(device=model.device)
    if seed is not None:
        generator.manual_seed(seed)

    for _ in range(num_completions):
        generated = prompt_ids.clone().to(model.device)
        past_key_values = None
        for _ in range(max_new_tokens):
            inputs = generated[:, -1:] if past_key_values is not None else generated
            outputs = model(
                input_ids=inputs,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values

        completions.append(generated.cpu())

    return completions


def choose_completion(
    completions: List[Tensor],
    tokenizer: GPT2TokenizerFast,
    prompt_tokens: Tensor,
) -> Tensor:
    print("\nAvailable completions:")
    for idx, tokens in enumerate(completions, start=1):
        tokens_cpu = tokens[0].detach().cpu()
        completion_text = tokenizer.decode(
            tokens_cpu[prompt_tokens.shape[-1] :], skip_special_tokens=True
        )
        print(f"  [{idx}] {pretty_completion(completion_text)}")

    while True:
        choice = input("Select completion number, 'm' to enter manually, or 'q' to quit: ").strip().lower()
        if choice == "q":
            sys.exit(0)
        if choice == "m":
            manual = input("Enter custom completion text: ")
            tokens = tokenizer.encode(manual, add_special_tokens=False)
            continuation = torch.tensor([tokens], dtype=torch.long)
            return torch.cat([prompt_tokens.cpu(), continuation], dim=-1)
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(completions):
                selected = completions[idx - 1]
                selected_cpu = selected.detach().cpu()
                default_tokens = selected_cpu[:, prompt_tokens.shape[-1] :]
                trunc = input(
                    f"Keep how many tokens from completion (<= {default_tokens.shape[-1]}), Enter for all: "
                ).strip()
                if trunc:
                    keep = max(0, min(int(trunc), default_tokens.shape[-1]))
                    return selected_cpu[:, : prompt_tokens.shape[-1] + keep]
                return selected_cpu
        print("Invalid selection, please try again.")


# ---------------------------------------------------------------------------
# Assembly of the final payload


def build_payload(
    tokenizer: GPT2TokenizerFast,
    model: GPT2LMHeadModel,
    prompt: str,
    prompt_ids: Tensor,
    completion_ids: Tensor,
    embeddings: Dict[str, List[QuantizedVector]],
    layer_traces: List[LayerTrace],
    final_trace: FinalLayerNormTrace,
    logits: Tensor,
    param_counts: Dict[str, int],
    flop_counts: Dict[str, List[int]],
    vector_stride: int,
    quant_mode: QuantizationMode,
    top_token_config: Dict[str, object],
    prompt_token_count: int,
) -> Dict[str, object]:
    seq_len = completion_ids.shape[-1]
    tokens = [
        {
            "token": tokenizer.decode([int(tok)], clean_up_tokenization_spaces=False),
            "token_id": int(tok),
        }
        for tok in completion_ids[0]
    ]

    def qv_list_to_json(vecs: List[QuantizedVector]) -> List[Dict[str, object]]:
        return [v.to_json() for v in vecs]

    layer_payload = []
    for layer_idx, layer in enumerate(layer_traces):
        layer_entry = {
            "index": layer_idx,
            "residual_in": qv_list_to_json(layer.residual_in),
            "ln1": {
                "norm": qv_list_to_json(layer.ln1_norm),
                "scale": qv_list_to_json(layer.ln1_scale),
                "shift": qv_list_to_json(layer.ln1_shift),
            },
            "attention": {
                "q": [[v.to_json() for v in head] for head in layer.q_vectors],
                "k": [[v.to_json() for v in head] for head in layer.k_vectors],
                "v": [[v.to_json() for v in head] for head in layer.v_vectors],
                "pre_softmax": layer.attention_logits,
                "post_softmax": layer.attention_probs,
                "projection": qv_list_to_json(layer.attn_proj),
                "residual": qv_list_to_json(layer.post_attn_residual),
            },
            "ln2": {
                "norm": qv_list_to_json(layer.ln2_norm),
                "scale": qv_list_to_json(layer.ln2_scale),
                "shift": qv_list_to_json(layer.ln2_shift),
            },
            "mlp": {
                "up": qv_list_to_json(layer.mlp_up),
                "act": qv_list_to_json(layer.mlp_act),
                "down": qv_list_to_json(layer.mlp_down),
                "residual": qv_list_to_json(layer.post_mlp_residual),
            },
        }
        layer_payload.append(layer_entry)

    final_ln_payload = {
        "norm": qv_list_to_json(final_trace.norm),
        "scale": qv_list_to_json(final_trace.scale),
        "shift": qv_list_to_json(final_trace.shift),
    }

    prompt_length = prompt_token_count
    generation_top = []
    for position in range(prompt_length, seq_len):
        logits_step = logits[0, position - 1]
        config_type = top_token_config.get("mode")
        if config_type == "top_k":
            k = int(top_token_config["k"])
            values, indices = torch.topk(logits_step, k)
        elif config_type == "top_p":
            sorted_logits, sorted_indices = torch.sort(logits_step, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            mask = cumulative <= float(top_token_config["p"])
            mask[..., 0] = True
            selected_logits = sorted_logits[mask]
            selected_indices = sorted_indices[mask]
            values, indices = selected_logits, selected_indices
        else:
            values, indices = torch.topk(logits_step, 10)
        generation_top.append(
            {
                "position": position,
                "candidates": [
                    {
                        "token_id": int(idx),
                        "token": tokenizer.decode([int(idx)], clean_up_tokenization_spaces=False),
                        "logit": float(val),
                    }
                    for idx, val in zip(indices, values)
                ],
            }
        )

    payload = {
        "meta": {
            "model": "gpt2",
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "vector_stride": vector_stride,
            "quantization": quant_mode,
        },
        "prompt": {
            "text": prompt,
            "token_ids": prompt_ids[0, :prompt_length].tolist(),
        },
        "tokens": tokens,
        "embeddings": {
            key: [vec.to_json() for vec in values]
            for key, values in embeddings.items()
        },
        "layers": layer_payload,
        "final_layernorm": final_ln_payload,
        "logits": generation_top,
        "parameters_used": param_counts,
        "flops_used": flop_counts,
    }

    return payload


# ---------------------------------------------------------------------------
# Main CLI entry point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace GPT-2 activations for llm-visualized")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to prime GPT-2")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generation length")
    parser.add_argument("--num-completions", type=int, default=4, help="Number of candidates to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (None disables)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling (0-1 range)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--vector-stride", type=int, default=32, help="Stride for vector sampling")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "float16", "int8"],
        default="float16",
        help="Quantisation mode for stored activations",
    )
    parser.add_argument("--output", type=str, default="gpt2_trace.json", help="Output JSON file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    prompt = args.prompt or input("Enter a prompt: ")
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

    completions = generate_completions(
        model,
        tokenizer,
        prompt_ids.to(model.device),
        max_new_tokens=args.max_new_tokens,
        num_completions=args.num_completions,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    selected = choose_completion(completions, tokenizer, prompt_ids)
    selected = selected.to(dtype=torch.long)
    if selected.shape[-1] <= prompt_ids.shape[-1]:
        print("Warning: no new tokens selected; tracing prompt only.")

    full_tokens = selected.to(model.device)

    quantizer = VectorQuantizer(args.quantization)  # type: ignore[arg-type]
    layer_traces, final_ln, embedding_trace, logits = trace_gpt2_forward(
        model, full_tokens, args.vector_stride, quantizer
    )

    param_counts = count_parameters(model)

    prompt_length = prompt_ids.shape[-1]
    generation_lengths = [prompt_length + i for i in range(1, full_tokens.shape[-1] - prompt_length + 1)]
    flop_counts = estimate_generation_flops(
        generation_lengths,
        n_layer=model.config.n_layer,
        n_head=model.config.n_head,
        n_embd=model.config.n_embd,
    )

    top_token_config: Dict[str, object] = {}
    if args.top_k:
        top_token_config = {"mode": "top_k", "k": args.top_k}
    elif args.top_p:
        top_token_config = {"mode": "top_p", "p": args.top_p}

    payload = build_payload(
        tokenizer,
        model,
        prompt,
        prompt_ids,
        full_tokens.cpu(),
        embedding_trace,
        layer_traces,
        final_ln,
        logits.cpu(),
        param_counts,
        flop_counts,
        args.vector_stride,
        args.quantization,
        top_token_config,
        prompt_ids.shape[-1],
    )

    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Trace saved to {args.output}")


if __name__ == "__main__":
    main()

