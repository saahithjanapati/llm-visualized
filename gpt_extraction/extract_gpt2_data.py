"""Utilities to capture GPT-2 (124M) activations for visualization.

This script provides an interactive CLI for sampling GPT-2 completions and
recording the internal vector states that power the llm-visualized project.

Run ``python -m gpt_extraction.extract_gpt2_data --help`` for usage details.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


# ---------------------------------------------------------------------------
# Quantisation helpers


@dataclass
class QuantizedArray:
    """Compact representation of a tensor for JSON serialisation."""

    dtype: str
    shape: Tuple[int, ...]
    data: str
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    layout: Optional[str] = None

    def to_json(self) -> Dict[str, object]:
        payload = {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data": self.data,
        }
        if self.scale is not None:
            payload["scale"] = self.scale
        if self.zero_point is not None:
            payload["zero_point"] = self.zero_point
        if self.layout is not None:
            payload["layout"] = self.layout
        return payload


class Quantizer:
    """Quantise numpy arrays into a compact serialisable representation."""

    def __init__(self, mode: str = "float16") -> None:
        allowed = {"float32", "float16", "int8"}
        if mode not in allowed:
            raise ValueError(f"Unsupported quantization mode {mode!r}. Choose from {allowed}.")
        self.mode = mode

    def _encode_bytes(self, arr: np.ndarray) -> str:
        return base64.b64encode(arr.tobytes()).decode("ascii")

    def encode(self, array: np.ndarray, *, layout: Optional[str] = None) -> QuantizedArray:
        array = np.asarray(array)
        if self.mode == "float32":
            payload = array.astype(np.float32)
            return QuantizedArray("float32", tuple(payload.shape), self._encode_bytes(payload), layout=layout)

        if self.mode == "float16":
            payload = array.astype(np.float16)
            return QuantizedArray("float16", tuple(payload.shape), self._encode_bytes(payload), layout=layout)

        # int8 quantisation (symmetric)
        max_abs = float(np.max(np.abs(array)))
        scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
        quantised = np.clip(np.round(array / scale), -128, 127).astype(np.int8)
        return QuantizedArray(
            "int8",
            tuple(quantised.shape),
            self._encode_bytes(quantised),
            scale=scale,
            zero_point=0,
            layout=layout,
        )


# ---------------------------------------------------------------------------
# Sampling helpers


class VectorSampler:
    """Sample evenly spaced elements from the last dimension."""

    def __init__(self, stride: int) -> None:
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        self.stride = stride

    def indices(self, dim: int) -> torch.Tensor:
        return torch.arange(0, dim, self.stride)

    def sample(self, tensor: torch.Tensor) -> torch.Tensor:
        idx = self.indices(tensor.size(-1)).to(tensor.device)
        return tensor.index_select(-1, idx)


# ---------------------------------------------------------------------------
# Cost tracking


@dataclass
class CostCheckpoint:
    name: str
    increment: float
    cumulative: float


class CostTracker:
    def __init__(self) -> None:
        self.total = 0.0
        self.checkpoints: List[CostCheckpoint] = []

    def add(self, name: str, increment: float) -> None:
        self.total += float(increment)
        self.checkpoints.append(CostCheckpoint(name, float(increment), self.total))


# ---------------------------------------------------------------------------
# Generation utilities


@dataclass
class GenerationStep:
    step: int
    input_token: int
    sampled_token: int
    sampled_logit: float
    sampled_prob: float
    top_k: Optional[Dict[str, object]]
    top_p: Optional[Dict[str, object]]

    def to_json(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "step": self.step,
            "input_token": self.input_token,
            "sampled_token": self.sampled_token,
            "sampled_logit": self.sampled_logit,
            "sampled_prob": self.sampled_prob,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        return payload


def _extract_top_p_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    if mask.dim() == 0:
        mask = mask.unsqueeze(0)
    mask[..., 0] = True  # Ensure at least one token survives
    selected_indices = sorted_indices[mask]
    selected_probs = sorted_probs[mask]
    return torch.stack((selected_indices, selected_probs), dim=-1)


def package_token_stats(
    token_ids: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
    quantizer: Quantizer,
) -> Dict[str, object]:
    return {
        "token_ids": token_ids.cpu().tolist(),
        "logits": quantizer.encode(logits.cpu().numpy()).to_json(),
        "probs": quantizer.encode(probs.cpu().numpy()).to_json(),
    }


def generate_candidates(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    input_ids: torch.Tensor,
    *,
    num_completions: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    quantizer: Quantizer,
) -> List[Dict[str, object]]:
    """Generate candidate completions with logits traces."""

    device = model.device
    processors = LogitsProcessorList()
    if temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature=temperature))
    if top_k > 0:
        processors.append(TopKLogitsWarper(top_k=top_k))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p=top_p))

    candidates: List[Dict[str, object]] = []
    with torch.no_grad():
        for _ in range(num_completions):
            generated = input_ids.clone()
            steps: List[GenerationStep] = []
            for step in range(max_new_tokens):
                outputs = model(generated)
                next_token_logits = outputs.logits[:, -1, :]
                processed = processors(generated, next_token_logits)
                probs = torch.softmax(processed, dim=-1)

                if top_k > 0:
                    tk_probs, tk_indices = torch.topk(probs, k=top_k, dim=-1)
                    tk_logits = torch.gather(processed, -1, tk_indices)
                    tk_quant = package_token_stats(
                        tk_indices.squeeze(0),
                        tk_logits.squeeze(0),
                        tk_probs.squeeze(0),
                        quantizer,
                    )
                else:
                    tk_quant = None

                if top_p < 1.0:
                    tp_pairs = _extract_top_p_probs(probs.squeeze(0), top_p)
                    indices = tp_pairs[:, 0].long()
                    tp_logits = processed.squeeze(0)[indices]
                    tp_quant = package_token_stats(
                        indices,
                        tp_logits,
                        tp_pairs[:, 1],
                        quantizer,
                    )
                else:
                    tp_quant = None

                next_token = torch.multinomial(probs, num_samples=1)
                sampled_token = int(next_token.item())
                sampled_logit = float(processed[0, sampled_token].item())
                sampled_prob = float(probs[0, sampled_token].item())

                steps.append(
                    GenerationStep(
                        step=step,
                        input_token=int(generated[0, -1].item()),
                        sampled_token=sampled_token,
                        sampled_logit=sampled_logit,
                        sampled_prob=sampled_prob,
                        top_k=tk_quant,
                        top_p=tp_quant,
                    )
                )

                generated = torch.cat([generated, next_token.to(device)], dim=-1)

            completion_ids = generated[0, input_ids.size(1) :]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            candidates.append(
                {
                    "tokens": completion_ids.tolist(),
                    "text": completion_text,
                    "trace": [s.to_json() for s in steps],
                }
            )

    return candidates


# ---------------------------------------------------------------------------
# LayerNorm helpers


@dataclass
class LayerNormBreakdown:
    normalised: torch.Tensor
    scaled: torch.Tensor
    shifted: torch.Tensor


def layer_norm_breakdown(x: torch.Tensor, ln: torch.nn.LayerNorm) -> LayerNormBreakdown:
    eps = ln.eps
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    normalised = (x - mean) / torch.sqrt(variance + eps)
    scaled = normalised * ln.weight
    shifted = scaled + ln.bias
    return LayerNormBreakdown(normalised, scaled, shifted)


# ---------------------------------------------------------------------------
# Data capture


def sample_per_token(tensor: torch.Tensor, sampler: VectorSampler, quantizer: Quantizer) -> List[Dict[str, object]]:
    sampled = sampler.sample(tensor).squeeze(0).cpu().numpy()
    return [quantizer.encode(vec).to_json() for vec in sampled]


def sample_per_head(
    tensor: torch.Tensor,
    sampler: VectorSampler,
    quantizer: Quantizer,
) -> List[List[Dict[str, object]]]:
    arr = sampler.sample(tensor).squeeze(0).cpu().numpy()  # (n_head, T, dim_samples)
    per_head: List[List[Dict[str, object]]] = []
    for head in arr:
        per_head.append([quantizer.encode(token).to_json() for token in head])
    return per_head


def serialize_lower_triangular(
    tensor: torch.Tensor,
    quantizer: Quantizer,
) -> List[Dict[str, object]]:
    arr = tensor.squeeze(0).cpu().numpy()  # (n_head, T, T)
    payload: List[Dict[str, object]] = []
    for head_matrix in arr:
        tril = np.tril(head_matrix)
        tril[np.isneginf(tril)] = -1e9
        mask = np.tri(tril.shape[0], dtype=bool)
        packed = tril[mask]
        quant = quantizer.encode(packed, layout="lower-triangular")
        payload.append({
            "size": head_matrix.shape[0],
            "values": quant.to_json(),
        })
    return payload


def linear_flops(seq_len: int, in_dim: int, out_dim: int) -> float:
    return 2.0 * seq_len * in_dim * out_dim + seq_len * out_dim


def layernorm_core_flops(seq_len: int, dim: int) -> float:
    return seq_len * dim * 5.0


def elementwise_flops(seq_len: int, dim: int) -> float:
    return seq_len * dim


def attention_scores_flops(seq_len: int, head_dim: int, num_heads: int) -> float:
    dot = 2.0 * num_heads * seq_len * seq_len * head_dim
    scale = num_heads * seq_len * seq_len
    return dot + scale


def attention_weighting_flops(seq_len: int, head_dim: int, num_heads: int) -> float:
    softmax = num_heads * seq_len * seq_len * 4.0
    weighted = 2.0 * num_heads * seq_len * seq_len * head_dim
    return softmax + weighted


def mlp_activation_flops(seq_len: int, dim: int) -> float:
    # Approximate GELU cost (tanh implementation)
    return seq_len * dim * 6.0


def summarise_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    quantizer: Quantizer,
) -> List[Dict[str, object]]:
    probs = torch.softmax(logits, dim=-1)
    summary: List[Dict[str, object]] = []
    for pos in range(logits.size(0)):
        pos_entry: Dict[str, object] = {
            "position": pos,
            "token_id": int(input_ids[pos].item()),
        }
        if top_k > 0:
            tk_probs, tk_indices = torch.topk(probs[pos], k=top_k)
            tk_logits = torch.gather(logits[pos], -1, tk_indices)
            pos_entry["top_k"] = package_token_stats(
                tk_indices,
                tk_logits,
                tk_probs,
                quantizer,
            )
        if top_p < 1.0:
            tp_pairs = _extract_top_p_probs(probs[pos], top_p)
            indices = tp_pairs[:, 0].long()
            tp_logits = logits[pos][indices]
            pos_entry["top_p"] = package_token_stats(
                indices,
                tp_logits,
                tp_pairs[:, 1],
                quantizer,
            )
        summary.append(pos_entry)
    return summary


def capture_activations(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    sampler: VectorSampler,
    quantizer: Quantizer,
    *,
    top_k: int,
    top_p: float,
) -> Dict[str, object]:
    model.eval()
    device = model.device
    input_ids = input_ids.to(device)
    with torch.no_grad():
        seq_len = input_ids.size(1)
        config: GPT2Config = model.config
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        tracker = CostTracker()

        token_embed = model.transformer.wte(input_ids)
        position_embed = model.transformer.wpe(position_ids)
        combined = token_embed + position_embed

        tracker.add("embeddings/token", 0.0)
        tracker.add("embeddings/position", 0.0)
        tracker.add("embeddings/sum", elementwise_flops(seq_len, config.n_embd))

        data: Dict[str, object] = {
            "sequence_length": seq_len,
            "embeddings": {
                "token": sample_per_token(token_embed, sampler, quantizer),
                "position": sample_per_token(position_embed, sampler, quantizer),
                "summed": sample_per_token(combined, sampler, quantizer),
            },
            "layers": [],
        }

        residual = combined
        num_heads = config.n_head
        head_dim = config.n_embd // config.n_head
        mlp_dim = getattr(config, "n_inner", None) or (4 * config.n_embd)

        for layer_idx, block in enumerate(model.transformer.h):
            layer_prefix = f"layer_{layer_idx:02d}"
            layer_payload: Dict[str, object] = {
                "index": layer_idx,
                "residual_in": sample_per_token(residual, sampler, quantizer),
            }

            ln1 = layer_norm_breakdown(residual, block.ln_1)
            tracker.add(f"{layer_prefix}/ln1/norm", layernorm_core_flops(seq_len, config.n_embd))
            tracker.add(f"{layer_prefix}/ln1/scale", elementwise_flops(seq_len, config.n_embd))
            tracker.add(f"{layer_prefix}/ln1/shift", elementwise_flops(seq_len, config.n_embd))

            layer_payload["ln1"] = {
                "norm": sample_per_token(ln1.normalised, sampler, quantizer),
                "scaled": sample_per_token(ln1.scaled, sampler, quantizer),
                "shifted": sample_per_token(ln1.shifted, sampler, quantizer),
            }

            attn_input = ln1.shifted
            qkv = block.attn.c_attn(attn_input)
            tracker.add(
                f"{layer_prefix}/attn/qkv",
                linear_flops(seq_len, config.n_embd, 3 * config.n_embd),
            )

            q, k, v = qkv.split(config.n_embd, dim=2)
            q = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            bias = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attn_scores = attn_scores.masked_fill(bias == 0, float("-inf"))
            tracker.add(
                f"{layer_prefix}/attn/scores",
                attention_scores_flops(seq_len, head_dim, num_heads),
            )
            attn_probs = torch.softmax(attn_scores, dim=-1)
            tracker.add(
                f"{layer_prefix}/attn/weights",
                attention_weighting_flops(seq_len, head_dim, num_heads),
            )
            attn_context = torch.matmul(attn_probs, v)

            layer_payload["attention"] = {
                "q": sample_per_head(q, sampler, quantizer),
                "k": sample_per_head(k, sampler, quantizer),
                "v": sample_per_head(v, sampler, quantizer),
                "pre_softmax": serialize_lower_triangular(attn_scores, quantizer),
                "post_softmax": serialize_lower_triangular(attn_probs, quantizer),
            }

            attn_context = attn_context.transpose(1, 2).contiguous().view(1, seq_len, config.n_embd)
            tracker.add(f"{layer_prefix}/attn/concat", 0.0)

            attn_out = block.attn.c_proj(attn_context)
            tracker.add(
                f"{layer_prefix}/attn/out_proj",
                linear_flops(seq_len, config.n_embd, config.n_embd),
            )

            layer_payload["attention"]["projected"] = sample_per_token(attn_out, sampler, quantizer)

            residual = residual + attn_out
            tracker.add(f"{layer_prefix}/attn/residual_add", elementwise_flops(seq_len, config.n_embd))
            layer_payload["residual_after_attn"] = sample_per_token(residual, sampler, quantizer)

            ln2 = layer_norm_breakdown(residual, block.ln_2)
            tracker.add(f"{layer_prefix}/ln2/norm", layernorm_core_flops(seq_len, config.n_embd))
            tracker.add(f"{layer_prefix}/ln2/scale", elementwise_flops(seq_len, config.n_embd))
            tracker.add(f"{layer_prefix}/ln2/shift", elementwise_flops(seq_len, config.n_embd))

            layer_payload["ln2"] = {
                "norm": sample_per_token(ln2.normalised, sampler, quantizer),
                "scaled": sample_per_token(ln2.scaled, sampler, quantizer),
                "shifted": sample_per_token(ln2.shifted, sampler, quantizer),
            }

            mlp_up = block.mlp.c_fc(ln2.shifted)
            tracker.add(
                f"{layer_prefix}/mlp/up_proj",
                linear_flops(seq_len, config.n_embd, mlp_dim),
            )
            mlp_act = block.mlp.act(mlp_up)
            tracker.add(f"{layer_prefix}/mlp/activation", mlp_activation_flops(seq_len, mlp_dim))
            mlp_down = block.mlp.c_proj(mlp_act)
            tracker.add(
                f"{layer_prefix}/mlp/down_proj",
                linear_flops(seq_len, mlp_dim, config.n_embd),
            )

            layer_payload["mlp"] = {
                "up": sample_per_token(mlp_up, sampler, quantizer),
                "activated": sample_per_token(mlp_act, sampler, quantizer),
                "down": sample_per_token(mlp_down, sampler, quantizer),
            }

            residual = residual + mlp_down
            tracker.add(f"{layer_prefix}/mlp/residual_add", elementwise_flops(seq_len, config.n_embd))
            layer_payload["residual_out"] = sample_per_token(residual, sampler, quantizer)

            data["layers"].append(layer_payload)

        ln_final = layer_norm_breakdown(residual, model.transformer.ln_f)
        tracker.add("final_ln/norm", layernorm_core_flops(seq_len, config.n_embd))
        tracker.add("final_ln/scale", elementwise_flops(seq_len, config.n_embd))
        tracker.add("final_ln/shift", elementwise_flops(seq_len, config.n_embd))

        final_payload = {
            "norm": sample_per_token(ln_final.normalised, sampler, quantizer),
            "scaled": sample_per_token(ln_final.scaled, sampler, quantizer),
            "shifted": sample_per_token(ln_final.shifted, sampler, quantizer),
        }

        data["final_layernorm"] = final_payload
        logits = model.lm_head(ln_final.shifted).squeeze(0)
        data["filtered_logits"] = summarise_logits(
            logits.cpu(),
            input_ids.squeeze(0).cpu(),
            top_k=top_k,
            top_p=top_p,
            quantizer=quantizer,
        )
        data["flops"] = [dataclasses.asdict(cp) for cp in tracker.checkpoints]

    return data


def compute_parameter_checkpoints(model: GPT2LMHeadModel) -> List[Dict[str, object]]:
    config: GPT2Config = model.config
    counts: List[Dict[str, object]] = []
    total = 0

    def push(name: str, increment: int) -> None:
        nonlocal total
        total += int(increment)
        counts.append({"name": name, "increment": int(increment), "cumulative": total})

    push("embeddings/token", model.transformer.wte.weight.numel())
    push("embeddings/position", model.transformer.wpe.weight.numel())

    for layer_idx, block in enumerate(model.transformer.h):
        prefix = f"layer_{layer_idx:02d}"
        push(f"{prefix}/ln1/scale", block.ln_1.weight.numel())
        push(f"{prefix}/ln1/shift", block.ln_1.bias.numel())
        push(f"{prefix}/attn/qkv", block.attn.c_attn.weight.numel() + block.attn.c_attn.bias.numel())
        push(f"{prefix}/attn/out_proj", block.attn.c_proj.weight.numel() + block.attn.c_proj.bias.numel())
        push(f"{prefix}/ln2/scale", block.ln_2.weight.numel())
        push(f"{prefix}/ln2/shift", block.ln_2.bias.numel())
        push(f"{prefix}/mlp/up_proj", block.mlp.c_fc.weight.numel() + block.mlp.c_fc.bias.numel())
        push(f"{prefix}/mlp/down_proj", block.mlp.c_proj.weight.numel() + block.mlp.c_proj.bias.numel())

    push("final_ln/scale", model.transformer.ln_f.weight.numel())
    push("final_ln/shift", model.transformer.ln_f.bias.numel())
    push("lm_head", model.lm_head.weight.numel())

    return counts


# ---------------------------------------------------------------------------
# CLI


def interactive_choice(candidates: List[Dict[str, object]], tokenizer: GPT2TokenizerFast) -> Tuple[List[int], Dict[str, object]]:
    print("\nGenerated completions:")
    for idx, cand in enumerate(candidates):
        preview = cand["text"].strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"[{idx}] {preview or '<empty>'}")

    selection = input("Select completion index (or 'm' to enter manually): ").strip()
    if selection.lower() == "m":
        manual = input("Enter manual completion text: ")
        tokens = tokenizer.encode(manual, add_special_tokens=False)
        return tokens, {"text": manual, "trace": []}

    try:
        choice = int(selection)
    except ValueError as exc:  # pragma: no cover - user input path
        raise SystemExit(f"Invalid selection: {selection}") from exc

    if not (0 <= choice < len(candidates)):
        raise SystemExit(f"Selection {choice} is out of range")

    chosen = candidates[choice]
    truncation = input("Optional: truncate completion to N tokens (press enter to keep all): ").strip()
    if truncation:
        try:
            limit = int(truncation)
        except ValueError as exc:  # pragma: no cover - user input path
            raise SystemExit(f"Invalid truncation value: {truncation}") from exc
        tokens = chosen["tokens"][:limit]
        chosen["tokens"] = tokens
        chosen["trace"] = chosen["trace"][:limit]
        chosen["text"] = tokenizer.decode(tokens, skip_special_tokens=True)
    else:
        tokens = chosen["tokens"]

    return tokens, chosen


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture GPT-2 activations for visualisation")
    parser.add_argument("--prompt", type=str, default=None, help="Initial prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum number of generated tokens per completion")
    parser.add_argument("--num-completions", type=int, default=4, help="Number of candidate completions to sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p (nucleus) sampling threshold")
    parser.add_argument("--stride", type=int, default=32, help="Sampling stride for vector states")
    parser.add_argument("--quantization", type=str, default="float16", choices=["float32", "float16", "int8"], help="Quantization mode for stored activations")
    parser.add_argument("--output", type=Path, default=Path("captures/latest_capture.json"), help="Output JSON file")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(args.device)

    prompt_text = args.prompt or input("Enter prompt: ")
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(encoded_prompt) == 0:
        raise SystemExit("Prompt must not be empty")

    input_ids = torch.tensor(encoded_prompt, dtype=torch.long, device=args.device).unsqueeze(0)

    quantizer = Quantizer(args.quantization)
    sampler = VectorSampler(args.stride)

    print("\nSampling completions...", flush=True)
    candidates = generate_candidates(
        model,
        tokenizer,
        input_ids,
        num_completions=args.num_completions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        quantizer=quantizer,
    )

    completion_tokens, chosen = interactive_choice(candidates, tokenizer)

    full_ids = torch.tensor(encoded_prompt + completion_tokens, dtype=torch.long, device=args.device).unsqueeze(0)

    capture = capture_activations(
        model,
        full_ids,
        sampler,
        quantizer,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    capture["prompt"] = {
        "text": prompt_text,
        "tokens": encoded_prompt,
    }
    capture["completion"] = {
        "tokens": completion_tokens,
        "text": chosen["text"],
    }
    capture["generation_trace"] = chosen.get("trace", [])
    capture["parameter_checkpoints"] = compute_parameter_checkpoints(model)
    capture["quantization"] = args.quantization
    capture["stride"] = args.stride

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(capture, f, indent=2)
    print(f"Saved capture to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

