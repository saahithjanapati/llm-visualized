#!/usr/bin/env python3
"""Interactive GPT-2 (124M) activation capture utility.

This script generates candidate completions for a prompt, lets the user pick
one (or provide their own), and then records a compact snapshot of internal
GPT-2 activations suitable for driving the llm-visualized Three.js frontend.

It also tracks per-stage FLOP usage during autoregressive generation and stores
per-step sampling logits for the configured top-k / top-p sampler.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

CheckpointDict = Dict[str, float]


# ---------------------------------------------------------------------------
# Quantisation helpers
# ---------------------------------------------------------------------------

@dataclass
class QuantConfig:
    mode: str = "float16"  # "none", "float16", "int8"


class Quantiser:
    def __init__(self, config: QuantConfig):
        self.config = config
        if self.config.mode not in {"none", "float16", "int8"}:
            raise ValueError(f"Unsupported quantisation mode: {self.config.mode}")

    def encode(self, array: np.ndarray) -> Dict[str, object]:
        if self.config.mode == "none":
            return {
                "encoding": "raw",
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data": array.tolist(),
            }

        if self.config.mode == "float16":
            arr = array.astype(np.float16, copy=False)
            payload = base64.b64encode(arr.tobytes()).decode("ascii")
            return {
                "encoding": "base64",
                "dtype": "float16",
                "shape": list(arr.shape),
                "data": payload,
            }

        # int8 symmetric quantisation with per-tensor scale
        max_abs = float(np.max(np.abs(array)))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        quantised = np.clip(np.round(array / scale), -128, 127).astype(np.int8, copy=False)
        payload = base64.b64encode(quantised.tobytes()).decode("ascii")
        return {
            "encoding": "base64",
            "dtype": "int8",
            "shape": list(quantised.shape),
            "scale": scale,
            "data": payload,
        }


# ---------------------------------------------------------------------------
# Utility math helpers
# ---------------------------------------------------------------------------


def sample_features(tensor: torch.Tensor, stride: int) -> torch.Tensor:
    """Samples evenly spaced features along the last dimension."""

    if stride <= 0:
        raise ValueError("Stride must be positive")
    last_dim = tensor.size(-1)
    idx = torch.arange(0, last_dim, stride, device=tensor.device)
    return tensor.index_select(-1, idx)


def layer_norm_states(layernorm: torch.nn.LayerNorm, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (normed, scaled, shifted) tensors for a LayerNorm."""

    mean = hidden_states.mean(dim=-1, keepdim=True)
    variance = (hidden_states - mean).pow(2).mean(dim=-1, keepdim=True)
    normed = (hidden_states - mean) / torch.sqrt(variance + layernorm.eps)
    scaled = normed * layernorm.weight
    shifted = scaled + layernorm.bias
    return normed, scaled, shifted


def flatten_causal(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten a causal [heads, T, T] tensor into [heads, T*(T+1)/2]."""

    heads, seq_len, _ = matrix.shape
    tri_entries = seq_len * (seq_len + 1) // 2
    out = matrix.new_zeros((heads, tri_entries))
    offset = 0
    for t in range(seq_len):
        length = t + 1
        out[:, offset : offset + length] = matrix[:, t, : length]
        offset += length
    return out


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------

@dataclass
class ModelDim:
    hidden: int = 768
    intermediate: int = 3072
    n_heads: int = 12
    head_dim: int = 64
    n_layer: int = 12
    vocab_size: int = 50257


class FlopEstimator:
    """Estimates FLOPs for GPT-2 checkpoints during generation."""

    def __init__(self, dims: ModelDim):
        self.dims = dims

    def layernorm_norm(self) -> float:
        # mean + variance + normalisation (very approximate)
        d = self.dims.hidden
        return 6.0 * d

    def layernorm_scale_or_shift(self) -> float:
        return float(self.dims.hidden)

    def dense_flops(self, in_dim: int, out_dim: int) -> float:
        # multiply + add per output
        return 2.0 * in_dim * out_dim

    def attention_scores(self, context_length: int) -> float:
        # q @ k^T per head
        return 2.0 * self.dims.n_heads * context_length * self.dims.head_dim

    def attention_weighted_sum(self, context_length: int) -> float:
        return 2.0 * self.dims.n_heads * context_length * self.dims.head_dim

    def checkpoints_for_token(self, context_length: int) -> CheckpointDict:
        cp: CheckpointDict = {}
        dims = self.dims

        total = 0.0
        total += dims.hidden  # embedding addition
        cp["after_embeddings"] = total

        for layer in range(dims.n_layer):
            ln_base = f"layer_{layer:02d}"

            ln_ops = self.layernorm_norm()
            total += ln_ops
            cp[f"{ln_base}/ln1_norm"] = total

            total += self.layernorm_scale_or_shift()
            cp[f"{ln_base}/ln1_scale"] = total

            total += self.layernorm_scale_or_shift()
            cp[f"{ln_base}/ln1_shift"] = total

            total += self.dense_flops(dims.hidden, 3 * dims.hidden)
            cp[f"{ln_base}/qkv_proj"] = total

            total += self.attention_scores(context_length)
            cp[f"{ln_base}/self_attention_scores"] = total

            total += self.attention_weighted_sum(context_length)
            cp[f"{ln_base}/self_attention_context"] = total

            cp[f"{ln_base}/self_attention_concat"] = total

            total += self.dense_flops(dims.hidden, dims.hidden)
            cp[f"{ln_base}/attn_output_proj"] = total

            total += dims.hidden  # residual add
            cp[f"{ln_base}/residual_add_1"] = total

            # LN2
            total += self.layernorm_norm()
            cp[f"{ln_base}/ln2_norm"] = total

            total += self.layernorm_scale_or_shift()
            cp[f"{ln_base}/ln2_scale"] = total

            total += self.layernorm_scale_or_shift()
            cp[f"{ln_base}/ln2_shift"] = total

            total += self.dense_flops(dims.hidden, dims.intermediate)
            cp[f"{ln_base}/mlp_up_proj"] = total

            total += dims.intermediate  # GELU approx as linear cost
            cp[f"{ln_base}/mlp_activation"] = total

            total += self.dense_flops(dims.intermediate, dims.hidden)
            cp[f"{ln_base}/mlp_down_proj"] = total

            total += dims.hidden
            cp[f"{ln_base}/residual_add_2"] = total

        total += self.layernorm_norm()
        cp["final_ln_norm"] = total
        total += self.layernorm_scale_or_shift()
        cp["final_ln_scale"] = total
        total += self.layernorm_scale_or_shift()
        cp["final_ln_shift"] = total

        total += self.dense_flops(dims.hidden, dims.vocab_size)
        cp["lm_head"] = total

        return cp


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

@dataclass
class CaptureConfig:
    stride_residual: int = 32
    stride_mlp: int = 32
    stride_qkv: int = 32
    quant: QuantConfig = QuantConfig()


class ActivationCollector:
    def __init__(self, model: GPT2LMHeadModel, config: CaptureConfig):
        self.model = model
        self.config = config
        self.quantiser = Quantiser(config.quant)

    def _encode(self, tensor: torch.Tensor, stride: int) -> Dict[str, object]:
        sampled = sample_features(tensor, stride)
        array = sampled.detach().cpu().numpy()
        return self.quantiser.encode(array)

    def collect(self, input_ids: torch.LongTensor) -> Dict[str, object]:
        device = self.model.device
        input_ids = input_ids.to(device)
        batch_size, seq_len = input_ids.shape
        if batch_size != 1:
            raise ValueError("Only batch_size=1 is supported for capture")

        config = self.model.config
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)

        wte = self.model.transformer.wte
        wpe = self.model.transformer.wpe

        with torch.no_grad():
            token_embeds = wte(input_ids)
            pos_embeds = wpe(position_ids)
            hidden_states = token_embeds + pos_embeds

        capture: Dict[str, object] = {
            "meta": {
                "model_name": config._name_or_path,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "stride_residual": self.config.stride_residual,
                "stride_mlp": self.config.stride_mlp,
                "stride_qkv": self.config.stride_qkv,
                "quantisation": self.config.quant.mode,
            },
            "tokens": input_ids[0].tolist(),
            "embeddings": {
                "token": self._encode(token_embeds.squeeze(0), self.config.stride_residual),
                "position": self._encode(pos_embeds.squeeze(0), self.config.stride_residual),
                "sum": self._encode(hidden_states.squeeze(0), self.config.stride_residual),
            },
            "layers": [],
        }

        attn_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).unsqueeze(0)

        for layer_idx, block in enumerate(self.model.transformer.h):
            layer_key = f"layer_{layer_idx:02d}"
            layer_record: Dict[str, object] = {
                "residual_in": self._encode(hidden_states.squeeze(0), self.config.stride_residual)
            }

            with torch.no_grad():
                normed, scaled, shifted = layer_norm_states(block.ln_1, hidden_states)
            layer_record["ln1_norm"] = self._encode(normed.squeeze(0), self.config.stride_residual)
            layer_record["ln1_scale"] = self._encode(scaled.squeeze(0), self.config.stride_residual)
            layer_record["ln1_shift"] = self._encode(shifted.squeeze(0), self.config.stride_residual)

            with torch.no_grad():
                qkv = block.attn.c_attn(shifted)
            q, k, v = qkv.split(self.model.config.n_embd, dim=-1)

            def reshape_heads(t: torch.Tensor) -> torch.Tensor:
                b, s, dim = t.size()
                head_dim = dim // self.model.config.n_head
                t = t.view(b, s, self.model.config.n_head, head_dim)
                return t.permute(0, 2, 1, 3)  # (b, heads, seq, head_dim)

            q_heads = reshape_heads(q)
            k_heads = reshape_heads(k)
            v_heads = reshape_heads(v)

            def encode_heads(t: torch.Tensor, stride: int) -> Dict[str, object]:
                b, h, s, d = t.shape
                sampled = sample_features(t, stride)
                array = sampled.squeeze(0).detach().cpu().numpy()  # (heads, seq, samples)
                return self.quantiser.encode(array)

            qkv_stride = max(1, self.config.stride_qkv)
            layer_record["q_heads"] = encode_heads(q_heads, qkv_stride)
            layer_record["k_heads"] = encode_heads(k_heads, qkv_stride)
            layer_record["v_heads"] = encode_heads(v_heads, qkv_stride)

            with torch.no_grad():
                attn_scores = torch.matmul(q_heads, k_heads.transpose(-1, -2))
                attn_scores = attn_scores / math.sqrt(self.model.config.n_embd // self.model.config.n_head)
                attn_scores = attn_scores.masked_fill(~attn_mask, torch.finfo(attn_scores.dtype).min)
                attn_probs = torch.softmax(attn_scores, dim=-1)

            flat_scores = flatten_causal(attn_scores.squeeze(0))
            flat_probs = flatten_causal(attn_probs.squeeze(0))

            layer_record["attention_pre_softmax"] = self.quantiser.encode(flat_scores.detach().cpu().numpy())
            layer_record["attention_post_softmax"] = self.quantiser.encode(flat_probs.detach().cpu().numpy())
            layer_record["attention_tri_seq_len"] = seq_len

            with torch.no_grad():
                attn_output = torch.matmul(attn_probs, v_heads)  # (b, heads, seq, head_dim)
                attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
                attn_output = attn_output.view(batch_size, seq_len, self.model.config.n_embd)

                attn_proj = block.attn.c_proj(attn_output)
            layer_record["attn_out_proj"] = self._encode(attn_proj.squeeze(0), self.config.stride_residual)

            hidden_states = hidden_states + attn_proj
            layer_record["residual_after_attn"] = self._encode(hidden_states.squeeze(0), self.config.stride_residual)

            with torch.no_grad():
                normed2, scaled2, shifted2 = layer_norm_states(block.ln_2, hidden_states)
            layer_record["ln2_norm"] = self._encode(normed2.squeeze(0), self.config.stride_residual)
            layer_record["ln2_scale"] = self._encode(scaled2.squeeze(0), self.config.stride_residual)
            layer_record["ln2_shift"] = self._encode(shifted2.squeeze(0), self.config.stride_residual)

            with torch.no_grad():
                mlp_up = block.mlp.c_fc(shifted2)
            layer_record["mlp_up_proj"] = self._encode(mlp_up.squeeze(0), self.config.stride_mlp)

            with torch.no_grad():
                mlp_act = block.mlp.act(mlp_up)
            layer_record["mlp_activation"] = self._encode(mlp_act.squeeze(0), self.config.stride_mlp)

            with torch.no_grad():
                mlp_down = block.mlp.c_proj(mlp_act)
            layer_record["mlp_down_proj"] = self._encode(mlp_down.squeeze(0), self.config.stride_residual)

            hidden_states = hidden_states + mlp_down
            layer_record["residual_after_mlp"] = self._encode(hidden_states.squeeze(0), self.config.stride_residual)

            capture["layers"].append(layer_record)

        with torch.no_grad():
            final_normed, final_scaled, final_shifted = layer_norm_states(self.model.transformer.ln_f, hidden_states)
        capture["final_layernorm"] = {
            "norm": self._encode(final_normed.squeeze(0), self.config.stride_residual),
            "scale": self._encode(final_scaled.squeeze(0), self.config.stride_residual),
            "shift": self._encode(final_shifted.squeeze(0), self.config.stride_residual),
        }

        with torch.no_grad():
            logits = self.model.lm_head(final_shifted)
        capture["logits"] = self.quantiser.encode(logits.squeeze(0).detach().cpu().numpy())
        capture["vocab_size"] = self.model.config.vocab_size
        capture["seq_len"] = seq_len
        return capture


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int, top_p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns filtered logits and their token indices."""

    logits = logits.clone()
    indices = torch.arange(logits.size(-1), device=logits.device, dtype=torch.long).unsqueeze(0)

    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[..., -1, None]
        mask = logits < min_values
        logits = logits.masked_fill(mask, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    mask = torch.isfinite(logits)
    filtered_indices = indices.masked_select(mask).view(1, -1)
    filtered_logits = logits.masked_select(mask).view(1, -1)
    return filtered_logits, filtered_indices


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    num_completions: int = 3
    seed: Optional[int] = None


@dataclass
class Completion:
    text: str
    token_ids: List[int]
    new_token_ids: List[int]
    sampling_trace: List[Dict[str, object]]


class CompletionGenerator:
    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast, gen_config: GenerationConfig, quant: Quantiser):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.quant = quant

    def generate(self, prompt_ids: torch.LongTensor) -> List[Completion]:
        device = self.model.device
        prompt_ids = prompt_ids.to(device)
        completions: List[Completion] = []
        generator = torch.Generator(device=device)
        for idx in range(self.gen_config.num_completions):
            if self.gen_config.seed is not None:
                generator.manual_seed(self.gen_config.seed + idx)

            input_ids = prompt_ids.clone()
            past_key_values = None
            sampling_trace: List[Dict[str, object]] = []
            new_tokens: List[int] = []

            with torch.no_grad():
                for step in range(self.gen_config.max_new_tokens):
                    outputs = self.model(input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                                          past_key_values=past_key_values,
                                          use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values

                    logits = logits / max(self.gen_config.temperature, 1e-5)
                    filtered_logits, filtered_indices = top_k_top_p_filtering(logits, self.gen_config.top_k, self.gen_config.top_p)

                    if filtered_indices.size(-1) == 0:
                        break

                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1, generator=generator)
                    next_token = filtered_indices.gather(-1, next_idx)

                    sampling_trace.append({
                        "candidate_ids": filtered_indices.squeeze(0).tolist(),
                        "logits": self.quant.encode(filtered_logits.detach().cpu().numpy()),
                        "probs": self.quant.encode(probs.detach().cpu().numpy()),
                        "chosen_token": int(next_token.item()),
                    })

                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    new_token = int(next_token.item())
                    new_tokens.append(new_token)

                    if new_token == self.tokenizer.eos_token_id:
                        break

            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(Completion(text=text,
                                          token_ids=input_ids[0].tolist(),
                                          new_token_ids=new_tokens,
                                          sampling_trace=sampling_trace))
        return completions


# ---------------------------------------------------------------------------
# Sampling trace replay
# ---------------------------------------------------------------------------


def replay_sampling_trace(model: GPT2LMHeadModel,
                          tokens: Sequence[int],
                          prompt_length: int,
                          gen_config: GenerationConfig,
                          quant: Quantiser) -> List[Dict[str, object]]:
    if len(tokens) <= prompt_length:
        return []

    device = model.device
    input_ids = torch.tensor(tokens[:prompt_length], device=device, dtype=torch.long).unsqueeze(0)
    past_key_values = None
    trace: List[Dict[str, object]] = []

    with torch.no_grad():
        for target in tokens[prompt_length:]:
            outputs = model(input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            logits = logits / max(gen_config.temperature, 1e-5)
            filtered_logits, filtered_indices = top_k_top_p_filtering(logits, gen_config.top_k, gen_config.top_p)

            if filtered_indices.numel() == 0:
                # ensure the chosen token is represented even if the filter removed everything
                filtered_indices = torch.tensor([[target]], device=device, dtype=torch.long)
                filtered_logits = logits.gather(-1, filtered_indices)

            probs = torch.softmax(filtered_logits, dim=-1)

            ids_list = filtered_indices.squeeze(0).tolist()
            if target not in ids_list:
                ids_list.append(int(target))
                extra_logit = logits.gather(-1, torch.tensor([[target]], device=device, dtype=torch.long))
                filtered_logits = torch.cat([filtered_logits, extra_logit], dim=-1)
                probs = torch.softmax(filtered_logits, dim=-1)

            trace.append({
                "candidate_ids": ids_list,
                "logits": quant.encode(filtered_logits.detach().cpu().numpy()),
                "probs": quant.encode(probs.detach().cpu().numpy()),
                "chosen_token": int(target),
            })

            next_token_tensor = torch.tensor([[target]], device=device, dtype=torch.long)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)

    return trace


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def parameter_report(model: GPT2LMHeadModel) -> Dict[str, int]:
    report: Dict[str, int] = {}
    total = 0

    embeddings = model.transformer.wte.weight.numel() + model.transformer.wpe.weight.numel()
    total += embeddings
    report["after_embeddings"] = total

    for idx, block in enumerate(model.transformer.h):
        prefix = f"layer_{idx:02d}"

        report[f"{prefix}/ln1_norm"] = total
        total += block.ln_1.weight.numel()
        report[f"{prefix}/ln1_scale"] = total
        total += block.ln_1.bias.numel()
        report[f"{prefix}/ln1_shift"] = total

        total += block.attn.c_attn.weight.numel() + block.attn.c_attn.bias.numel()
        report[f"{prefix}/qkv_proj"] = total
        report[f"{prefix}/self_attention_scores"] = total
        report[f"{prefix}/self_attention_context"] = total
        report[f"{prefix}/self_attention_concat"] = total

        total += block.attn.c_proj.weight.numel() + block.attn.c_proj.bias.numel()
        report[f"{prefix}/attn_output_proj"] = total
        report[f"{prefix}/residual_add_1"] = total

        report[f"{prefix}/ln2_norm"] = total
        total += block.ln_2.weight.numel()
        report[f"{prefix}/ln2_scale"] = total
        total += block.ln_2.bias.numel()
        report[f"{prefix}/ln2_shift"] = total

        total += block.mlp.c_fc.weight.numel() + block.mlp.c_fc.bias.numel()
        report[f"{prefix}/mlp_up_proj"] = total
        report[f"{prefix}/mlp_activation"] = total

        total += block.mlp.c_proj.weight.numel() + block.mlp.c_proj.bias.numel()
        report[f"{prefix}/mlp_down_proj"] = total
        report[f"{prefix}/residual_add_2"] = total

    report["final_ln_norm"] = total
    total += model.transformer.ln_f.weight.numel()
    report["final_ln_scale"] = total
    total += model.transformer.ln_f.bias.numel()
    report["final_ln_shift"] = total

    total += model.lm_head.weight.numel()
    report["lm_head"] = total

    report["total_parameters"] = total
    return report


# ---------------------------------------------------------------------------
# User interaction helpers
# ---------------------------------------------------------------------------


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-2 activation capture tool")
    parser.add_argument("--prompt", help="Seed prompt text. If omitted an interactive prompt is shown.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-completions", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stride", type=int, default=32, help="Stride for residual stream sampling")
    parser.add_argument("--mlp-stride", type=int, default=32, help="Stride for MLP activations")
    parser.add_argument("--qkv-stride", type=int, default=32, help="Stride for per-head Q/K/V sampling")
    parser.add_argument("--quant", choices=["none", "float16", "int8"], default="float16")
    parser.add_argument("--output", default="capture.json", help="Where to store the capture JSON")
    parser.add_argument("--param-report", default="gpt2_parameter_report.json")
    return parser.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 Loading GPT-2 (small)...", file=sys.stderr)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    quant_config = QuantConfig(mode=args.quant)
    capture_config = CaptureConfig(stride_residual=args.stride,
                                   stride_mlp=args.mlp_stride,
                                   stride_qkv=args.qkv_stride,
                                   quant=quant_config)

    if args.prompt is None:
        prompt_text = prompt_user("Enter a prompt to kick things off: ")
    else:
        prompt_text = args.prompt

    if not prompt_text:
        print("No prompt provided. Aborting.")
        return 1

    encoded_prompt = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = encoded_prompt["input_ids"].to(device)

    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature,
                                  top_k=args.top_k,
                                  top_p=args.top_p,
                                  num_completions=args.num_completions,
                                  seed=args.seed)

    quantiser = Quantiser(quant_config)
    generator = CompletionGenerator(model, tokenizer, gen_config, quantiser)

    print("\n🌈 Generating sample completions...\n")
    completions = generator.generate(prompt_ids)

    for idx, completion in enumerate(completions):
        print(f"[{idx}] {completion.text!r}")

    selection = prompt_user("\nPick a completion by index, or type 'm' to craft your own: ")
    chosen_tokens: Optional[List[int]] = None
    chosen_sampling: Optional[List[Dict[str, object]]] = None

    if selection.strip().lower() == "m":
        manual = prompt_user("Enter your custom continuation: ")
        manual_ids = tokenizer(manual, return_tensors="pt", add_special_tokens=False)["input_ids"]
        chosen_tokens = torch.cat([prompt_ids, manual_ids.to(device)], dim=-1)[0].tolist()
        chosen_sampling = []
    else:
        try:
            idx = int(selection.strip())
            completion = completions[idx]
            chosen_tokens = completion.token_ids
            chosen_sampling = completion.sampling_trace
        except (ValueError, IndexError):
            print("Invalid selection.")
            return 1

    prompt_token_count = prompt_ids.size(1)
    total_tokens = len(chosen_tokens)
    generated_tokens = total_tokens - prompt_token_count

    if generated_tokens > 0:
        print("\nGenerated token breakdown (GPT-2 BPE ids):")
        for i, token_id in enumerate(chosen_tokens[prompt_token_count:], start=1):
            token_piece = tokenizer.decode([token_id])
            print(f"  {i:>3}: id={token_id:<5} piece={token_piece!r}")

        truncate_resp = prompt_user("\nKeep how many generated tokens? (press Enter for all): ")
        if truncate_resp.strip():
            try:
                keep = int(truncate_resp.strip())
            except ValueError:
                print("Invalid truncate value.")
                return 1
            keep = max(0, min(keep, generated_tokens))
            chosen_tokens = chosen_tokens[: prompt_token_count + keep]
            if chosen_sampling is not None:
                chosen_sampling = chosen_sampling[:keep]
            generated_tokens = keep

    chosen_sampling = replay_sampling_trace(model, chosen_tokens, prompt_token_count, gen_config, quantiser)

    final_text = tokenizer.decode(chosen_tokens)
    print(f"\n✨ Capturing activations for:\n{final_text}\n")

    capture = ActivationCollector(model, capture_config).collect(torch.tensor([chosen_tokens], device=device, dtype=torch.long))
    capture["prompt"] = prompt_text
    capture["completion"] = final_text[len(prompt_text):]
    capture["sampling_trace"] = chosen_sampling
    capture["prompt_token_count"] = prompt_token_count
    capture["sampling_config"] = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    dims = ModelDim(hidden=model.config.n_embd,
                    intermediate=getattr(model.config, "n_inner", model.config.n_embd * 4),
                    n_heads=model.config.n_head,
                    head_dim=model.config.n_embd // model.config.n_head,
                    n_layer=model.config.n_layer,
                    vocab_size=model.config.vocab_size)
    flop_estimator = FlopEstimator(dims)
    gen_flops: List[CheckpointDict] = []
    context_length = prompt_ids.size(1)
    for step in range(len(chosen_tokens) - context_length):
        cp = flop_estimator.checkpoints_for_token(context_length + step + 1)
        gen_flops.append(cp)
    capture["generation_flops"] = gen_flops

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(capture, f)
    print(f"Saved activation capture to {args.output}")

    params = parameter_report(model)
    with open(args.param_report, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"Parameter breakdown written to {args.param_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
