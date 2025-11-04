"""High level utilities for tracing GPT-2 activations.

The :class:`GPT2TraceCollector` runs a forward pass through GPT-2 (124M) and
captures the activations required by the Three.js visualisation.  Only a sampled
subset of each tensor is recorded to keep the exported payload compact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import GPT2LMHeadModel

from .quantization import QuantizationMode, quantize_array


@dataclass
class TraceConfig:
    stride: int = 32
    head_stride: int | None = None
    mlp_stride: int | None = None
    quantization: QuantizationMode = QuantizationMode.FLOAT16

    def head_step(self) -> int:
        return self.head_stride or self.stride

    def mlp_step(self) -> int:
        return self.mlp_stride or self.stride


def _sample_tensor(tensor: torch.Tensor, step: int) -> torch.Tensor:
    if step <= 0:
        raise ValueError("step must be positive")
    size = tensor.size(-1)
    indices = list(range(0, size, step))
    return tensor.index_select(-1, torch.tensor(indices, device=tensor.device))


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def _quantize(tensor: torch.Tensor, *, config: TraceConfig, meta: Dict[str, object]) -> Dict[str, object]:
    return quantize_array(_to_numpy(tensor), config.quantization, meta=meta).to_json()


def _layer_norm_states(module: nn.LayerNorm, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normed = (x - mean) / torch.sqrt(variance + module.eps)
    scaled = normed * module.weight
    shifted = scaled + module.bias
    return normed, scaled, shifted


def _add_flops(bucket: Dict[str, int], key: str, value: float) -> None:
    bucket[key] = bucket.get(key, 0) + int(value)


class GPT2TraceCollector:
    def __init__(self, model: GPT2LMHeadModel, config: TraceConfig):
        self.model = model
        self.config = config
        self.model.eval()

        if model.config.n_head is None or model.config.n_embd is None:
            raise ValueError("Model must expose n_head and n_embd in its config")

        self.hidden_size = model.config.n_embd
        self.num_heads = model.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_dim = getattr(model.config, "n_inner", None) or 4 * self.hidden_size

    def collect(self, input_ids: torch.Tensor) -> Dict[str, object]:
        device = next(self.model.parameters()).device
        batch = input_ids.unsqueeze(0).to(device)
        seq_len = input_ids.size(0)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        with torch.no_grad():
            token_embeddings = self.model.transformer.wte(batch)
            position_embeddings = self.model.transformer.wpe(position_ids)
            residual = token_embeddings + position_embeddings

            activations: Dict[str, object] = {
                "embeddings": {
                    "token": _quantize(
                        _sample_tensor(token_embeddings.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "source": "token"},
                    ),
                    "position": _quantize(
                        _sample_tensor(position_embeddings.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "source": "position"},
                    ),
                    "summed": _quantize(
                        _sample_tensor(residual.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "source": "token+position"},
                    ),
                },
                "layers": [],
            }

            flops: Dict[str, object] = {
                "embeddings": {},
                "layers": [],
                "final_layernorm": {},
                "total": 0,
            }

            _add_flops(flops["embeddings"], "token_lookup", 0)
            _add_flops(flops["embeddings"], "position_lookup", 0)
            _add_flops(flops["embeddings"], "sum", seq_len * self.hidden_size)
            flops["total"] += sum(flops["embeddings"].values())

            for block in self.model.transformer.h:
                layer_bucket: Dict[str, object] = {}
                layer_flops: Dict[str, int] = {}

                residual_in = residual
                layer_bucket["residual_in"] = _quantize(
                    _sample_tensor(residual_in.squeeze(0), self.config.stride),
                    config=self.config,
                    meta={"stride": self.config.stride, "stage": "layer_input"},
                )

                ln1_norm, ln1_scaled, ln1_shifted = _layer_norm_states(block.ln_1, residual_in)
                layer_bucket["ln1"] = {
                    "norm": _quantize(
                        _sample_tensor(ln1_norm.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln1_norm"},
                    ),
                    "scale": _quantize(
                        _sample_tensor(ln1_scaled.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln1_scale"},
                    ),
                    "shift": _quantize(
                        _sample_tensor(ln1_shifted.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln1_shift"},
                    ),
                }

                _add_flops(layer_flops, "ln1_norm", seq_len * self.hidden_size * 5)
                _add_flops(layer_flops, "ln1_scale", seq_len * self.hidden_size)
                _add_flops(layer_flops, "ln1_shift", seq_len * self.hidden_size)

                attn_input = ln1_shifted
                qkv = block.attn.c_attn(attn_input)
                q, k, v = qkv.split(self.hidden_size, dim=2)
                q = q.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                head_stride = self.config.head_step()
                q_states = _sample_tensor(q.transpose(1, 2).squeeze(0), head_stride)
                k_states = _sample_tensor(k.transpose(1, 2).squeeze(0), head_stride)
                v_states = _sample_tensor(v.transpose(1, 2).squeeze(0), head_stride)

                q_heads = q.squeeze(0)
                k_heads = k.squeeze(0)
                v_heads = v.squeeze(0)

                scale = 1.0 / math.sqrt(self.head_dim)
                attn_scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
                mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
                attn_scores_masked = attn_scores.masked_fill(~mask, 0.0)
                masked_logits = attn_scores.masked_fill(~mask, float("-inf"))
                attn_probs = torch.softmax(masked_logits, dim=-1)
                attn_probs = attn_probs.masked_fill(~mask, 0.0)

                context = torch.matmul(attn_probs, v_heads)
                context = context.transpose(0, 1).contiguous().view(1, seq_len, self.hidden_size)
                attn_output = block.attn.c_proj(context)
                residual = residual + attn_output

                layer_bucket["attention"] = {
                    "q": _quantize(
                        q_states,
                        config=self.config,
                        meta={"stride": head_stride, "stage": "q"},
                    ),
                    "k": _quantize(
                        k_states,
                        config=self.config,
                        meta={"stride": head_stride, "stage": "k"},
                    ),
                    "v": _quantize(
                        v_states,
                        config=self.config,
                        meta={"stride": head_stride, "stage": "v"},
                    ),
                    "pre_softmax": _quantize(
                        attn_scores_masked.permute(1, 0, 2),
                        config=self.config,
                        meta={
                            "shape": [seq_len, self.num_heads, seq_len],
                            "mask": "lower_triangular",
                            "stage": "pre_softmax",
                        },
                    ),
                    "post_softmax": _quantize(
                        attn_probs.permute(1, 0, 2),
                        config=self.config,
                        meta={
                            "shape": [seq_len, self.num_heads, seq_len],
                            "mask": "lower_triangular",
                            "stage": "post_softmax",
                        },
                    ),
                    "post_projection": _quantize(
                        _sample_tensor(attn_output.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "attn_output"},
                    ),
                    "residual": _quantize(
                        _sample_tensor(residual.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "residual_after_attn"},
                    ),
                }

                attn_matmul_flops = 2 * self.num_heads * seq_len * seq_len * self.head_dim
                _add_flops(layer_flops, "qkv_projection", 2 * seq_len * self.hidden_size * (3 * self.hidden_size))
                _add_flops(layer_flops, "attention_scores", attn_matmul_flops)
                _add_flops(layer_flops, "softmax", self.num_heads * seq_len * seq_len * 5)
                _add_flops(layer_flops, "attention_weighted_sum", attn_matmul_flops)
                _add_flops(layer_flops, "concat", 0)
                _add_flops(layer_flops, "output_projection", 2 * seq_len * self.hidden_size * self.hidden_size)
                _add_flops(layer_flops, "residual_add", seq_len * self.hidden_size)

                ln2_norm, ln2_scaled, ln2_shifted = _layer_norm_states(block.ln_2, residual)
                layer_bucket["ln2"] = {
                    "norm": _quantize(
                        _sample_tensor(ln2_norm.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln2_norm"},
                    ),
                    "scale": _quantize(
                        _sample_tensor(ln2_scaled.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln2_scale"},
                    ),
                    "shift": _quantize(
                        _sample_tensor(ln2_shifted.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "ln2_shift"},
                    ),
                }

                _add_flops(layer_flops, "ln2_norm", seq_len * self.hidden_size * 5)
                _add_flops(layer_flops, "ln2_scale", seq_len * self.hidden_size)
                _add_flops(layer_flops, "ln2_shift", seq_len * self.hidden_size)

                mlp_stride = self.config.mlp_step()
                mlp_up = block.mlp.c_fc(ln2_shifted)
                mlp_act = block.mlp.act(mlp_up)
                mlp_down = block.mlp.c_proj(mlp_act)
                residual = residual + mlp_down

                layer_bucket["mlp"] = {
                    "up": _quantize(
                        _sample_tensor(mlp_up.squeeze(0), mlp_stride),
                        config=self.config,
                        meta={"stride": mlp_stride, "stage": "mlp_up"},
                    ),
                    "activation": _quantize(
                        _sample_tensor(mlp_act.squeeze(0), mlp_stride),
                        config=self.config,
                        meta={"stride": mlp_stride, "stage": "mlp_activation"},
                    ),
                    "down": _quantize(
                        _sample_tensor(mlp_down.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "mlp_down"},
                    ),
                    "residual": _quantize(
                        _sample_tensor(residual.squeeze(0), self.config.stride),
                        config=self.config,
                        meta={"stride": self.config.stride, "stage": "residual_after_mlp"},
                    ),
                }

                _add_flops(layer_flops, "mlp_up", 2 * seq_len * self.hidden_size * self.mlp_dim)
                _add_flops(layer_flops, "mlp_act", seq_len * self.mlp_dim * 6)
                _add_flops(layer_flops, "mlp_down", 2 * seq_len * self.mlp_dim * self.hidden_size)
                _add_flops(layer_flops, "residual_add_2", seq_len * self.hidden_size)

                activations["layers"].append(layer_bucket)
                flops["layers"].append(layer_flops)
                flops["total"] += sum(layer_flops.values())

            final_norm, final_scaled, final_shifted = _layer_norm_states(self.model.transformer.ln_f, residual)
            activations["final_layernorm"] = {
                "norm": _quantize(
                    _sample_tensor(final_norm.squeeze(0), self.config.stride),
                    config=self.config,
                    meta={"stride": self.config.stride, "stage": "final_norm"},
                ),
                "scale": _quantize(
                    _sample_tensor(final_scaled.squeeze(0), self.config.stride),
                    config=self.config,
                    meta={"stride": self.config.stride, "stage": "final_scale"},
                ),
                "shift": _quantize(
                    _sample_tensor(final_shifted.squeeze(0), self.config.stride),
                    config=self.config,
                    meta={"stride": self.config.stride, "stage": "final_shift"},
                ),
            }

            final_flops = flops["final_layernorm"]
            _add_flops(final_flops, "norm", seq_len * self.hidden_size * 5)
            _add_flops(final_flops, "scale", seq_len * self.hidden_size)
            _add_flops(final_flops, "shift", seq_len * self.hidden_size)
            flops["total"] += sum(final_flops.values())

        return {
            "sequence_length": seq_len,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "activations": activations,
            "flops": flops,
        }


def parameter_checkpoints(model: GPT2LMHeadModel) -> Dict[str, object]:
    hidden = model.config.n_embd

    params: Dict[str, object] = {
        "embeddings": {
            "token_lookup": model.transformer.wte.weight.numel(),
            "position_lookup": model.transformer.wpe.weight.numel(),
            "sum": 0,
        },
        "layers": [],
        "final_layernorm": {
            "norm": 0,
            "scale": hidden,
            "shift": hidden,
        },
        "total": 0,
    }

    for block in model.transformer.h:
        layer_params = {
            "ln1": {
                "norm": 0,
                "scale": hidden,
                "shift": hidden,
            },
            "attention": {
                "qkv_projection": block.attn.c_attn.weight.numel() + block.attn.c_attn.bias.numel(),
                "attention_scores": 0,
                "softmax": 0,
                "attention_weighted_sum": 0,
                "concat": 0,
                "output_projection": block.attn.c_proj.weight.numel() + block.attn.c_proj.bias.numel(),
                "residual_add": 0,
            },
            "ln2": {
                "norm": 0,
                "scale": hidden,
                "shift": hidden,
            },
            "mlp": {
                "up": block.mlp.c_fc.weight.numel() + block.mlp.c_fc.bias.numel(),
                "activation": 0,
                "down": block.mlp.c_proj.weight.numel() + block.mlp.c_proj.bias.numel(),
                "residual_add_2": 0,
            },
        }
        params["layers"].append(layer_params)

    params["total"] = sum(
        params["embeddings"][key] for key in params["embeddings"]
    ) + sum(
        sum(section.values())
        for layer in params["layers"]
        for section in layer.values()
        if isinstance(section, dict)
    ) + sum(params["final_layernorm"].values())

    return params


def save_trace(path: str, trace: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

