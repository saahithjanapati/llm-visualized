"""Extract sampled GPT-2 learned biases for visualization.

This mirrors the LayerNorm parameter export pattern, but covers the learned
bias vectors used throughout GPT-2:

- attention query / key / value biases
- attention output projection bias
- MLP up projection bias
- MLP down projection bias
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "huggingface_hub is required. Install with `pip install huggingface_hub`."
    ) from exc

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "safetensors is required. Install with `pip install safetensors`."
    ) from exc


def _maybe_round(values: List[float], decimals: Optional[int]) -> List[float]:
    if decimals is None:
        return [float(v) for v in values]
    return [round(float(v), decimals) for v in values]


def _sample_vector(values: List[float], stride: Optional[int]) -> List[float]:
    if stride is None:
        return values
    step = int(stride)
    if step <= 1:
        return values
    return [values[i] for i in range(0, len(values), step)]


def _tensor_to_vector(
    tensor: torch.Tensor,
    decimals: Optional[int],
    stride: Optional[int],
) -> List[float]:
    values = tensor.detach().cpu().to(torch.float32).tolist()
    values = _sample_vector(values, stride)
    return _maybe_round(values, decimals)


def _resolve_local_paths(model_name: str) -> Optional[Tuple[Path, Path]]:
    model_path = Path(model_name)
    if not model_path.exists():
        return None
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {model_path}")
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        weights_path = model_path / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights under {model_path}")
    return config_path, weights_path


def _download_paths(model_name: str) -> Tuple[Path, Path]:
    config_path = Path(hf_hub_download(repo_id=model_name, filename="config.json"))
    weights_name = "model.safetensors"
    try:
        weights_path = Path(hf_hub_download(repo_id=model_name, filename=weights_name))
    except Exception:
        weights_name = "pytorch_model.bin"
        weights_path = Path(hf_hub_download(repo_id=model_name, filename=weights_name))
    return config_path, weights_path


def _load_state(weights_path: Path) -> Dict[str, torch.Tensor]:
    if weights_path.suffix == ".safetensors":
        state = load_safetensors(str(weights_path))
    else:
        state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            return state["state_dict"]
        if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
            return state["model_state_dict"]
    return state


def _infer_n_layer(state: Dict[str, torch.Tensor]) -> int:
    max_idx = -1
    for key in state.keys():
        if not key.startswith("transformer.h.") and not key.startswith("h."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2] if parts[0] == "transformer" else parts[1])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def _resolve_prefixes(state: Dict[str, torch.Tensor]) -> str:
    if "transformer.h.0.attn.c_attn.bias" in state:
        return "transformer.h."
    if "h.0.attn.c_attn.bias" in state:
        return "h."
    raise KeyError("Unable to locate GPT-2 attention/MLP bias keys in the model state dict.")


def build_payload_from_state(
    state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    model_name: str,
    decimals: Optional[int],
    bias_stride: Optional[int],
) -> Dict[str, Any]:
    n_layer = config.get("n_layer") or config.get("num_hidden_layers") or _infer_n_layer(state)
    hidden_size = config.get("n_embd") or config.get("hidden_size")
    intermediate_size = (
        config.get("n_inner")
        or config.get("intermediate_size")
        or (hidden_size * 4 if hidden_size else None)
    )
    h_prefix = _resolve_prefixes(state)

    layers: List[Dict[str, Any]] = []
    for layer_idx in range(n_layer):
        attn_bias = state[f"{h_prefix}{layer_idx}.attn.c_attn.bias"]
        q_bias, k_bias, v_bias = torch.chunk(attn_bias, 3, dim=0)

        attn_out_bias = state[f"{h_prefix}{layer_idx}.attn.c_proj.bias"]
        mlp_up_bias = state[f"{h_prefix}{layer_idx}.mlp.c_fc.bias"]
        mlp_down_bias = state[f"{h_prefix}{layer_idx}.mlp.c_proj.bias"]

        layers.append(
            {
                "attention": {
                    "query": _tensor_to_vector(q_bias, decimals, bias_stride),
                    "key": _tensor_to_vector(k_bias, decimals, bias_stride),
                    "value": _tensor_to_vector(v_bias, decimals, bias_stride),
                    "output": _tensor_to_vector(attn_out_bias, decimals, bias_stride),
                },
                "mlp": {
                    "up": _tensor_to_vector(mlp_up_bias, decimals, bias_stride),
                    "down": _tensor_to_vector(mlp_down_bias, decimals, bias_stride),
                },
            }
        )

    return {
        "meta": {
            "model": model_name,
            "n_layer": n_layer,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "bias_stride": bias_stride,
        },
        "layers": layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sampled GPT-2 learned biases for visualization."
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model name or local path (default: gpt2).",
    )
    parser.add_argument(
        "--output",
        default="src/data/gpt2_bias_params.json",
        help="Output path for the sampled bias JSON.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Round exported floats to this many decimals (default: 4).",
    )
    parser.add_argument(
        "--bias-stride",
        type=int,
        default=64,
        help="Sample one bias value every N dimensions (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_paths = _resolve_local_paths(args.model)
    if resolved_paths is None:
        config_path, weights_path = _download_paths(args.model)
    else:
        config_path, weights_path = resolved_paths

    config = json.loads(config_path.read_text())
    state = _load_state(weights_path)
    payload = build_payload_from_state(
        state=state,
        config=config,
        model_name=args.model,
        decimals=args.round_decimals,
        bias_stride=args.bias_stride,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote sampled bias params to {output_path}")


if __name__ == "__main__":
    main()
