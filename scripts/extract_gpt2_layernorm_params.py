"""Extract GPT-2 LayerNorm scale/shift parameters for visualization.

This script loads the GPT-2 124M checkpoint and writes out the LayerNorm
gamma (scale) and beta (shift) vectors for every transformer block plus
the final layernorm. The output is meant to be consumed by the Three.js
visualization so LayerNorm multiplier/addition vectors can be colored
using real parameter values.
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


def _sample_values(values: List[float], stride: Optional[int]) -> List[float]:
    if stride is None:
        return values
    step = int(stride)
    if step <= 1:
        return values
    return [values[i] for i in range(0, len(values), step)]


def _tensor_to_list(
    tensor: torch.Tensor,
    decimals: Optional[int],
    stride: Optional[int],
) -> List[float]:
    values = tensor.detach().cpu().to(torch.float32).tolist()
    values = _sample_values(values, stride)
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
        if not key.startswith("transformer.h."):
            continue
        parts = key.split(".")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[2])
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def build_payload_from_state(
    state: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    model_name: str,
    decimals: Optional[int],
    stride: Optional[int],
) -> Dict[str, Any]:
    n_layer = config.get("n_layer") or config.get("num_hidden_layers") or _infer_n_layer(state)
    hidden_size = config.get("n_embd") or config.get("hidden_size")
    if f"transformer.h.0.ln_1.weight" in state:
        h_prefix = "transformer.h."
        ln_f_prefix = "transformer.ln_f."
    elif f"h.0.ln_1.weight" in state:
        h_prefix = "h."
        ln_f_prefix = "ln_f."
    else:
        raise KeyError("Unable to locate LayerNorm keys in the model state dict.")
    layers: List[Dict[str, Any]] = []
    for layer_idx in range(n_layer):
        layers.append(
            {
                "ln1": {
                    "scale": _tensor_to_list(state[f"{h_prefix}{layer_idx}.ln_1.weight"], decimals, stride),
                    "shift": _tensor_to_list(state[f"{h_prefix}{layer_idx}.ln_1.bias"], decimals, stride),
                },
                "ln2": {
                    "scale": _tensor_to_list(state[f"{h_prefix}{layer_idx}.ln_2.weight"], decimals, stride),
                    "shift": _tensor_to_list(state[f"{h_prefix}{layer_idx}.ln_2.bias"], decimals, stride),
                },
            }
        )

    payload = {
        "meta": {
            "model": model_name,
            "n_layer": n_layer,
            "hidden_size": hidden_size,
        },
        "layers": layers,
        "final": {
            "scale": _tensor_to_list(state[f"{ln_f_prefix}weight"], decimals, stride),
            "shift": _tensor_to_list(state[f"{ln_f_prefix}bias"], decimals, stride),
        },
    }
    if stride is not None and stride > 1:
        payload["meta"]["param_stride"] = stride
    return payload
    layers: List[Dict[str, Any]] = []
    for block in model.transformer.h:
        layers.append(
            {
                "ln1": {
                    "scale": _tensor_to_list(block.ln_1.weight, decimals),
                    "shift": _tensor_to_list(block.ln_1.bias, decimals),
                },
                "ln2": {
                    "scale": _tensor_to_list(block.ln_2.weight, decimals),
                    "shift": _tensor_to_list(block.ln_2.bias, decimals),
                },
            }
        )

    final_ln = model.transformer.ln_f
    payload = {
        "meta": {
            "model": model_name,
            "n_layer": len(model.transformer.h),
            "hidden_size": model.config.n_embd,
        },
        "layers": layers,
        "final": {
            "scale": _tensor_to_list(final_ln.weight, decimals),
            "shift": _tensor_to_list(final_ln.bias, decimals),
        },
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GPT-2 LayerNorm parameters for visualization."
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model name or local path (default: gpt2).",
    )
    parser.add_argument(
        "--output",
        default="src/data/gpt2_layernorm_params.json",
        help="Output path for the parameters JSON.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=6,
        help="Round parameter values to this many decimals (default: 6).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=64,
        help="Stride for sampling vectors (default: 64; set to 1 to keep full size).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = args.model
    output_path = Path(args.output)
    round_decimals = args.round_decimals if args.round_decimals is not None else None
    stride = args.stride if args.stride is not None else None

    local_paths = _resolve_local_paths(model_name)
    if local_paths:
        config_path, weights_path = local_paths
    else:
        config_path, weights_path = _download_paths(model_name)

    config = json.loads(config_path.read_text())
    state = _load_state(weights_path)

    with torch.no_grad():
        payload = build_payload_from_state(
            state,
            config,
            model_name,
            round_decimals,
            stride,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote LayerNorm params to {output_path}")


if __name__ == "__main__":
    main()
