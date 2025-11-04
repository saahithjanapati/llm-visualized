"""Utility helpers for quantizing activation tensors for export.

The goal of this module is to keep the runtime dependency surface small while
providing a predictable interface for writing compact activation dumps that can
be parsed in the browser.  Quantized tensors are represented as dictionaries
that can be directly serialised to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Tuple

import base64

import numpy as np


class QuantizationMode(str, Enum):
    """Supported quantisation modes for activation exports."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"


@dataclass
class QuantizedTensor:
    """Container describing how to decode a quantised tensor.

    Attributes
    ----------
    data:
        Base64 encoded bytes of the quantised tensor laid out in C-order.
    dtype:
        Name of the numpy dtype of the stored payload.
    shape:
        Shape of the tensor.
    scale:
        Multiplicative factor to recover floating values.  ``None`` when the
        tensor is stored losslessly (float16/float32).
    zero_point:
        Additive offset used for the integer data formats.  ``None`` when not
        required.
    meta:
        Optional dictionary with additional metadata (for example, when the
        tensor is padded).  Included verbatim in the JSON export.
    """

    data: str
    dtype: str
    shape: Tuple[int, ...]
    scale: float | None
    zero_point: float | None
    meta: Dict[str, Any] | None = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "data": self.data,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }
        if self.scale is not None:
            payload["scale"] = self.scale
        if self.zero_point is not None:
            payload["zero_point"] = self.zero_point
        if self.meta:
            payload["meta"] = self.meta
        return payload


def _encode_bytes(buffer: np.ndarray) -> str:
    """Encode a numpy array into a base64 string."""

    return base64.b64encode(buffer.tobytes()).decode("ascii")


def quantize_array(
    array: np.ndarray,
    mode: QuantizationMode,
    *,
    meta: Dict[str, Any] | None = None,
) -> QuantizedTensor:
    """Quantise ``array`` according to ``mode`` and return a serialisable blob."""

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if mode == QuantizationMode.FLOAT32:
        buffer = array.astype(np.float32, copy=False)
        return QuantizedTensor(
            data=_encode_bytes(buffer),
            dtype="float32",
            shape=tuple(buffer.shape),
            scale=None,
            zero_point=None,
            meta=meta,
        )

    if mode == QuantizationMode.FLOAT16:
        buffer = array.astype(np.float16)
        return QuantizedTensor(
            data=_encode_bytes(buffer),
            dtype="float16",
            shape=tuple(buffer.shape),
            scale=None,
            zero_point=None,
            meta=meta,
        )

    if mode == QuantizationMode.INT8:
        if array.size == 0:
            scale = 1.0
            quantised = np.zeros(array.shape, dtype=np.int8)
        else:
            max_abs = np.max(np.abs(array))
            scale = float(max_abs / 127.0) if max_abs > 0 else 1.0
            quantised = np.round(array / scale).clip(-128, 127).astype(np.int8)
        return QuantizedTensor(
            data=_encode_bytes(quantised),
            dtype="int8",
            shape=tuple(quantised.shape),
            scale=scale,
            zero_point=0.0,
            meta=meta,
        )

    raise ValueError(f"Unsupported quantization mode: {mode}")


def quantize_sequence(
    tensors: Iterable[np.ndarray],
    mode: QuantizationMode,
) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Quantise a sequence of tensors and return JSON payloads.

    The helper returns a tuple of ``(encoded_list, summary)`` to make it easy to
    append quantised arrays to JSON serialisable dictionaries while also keeping
    track of aggregate statistics (min/max) for debugging.
    """

    encoded: list[Dict[str, Any]] = []
    total_min = float("inf")
    total_max = float("-inf")
    for tensor in tensors:
        tensor = np.asarray(tensor)
        if tensor.size:
            total_min = float(min(total_min, tensor.min()))
            total_max = float(max(total_max, tensor.max()))
        encoded.append(quantize_array(tensor, mode).to_json())

    summary = {
        "count": len(encoded),
        "min": None if total_min == float("inf") else total_min,
        "max": None if total_max == float("-inf") else total_max,
    }
    return encoded, summary

