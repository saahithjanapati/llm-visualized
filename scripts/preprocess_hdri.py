#!/usr/bin/env python3
"""Downscale an HDRI EXR for lightweight environment lighting use."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source EXR path")
    parser.add_argument("dest", type=Path, help="Destination EXR path")
    parser.add_argument("--width", type=int, default=64, help="Target width")
    parser.add_argument("--height", type=int, default=32, help="Target height")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Unable to read EXR: {args.source}")

    resized = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.dest), resized)
    if not ok:
        raise SystemExit(f"Unable to write EXR: {args.dest}")

    print(f"Wrote {args.dest} at {args.width}x{args.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
