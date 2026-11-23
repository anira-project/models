# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["litert-torch", "onnxruntime", "numpy"]
# ///
"""Exports silero.py's explicit-state 16 kHz model to LiteRT (.tflite, also run by anira's TensorFlow
Lite engine) and checks it against the ONNX export.

    uv run --no-project export_litert.py <silero_vad.jit> <prefix>

reads <prefix>.onnx and writes <prefix>.tflite.
"""

import sys

import litert_torch
import torch

from check import compare
from silero import BLOCK, load


def main():
    jit_path, prefix = sys.argv[1:3]
    model = load(jit_path)
    converted = litert_torch.convert(model, (torch.zeros(1, 1, BLOCK), torch.zeros(1, model.state_size)))
    converted.export(f"{prefix}.tflite")
    compare("LiteRT", f"{prefix}.onnx", model.state_size, lambda inputs: converted(*inputs))


if __name__ == "__main__":
    main()
