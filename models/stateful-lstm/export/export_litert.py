# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["litert-torch", "onnxruntime", "numpy"]
# ///
"""Exports lstm.py's explicit-state LSTM to LiteRT (.tflite, also run by anira's TensorFlow Lite
engine) and checks it against the ONNX export over consecutive blocks.

    uv run --no-project export_litert.py <stateful-lstm-dynamic.pt> <prefix>

reads <prefix>.onnx and writes <prefix>.tflite.
"""

import sys

import litert_torch
import torch

from check import compare
from lstm import BLOCK, load


def main():
    ts_path, prefix = sys.argv[1:3]
    model = load(ts_path)
    converted = litert_torch.convert(model, (torch.zeros(BLOCK, 1, 1), torch.zeros(1, model.state_size)))
    converted.export(f"{prefix}.tflite")
    compare("LiteRT", f"{prefix}.onnx", model.state_size, lambda inputs: converted(*inputs))


if __name__ == "__main__":
    main()
