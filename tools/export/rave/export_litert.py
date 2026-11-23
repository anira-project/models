# SPDX-License-Identifier: CC-BY-NC-4.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["litert-torch", "onnxruntime", "numpy"]
# ///
"""Exports export.py's re-implementations of the encoder, decoder and forward (explicit state, noise
as an input) to LiteRT (.tflite, also run by anira's TensorFlow Lite engine), and checks each against
its ONNX export over several blocks, the state fed back.

    uv run --no-project export_litert.py <model.ts> <prefix>

reads <prefix>_<model>.onnx and writes <prefix>_<model>.tflite.
"""

import sys

import litert_torch

from check import compare, load


def main():
    ts_path, prefix = sys.argv[1:3]
    rave, models = load(ts_path)
    for name, module, make_inputs, _, _ in models:
        converted = litert_torch.convert(module, tuple(make_inputs()))
        converted.export(f"{prefix}_{name}.tflite")
        compare("LiteRT", name, f"{prefix}_{name}.onnx", make_inputs, lambda inputs: converted(*inputs))


if __name__ == "__main__":
    main()
