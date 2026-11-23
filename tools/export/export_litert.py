# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["litert-torch", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports an ONNX model to LiteRT (.tflite, also run by anira's TensorFlow Lite engine) from the
PyTorch module sources.py gives, so that it computes what the ONNX model does, and checks it against
ONNX Runtime over consecutive calls with the state fed back.

    uv run --no-project export_litert.py <model.onnx> <out.tflite> [<description.model.json>]
                                         [--guitar-lstm <GuitarLSTM-dynamic.pt>]
"""

import sys

import litert_torch
import numpy as np
import torch

from check import compare
from sources import load, options


def main():
    onnx_path, out_path = sys.argv[1:3]
    module, input_names, output_names, shapes = load(onnx_path, *options(sys.argv[3:]))
    converted = litert_torch.convert(module, tuple(torch.zeros(*shapes[n]) for n in input_names))
    converted.export(out_path)

    def run(values):
        out = converted(*[torch.from_numpy(v) for v in values])
        return [np.asarray(o) for o in (out if isinstance(out, (list, tuple)) else [out])]

    compare("LiteRT", onnx_path, input_names, output_names, shapes, run)


if __name__ == "__main__":
    main()
