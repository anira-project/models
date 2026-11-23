# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.12.0", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports an ONNX model to TorchScript for LibTorch (a trace of the PyTorch module sources.py gives,
with the ONNX model's state as tensors) and checks it against ONNX Runtime over consecutive calls with
the state fed back. torch 2.12.0 is the LibTorch anira's backends ship.

    uv run --no-project export_libtorch.py <model.onnx> <out.pt> [<description.model.json>]
                                           [--guitar-lstm <GuitarLSTM-dynamic.pt>]
"""

import sys

import torch

from check import compare
from sources import load, options


def main():
    onnx_path, out_path = sys.argv[1:3]
    module, input_names, output_names, shapes = load(onnx_path, *options(sys.argv[3:]))
    torch.jit.trace(module, tuple(torch.zeros(*shapes[n]) for n in input_names)).save(out_path)
    traced = torch.jit.load(out_path).eval()

    def run(values):
        with torch.no_grad():
            out = traced(*[torch.from_numpy(v) for v in values])
        return [o.numpy() for o in (out if isinstance(out, (list, tuple)) else [out])]

    compare("LibTorch", onnx_path, input_names, output_names, shapes, run)


if __name__ == "__main__":
    main()
