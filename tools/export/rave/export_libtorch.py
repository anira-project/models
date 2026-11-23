# SPDX-License-Identifier: CC-BY-NC-4.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.12.0", "onnxruntime", "numpy"]
# ///
"""Exports export.py's re-implementations of the encoder, decoder and forward (explicit state, noise
as an input) to TorchScript for LibTorch (traces), and checks each against its ONNX export over
several blocks, the state fed back. torch 2.12.0 is the LibTorch anira's backends ship.

    uv run --no-project export_libtorch.py <model.ts> <prefix>

reads <prefix>_<model>.onnx and writes <prefix>_<model>.pt.
"""

import sys

import torch

from check import compare, load


def main():
    ts_path, prefix = sys.argv[1:3]
    rave, models = load(ts_path)
    for parameter in rave.ts.parameters():  # constants of the trace
        parameter.requires_grad_(False)
    for name, module, make_inputs, _, _ in models:
        torch.jit.trace(module, tuple(make_inputs())).save(f"{prefix}_{name}.pt")
        traced = torch.jit.load(f"{prefix}_{name}.pt").eval()
        compare("LibTorch", name, f"{prefix}_{name}.onnx", make_inputs, lambda inputs: traced(*inputs))


if __name__ == "__main__":
    main()
