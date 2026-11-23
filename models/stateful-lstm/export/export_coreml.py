# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.7.0", "onnxruntime", "numpy"]
# ///
"""Exports lstm.py's explicit-state LSTM to Core ML (.mlpackage, an ML program in float32, Apple
platforms) with the ONNX model's tensor names, and checks it against the ONNX export over
consecutive blocks. torch 2.7 is the newest coremltools 9.0 is tested with.

    uv run --no-project export_coreml.py <stateful-lstm-dynamic.pt> <prefix>

reads <prefix>.onnx and writes <prefix>.mlpackage.
"""

import sys

import coremltools as ct
import torch

from check import TOLERANCE_COREML, compare
from lstm import BLOCK, load

INPUTS, OUTPUTS = ["audio_in", "state_in"], ["audio_out", "state_out"]


def main():
    ts_path, prefix = sys.argv[1:3]
    model = load(ts_path)
    inputs = (torch.zeros(BLOCK, 1, 1), torch.zeros(1, model.state_size))
    converted = ct.convert(torch.jit.trace(model, inputs), convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                           minimum_deployment_target=ct.target.macOS13,
                           inputs=[ct.TensorType(name=n, shape=x.shape) for n, x in zip(INPUTS, inputs)],
                           outputs=[ct.TensorType(name=n) for n in OUTPUTS])
    converted.save(f"{prefix}.mlpackage")

    def run(values):
        out = converted.predict({n: v.numpy() for n, v in zip(INPUTS, values)})
        return [out[n] for n in OUTPUTS]

    compare("Core ML", f"{prefix}.onnx", model.state_size, run, TOLERANCE_COREML)


if __name__ == "__main__":
    main()
