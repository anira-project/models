# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.7.0", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports the four hand tracking models to Core ML (.mlpackage, an ML program in float16 for the
Neural Engine, Apple platforms) with the ONNX models' tensor names, from the PyTorch modules
onnx_to_torch.py makes, and checks each against ONNX Runtime on the same random input, relative to
each output's peak. torch 2.7 is the newest coremltools 9.0 is tested with.

    uv run export_coreml.py <weights dir laid out like the HF repo>

writes <weights dir>/coreml/<model>.mlpackage.
"""

import os
import sys

import coremltools as ct
import numpy as np
import onnxruntime
import torch

# onnx_to_torch.py is the generic converter in tools/export.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "export"))
from onnx_to_torch import MODELS, torch_module

TOLERANCE = 1e-2  # float16, relative to the output's peak


def export(weights, name):
    onnx_path = os.path.join(weights, "onnx", f"{name}.onnx")
    module, shapes = torch_module(onnx_path)
    generator = torch.Generator().manual_seed(0)
    inputs = tuple(torch.rand(*dims, generator=generator) for dims in shapes.values())
    session = onnxruntime.InferenceSession(onnx_path)
    input_names = [v.name for v in session.get_inputs()]
    output_names = [v.name for v in session.get_outputs()]

    exported = torch.export.export(module, inputs).run_decompositions({})
    model = ct.convert(exported, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.macOS13)
    spec = model.get_spec()
    for old, new in zip([i.name for i in spec.description.input], input_names):
        ct.utils.rename_feature(spec, old, new)
    for old, new in zip([o.name for o in spec.description.output], output_names):
        ct.utils.rename_feature(spec, old, new)
    model = ct.models.MLModel(spec, weights_dir=model.weights_dir)
    os.makedirs(os.path.join(weights, "coreml"), exist_ok=True)
    model.save(os.path.join(weights, "coreml", f"{name}.mlpackage"))

    expected = session.run(None, {n: x.numpy() for n, x in zip(input_names, inputs)})
    actual = model.predict({n: x.numpy() for n, x in zip(input_names, inputs)})
    worst = max(float(np.max(np.abs(e - actual[n]))) / (1.0 + float(np.max(np.abs(e))))
                for n, e in zip(output_names, expected))
    print(f"{name}: max |Core ML - ONNX Runtime| = {worst:.2e}")
    if worst > TOLERANCE:
        sys.exit(f"{name}: the export differs from ONNX Runtime by {worst:.2e}")


def main():
    for name in MODELS:
        export(sys.argv[1], name)


if __name__ == "__main__":
    main()
