# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.7.0", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports an ONNX model to Core ML (.mlpackage, an ML program in float32, Apple platforms) with its
tensor names, from the PyTorch module sources.py gives, and checks it against ONNX Runtime over
consecutive calls with the state fed back. torch 2.7 is the newest coremltools 9.0 is tested with.
A torch.export program converts, or for GuitarLSTM a trace: coremltools converts an LSTM from
TorchScript only.

    uv run --no-project export_coreml.py <model.onnx> <out.mlpackage> [<description.model.json>]
                                         [--guitar-lstm <GuitarLSTM-dynamic.pt>]
"""

import sys

import coremltools as ct
import torch

from check import TOLERANCE_COREML, compare
from sources import load, options


def main():
    onnx_path, out_path = sys.argv[1:3]
    description, guitar_lstm = options(sys.argv[3:])
    module, input_names, output_names, shapes = load(onnx_path, description, guitar_lstm)
    example = tuple(torch.zeros(*shapes[n]) for n in input_names)
    if guitar_lstm:
        model = ct.convert(torch.jit.trace(module, example), convert_to="mlprogram",
                           compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS13,
                           inputs=[ct.TensorType(shape=x.shape) for x in example])
    else:
        exported = torch.export.export(module, example).run_decompositions({})
        model = ct.convert(exported, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                           minimum_deployment_target=ct.target.macOS13)
    spec = model.get_spec()
    for old, new in zip([i.name for i in spec.description.input], input_names):
        ct.utils.rename_feature(spec, old, new)
    for old, new in zip([o.name for o in spec.description.output], output_names):
        ct.utils.rename_feature(spec, old, new)
    model = ct.models.MLModel(spec, weights_dir=model.weights_dir)
    model.save(out_path)

    def run(values):
        out = model.predict(dict(zip(input_names, values)))
        return [out[n] for n in output_names]

    compare("Core ML", onnx_path, input_names, output_names, shapes, run, TOLERANCE_COREML)


if __name__ == "__main__":
    main()
