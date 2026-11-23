# SPDX-License-Identifier: CC-BY-NC-4.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.7.0", "onnxruntime", "numpy"]
# ///
"""Exports export.py's re-implementations of the encoder, decoder and forward (explicit state, noise
as an input) to Core ML (.mlpackage, an ML program in float32, Apple platforms) with the ONNX
models' tensor names, and checks each against its ONNX export over several blocks, the state fed
back. torch 2.7 is the newest coremltools 9.0 is tested with.

    uv run --no-project export_coreml.py <model.ts> <prefix>

reads <prefix>_<model>.onnx and writes <prefix>_<model>.mlpackage.
"""

import sys

import coremltools as ct
import torch

from check import TOLERANCE_COREML, compare, load


def main():
    ts_path, prefix = sys.argv[1:3]
    rave, models = load(ts_path)
    for name, module, make_inputs, input_names, output_names in models:
        # torch.export, not a trace: its shapes are static, where a trace keeps shape arithmetic
        # (int casts) that Core ML cannot convert.
        exported = torch.export.export(module, tuple(make_inputs()), strict=False).run_decompositions({})
        model = ct.convert(exported, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                           minimum_deployment_target=ct.target.macOS13)
        spec = model.get_spec()
        for old, new in zip([i.name for i in spec.description.input], input_names):
            ct.utils.rename_feature(spec, old, new)
        for old, new in zip([o.name for o in spec.description.output], output_names):
            ct.utils.rename_feature(spec, old, new)
        model = ct.models.MLModel(spec, weights_dir=model.weights_dir)
        model.save(f"{prefix}_{name}.mlpackage")

        def run(values, model=model, input_names=input_names, output_names=output_names):
            out = model.predict({n: v.numpy() for n, v in zip(input_names, values)})
            return [out[n] for n in output_names]

        compare("Core ML", name, f"{prefix}_{name}.onnx", make_inputs, run, TOLERANCE_COREML)


if __name__ == "__main__":
    main()
