# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.7.0", "onnxruntime", "numpy"]
# ///
"""Exports silero.py's explicit-state 16 kHz model to Core ML (.mlpackage, an ML program in float32,
Apple platforms) with the ONNX model's tensor names, and checks it against the ONNX export. torch
2.7 is the newest coremltools 9.0 is tested with.

    uv run --no-project export_coreml.py <silero_vad.jit> <prefix>

reads <prefix>.onnx and writes <prefix>.mlpackage.
"""

import sys

import coremltools as ct
import torch

from check import TOLERANCE_COREML, compare
from silero import BLOCK, load

INPUTS, OUTPUTS = ["audio_in", "state_in"], ["probability", "state_out"]


def main():
    jit_path, prefix = sys.argv[1:3]
    module = load(jit_path)
    exported = torch.export.export(module, (torch.zeros(1, 1, BLOCK), torch.zeros(1, module.state_size)),
                                   strict=False).run_decompositions({})
    model = ct.convert(exported, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                       minimum_deployment_target=ct.target.macOS13)
    spec = model.get_spec()
    for old, new in zip([i.name for i in spec.description.input], INPUTS):
        ct.utils.rename_feature(spec, old, new)
    for old, new in zip([o.name for o in spec.description.output], OUTPUTS):
        ct.utils.rename_feature(spec, old, new)
    model = ct.models.MLModel(spec, weights_dir=model.weights_dir)
    model.save(f"{prefix}.mlpackage")

    def run(values):
        out = model.predict({n: v.numpy() for n, v in zip(INPUTS, values)})
        return [out[n] for n in OUTPUTS]

    compare("Core ML", f"{prefix}.onnx", module.state_size, run, TOLERANCE_COREML)


if __name__ == "__main__":
    main()
