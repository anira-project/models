# SPDX-License-Identifier: Apache-2.0
"""Exports parakeet.py's three parts to Core ML ML programs (.mlpackage, macOS 13 / iOS 16) in the 31 s
window with the ONNX export's tensor names, and checks them against NeMo (check.py).

    uv run export_coreml.py <folder> <audio.wav>...

writes <folder>/{preprocessor,encoder,decoder_joint}.mlpackage: the preprocessor and decoder_joint in
float32, the encoder in float16 (CPU, GPU and Neural Engine). Lengths are int32: Core ML has no int64.
"""

import sys
import time

import coremltools as ct
import numpy as np
import torch

from check import compare
from parakeet import NAMES, examples, load, parts


def drop_aliases(exported):
    """aten.alias is a no-op that coremltools cannot convert."""
    graph = exported.graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    exported.graph_module.recompile()


def main():
    folder, files = sys.argv[1], sys.argv[2:]
    model = load()
    converted = {}
    for part, module in parts(model).items():
        start = time.perf_counter()
        example = examples()[part]
        if part == "decoder_joint":
            # coremltools' torch.export frontend has no LSTM (lstm.input); its TorchScript frontend has.
            exported = torch.jit.trace(module, example)
        else:
            # The preprocessor to core ATen (coremltools lacks where.ScalarOther of NeMo's normalisation).
            exported = torch.export.export(module, example, strict=False)
            exported = exported.run_decompositions() if part == "preprocessor" else exported.run_decompositions({})
            drop_aliases(exported)
        inputs, outputs = NAMES[part]
        converted[part] = ct.convert(
            exported, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16 if part == "encoder" else ct.precision.FLOAT32,
            inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.float32 if x.is_floating_point() else np.int32)
                    for n, x in zip(inputs, example)],
            outputs=[ct.TensorType(name=n) for n in outputs])
        converted[part].save(f"{folder}/{part}.mlpackage")
        print(f"{folder}/{part}.mlpackage: {time.perf_counter() - start:.0f} s", flush=True)

    def runner(part):
        inputs, outputs = NAMES[part]

        def run(*values):
            feed = {n: np.asarray(v).astype(np.int32 if np.asarray(v).dtype.kind == "i" else np.float32)
                    for n, v in zip(inputs, values)}
            result = converted[part].predict(feed)
            return [result[n] for n in outputs]

        return run

    run = {part: runner(part) for part in converted}
    for part in converted:
        example = [x.numpy() for x in examples()[part]]
        run[part](*example)
        start = time.perf_counter()
        run[part](*example)
        print(f"{part}: {(time.perf_counter() - start) * 1e3:.1f} ms per call", flush=True)
    compare("Core ML", run, files, model, tolerance=2e-2)


if __name__ == "__main__":
    main()
