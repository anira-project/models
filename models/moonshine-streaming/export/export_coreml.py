# SPDX-License-Identifier: Apache-2.0
"""Exports moonshine.py's encoder and decoder to Core ML ML programs (.mlpackage, macOS 13 / iOS 16,
float32) with the ONNX export's tensor names, and checks them (check.py).

    uv run export_coreml.py <folder> <variant> <audio.wav>...

writes <folder>/<variant>/coreml/{encoder,decoder}.mlpackage. Token, positions and lengths are int32.
"""

import sys
import time

import coremltools as ct
import numpy as np
import torch

import moonshine as m
from check import compare


def drop_aliases(exported):
    """aten.alias is a no-op that coremltools cannot convert."""
    graph = exported.graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target == torch.ops.aten.alias.default:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
    exported.graph_module.recompile()


def main():
    folder, variant, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    model, tokenizer = m.load(variant)
    modules = {"encoder": m.Encoder(model).eval(), "decoder": m.Decoder(model).eval()}
    converted = {}
    for graph, module in modules.items():
        start = time.perf_counter()
        example = m.examples(model)[graph]
        exported = torch.export.export(module, example, strict=False).run_decompositions({})
        drop_aliases(exported)
        inputs, outputs = m.NAMES[graph]
        converted[graph] = ct.convert(
            exported, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT32,
            inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.float32 if x.is_floating_point() else np.int32)
                    for n, x in zip(inputs, example)],
            outputs=[ct.TensorType(name=n) for n in outputs])
        converted[graph].save(f"{folder}/{variant}/coreml/{graph}.mlpackage")
        print(f"{folder}/{variant}/coreml/{graph}.mlpackage: {time.perf_counter() - start:.0f} s", flush=True)

    def run(graph, *values):
        inputs, outputs = m.NAMES[graph]
        feed = {n: np.asarray(v).astype(np.int32 if np.asarray(v).dtype.kind == "i" else np.float32)
                for n, v in zip(inputs, values)}
        result = converted[graph].predict(feed)
        return [result[n] for n in outputs]

    for graph in modules:
        example = [x.numpy() for x in m.examples(model)[graph]]
        run(graph, *example)
        start = time.perf_counter()
        run(graph, *example)
        print(f"{graph}: {(time.perf_counter() - start) * 1e3:.2f} ms per call", flush=True)
    compare("Core ML", run, variant, files, model, tokenizer, tolerance=1e-2)


if __name__ == "__main__":
    main()
