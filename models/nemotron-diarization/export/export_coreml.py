# SPDX-License-Identifier: Apache-2.0
"""Exports nemotron.py's streaming step to a Core ML ML program (.mlpackage, macOS 13 / iOS 16, float32)
with the ONNX export's tensor names, and checks it against transformers over a stream with the speaker
cache fed back (check.py).

    uv run export_coreml.py <folder> <audio.wav>...

writes <folder>/nemotron_diarization.mlpackage. attention_mask is int32: Core ML has no int64.
"""

import sys
import time

import coremltools as ct
import numpy as np
import torch

from check import compare
from nemotron import INPUTS, OUTPUTS, Step, example, load


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
    loaded = load()
    start = time.perf_counter()
    exported = torch.export.export(Step(loaded[0]).eval(), example(), strict=False).run_decompositions()
    drop_aliases(exported)
    model = ct.convert(exported, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13,
                       compute_precision=ct.precision.FLOAT32,
                       inputs=[ct.TensorType(name=n, shape=x.shape, dtype=np.float32 if x.is_floating_point() else np.int32)
                               for n, x in zip(INPUTS, example())],
                       outputs=[ct.TensorType(name=n) for n in OUTPUTS])
    path = f"{folder}/nemotron_diarization.mlpackage"
    model.save(path)
    print(f"{path}: {time.perf_counter() - start:.0f} s", flush=True)

    def run(*values):
        feed = {n: np.asarray(v).astype(np.int32 if np.asarray(v).dtype.kind == "i" else np.float32)
                for n, v in zip(INPUTS, values)}
        result = model.predict(feed)
        return [result[n] for n in OUTPUTS]

    run(*(x.numpy() for x in example()))
    start = time.perf_counter()
    for _ in range(10):
        run(*(x.numpy() for x in example()))
    print(f"{(time.perf_counter() - start) * 100:.1f} ms per chunk", flush=True)
    compare("Core ML", run, files, loaded, tolerance=1e-3)


if __name__ == "__main__":
    main()
