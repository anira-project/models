# SPDX-License-Identifier: Apache-2.0
"""Exports moonshine.py's encoder and decoder to ExecuTorch 1.3.1 (the runtime anira's backends ship) and
checks them (check.py).

    uv run export_executorch.py <folder> <variant> [--coreml] <audio.wav>...

writes <folder>/<variant>/executorch/{encoder,decoder}.pte, lowered to XNNPACK (CPU), float32; with
--coreml {encoder,decoder}_coreml.pte, lowered to Core ML (Apple platforms), float32.
"""

import os
import sys
import time

import coremltools as ct
import numpy as np
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

import moonshine as m
from check import compare


def partitioner(coreml):
    if not coreml:
        return XnnpackPartitioner()
    return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13))


def main():
    folder, variant = sys.argv[1], sys.argv[2]
    coreml = "--coreml" in sys.argv[3:]
    files = [a for a in sys.argv[3:] if not a.startswith("--")]
    model, tokenizer = m.load(variant)
    os.makedirs(f"{folder}/{variant}/executorch", exist_ok=True)
    modules = {"encoder": m.Encoder(model).eval(), "decoder": m.Decoder(model).eval()}
    methods = {}
    for graph, module in modules.items():
        start = time.perf_counter()
        exported = torch.export.export(module, m.examples(model)[graph], strict=False)
        edge = to_edge_transform_and_lower(exported, partitioner=[partitioner(coreml)],
                                           compile_config=EdgeCompileConfig(_check_ir_validity=False))
        path = f"{folder}/{variant}/executorch/{graph}{'_coreml' if coreml else ''}.pte"
        with open(path, "wb") as f:
            f.write(edge.to_executorch().buffer)
        print(f"{path}: {os.path.getsize(path):,} bytes, {time.perf_counter() - start:.0f} s", flush=True)
        methods[graph] = Runtime.get().load_program(path).load_method("forward")

    def run(graph, *values):
        return [t.numpy() for t in methods[graph].execute([torch.from_numpy(np.ascontiguousarray(v)) for v in values])]

    for graph in modules:
        example = [x.numpy() for x in m.examples(model)[graph]]
        run(graph, *example)
        start = time.perf_counter()
        run(graph, *example)
        print(f"{graph}: {(time.perf_counter() - start) * 1e3:.2f} ms per call", flush=True)
    compare("ExecuTorch Core ML" if coreml else "ExecuTorch", run, variant, files, model, tokenizer,
            tolerance=1e-2 if coreml else 1e-3)


if __name__ == "__main__":
    main()
