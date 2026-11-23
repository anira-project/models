# SPDX-License-Identifier: Apache-2.0
"""Exports nemotron.py's streaming step to ExecuTorch 1.3.1 (the runtime anira's backends ship) and checks
it against transformers over a stream with the speaker cache fed back (check.py).

    uv run export_executorch.py <folder> [--coreml] <audio.wav>...

writes <folder>/nemotron_diarization.pte, lowered to XNNPACK (CPU), float32, or with --coreml
nemotron_diarization_coreml.pte, lowered to Core ML (Apple platforms), float32.
"""

import os
import sys
import time

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import compare
from nemotron import Step, example, load


def main():
    folder = sys.argv[1]
    coreml = "--coreml" in sys.argv[2:]
    files = [a for a in sys.argv[2:] if not a.startswith("--")]
    loaded = load()
    partitioner = CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS13)) if coreml else XnnpackPartitioner()
    start = time.perf_counter()
    exported = torch.export.export(Step(loaded[0]).eval(), example(), strict=False)
    edge = to_edge_transform_and_lower(exported, partitioner=[partitioner])
    path = f"{folder}/nemotron_diarization{'_coreml' if coreml else ''}.pte"
    with open(path, "wb") as f:
        f.write(edge.to_executorch().buffer)
    print(f"{path}: {os.path.getsize(path):,} bytes, {time.perf_counter() - start:.0f} s", flush=True)
    method = Runtime.get().load_program(path).load_method("forward")
    method.execute(list(example()))
    start = time.perf_counter()
    for _ in range(10):
        method.execute(list(example()))
    print(f"{(time.perf_counter() - start) * 100:.1f} ms per chunk", flush=True)
    compare("ExecuTorch Core ML" if coreml else "ExecuTorch",
            lambda *x: [t.numpy() for t in method.execute([torch.as_tensor(v) for v in x])], files, loaded,
            tolerance=1e-3)


if __name__ == "__main__":
    main()
