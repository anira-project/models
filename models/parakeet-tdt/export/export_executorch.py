# SPDX-License-Identifier: Apache-2.0
"""Exports parakeet.py's three parts to ExecuTorch 1.3.1 (the runtime anira's backends ship) in the 31 s
window and checks them against NeMo (check.py).

    uv run export_executorch.py <folder> [--coreml] <audio.wav>...

writes <folder>/{preprocessor,encoder,decoder_joint}.pte, lowered to XNNPACK (CPU; the preprocessor's
STFT stays on the portable kernels), float32; with --coreml *_coreml.pte, lowered to Core ML (Apple
platforms): the preprocessor and decoder_joint in float32, the encoder in float16.
"""

import os
import sys
import time

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import compare
from parakeet import examples, load, parts


def partitioner(part, coreml):
    if not coreml:
        return XnnpackPartitioner()
    precision = ct.precision.FLOAT16 if part == "encoder" else ct.precision.FLOAT32
    return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=precision, compute_unit=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.macOS13))


def main():
    folder = sys.argv[1]
    coreml = "--coreml" in sys.argv[2:]
    files = [a for a in sys.argv[2:] if not a.startswith("--")]
    model = load()
    methods = {}
    for part, module in parts(model).items():
        start = time.perf_counter()
        exported = torch.export.export(module, examples()[part], strict=False)
        # The preprocessor's STFT has complex intermediates, which the IR check refuses (as in ExecuTorch's
        # own Parakeet example).
        edge = to_edge_transform_and_lower(exported, partitioner=[partitioner(part, coreml)],
                                           compile_config=EdgeCompileConfig(_check_ir_validity=False))
        path = f"{folder}/{part}{'_coreml' if coreml else ''}.pte"
        with open(path, "wb") as f:
            f.write(edge.to_executorch().buffer)
        print(f"{path}: {os.path.getsize(path):,} bytes, {time.perf_counter() - start:.0f} s", flush=True)
        methods[part] = Runtime.get().load_program(path).load_method("forward")
    for part, method in methods.items():
        method.execute([torch.as_tensor(x) for x in examples()[part]])
        start = time.perf_counter()
        method.execute([torch.as_tensor(x) for x in examples()[part]])
        print(f"{part}: {(time.perf_counter() - start) * 1e3:.1f} ms per call", flush=True)
    run = {part: (lambda m: lambda *x: [t.numpy() for t in m.execute([torch.as_tensor(v) for v in x])])(m)
           for part, m in methods.items()}
    compare("ExecuTorch Core ML" if coreml else "ExecuTorch", run, files, model, tolerance=2e-2 if coreml else 1e-3)


if __name__ == "__main__":
    main()
