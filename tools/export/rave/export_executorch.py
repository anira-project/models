# SPDX-License-Identifier: CC-BY-NC-4.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools", "onnxruntime", "numpy"]
# ///
"""Exports export.py's re-implementations of the encoder, decoder and forward (explicit state, noise
as an input) to ExecuTorch, and checks each against its ONNX export over several blocks, the state
fed back. ExecuTorch 1.3.1 is the runtime anira's backends ship; its Python bindings are built
against torch 2.12.0, hence the pin.

    uv run --no-project export_executorch.py <model.ts> <prefix> [--coreml | --vulkan]

reads <prefix>_<model>.onnx and writes <prefix>_<model>.pte, lowered to XNNPACK (every CPU
platform), or with --coreml <prefix>_<model>_coreml.pte, lowered to Core ML (Apple platforms) in
float32 (the Neural Engine computes in float16 and is left out), or with --vulkan
<prefix>_<model>_vulkan.pte, lowered to Vulkan (GPUs with Vulkan), exported but not run: Python's
ExecuTorch runtime has no Vulkan delegate.
"""

import sys

import coremltools as ct
import torch
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import TOLERANCE, TOLERANCE_COREML, compare, load


def partitioner(coreml, vulkan=False):
    if vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return VulkanPartitioner()
    if not coreml:
        return XnnpackPartitioner()
    return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.CPU_AND_GPU))


def main():
    ts_path, prefix = sys.argv[1:3]
    coreml = "--coreml" in sys.argv[3:]
    vulkan = "--vulkan" in sys.argv[3:]
    suffix = "_coreml" if coreml else "_vulkan" if vulkan else ""
    rave, models = load(ts_path)
    for name, module, make_inputs, _, _ in models:
        exported = torch.export.export(module, tuple(make_inputs()), strict=False)
        edge = to_edge_transform_and_lower(exported, partitioner=[partitioner(coreml, vulkan)])
        program = edge.to_executorch()
        path = f"{prefix}_{name}{suffix}.pte"
        with open(path, "wb") as f:
            f.write(program.buffer)
        if vulkan:
            print(f"{name}: {vulkan_summary(edge)}", flush=True)
            continue
        method = Runtime.get().load_program(path).load_method("forward")
        compare("ExecuTorch Core ML" if coreml else "ExecuTorch", name, f"{prefix}_{name}.onnx", make_inputs,
                lambda inputs: method.execute(inputs), TOLERANCE_COREML if coreml else TOLERANCE)


def vulkan_summary(edge):
    """Python's ExecuTorch runtime has no Vulkan delegate: a Vulkan program is counted, not run."""
    graph = edge.exported_program().graph
    calls = [n for n in graph.nodes if n.op == "call_function" and "getitem" not in str(n.target)]
    delegated = sum(1 for n in calls if "executorch_call_delegate" in str(n.target))
    return (f"Vulkan: {delegated} delegated subgraphs, {len(calls) - delegated} operators left to the portable "
            "CPU kernels; not run here")


if __name__ == "__main__":
    main()
