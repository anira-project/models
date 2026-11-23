# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools", "onnxruntime", "numpy"]
# ///
"""Exports silero.py's explicit-state 16 kHz model to ExecuTorch and checks it against the ONNX
export. ExecuTorch 1.3.1 is the runtime anira's backends ship; its
Python bindings are built against torch 2.12.0, hence the pin.

    uv run --no-project export_executorch.py <silero_vad.jit> <prefix> [--coreml | --vulkan]

reads <prefix>.onnx and writes <prefix>.pte, lowered to XNNPACK (every CPU platform), or with
--coreml <prefix>_coreml.pte, lowered to Core ML (Apple platforms) in float32, or with --vulkan
<prefix>_vulkan.pte, lowered to Vulkan (GPUs with Vulkan), exported but not run: Python's ExecuTorch
runtime has no Vulkan delegate.
"""

import sys

import torch
import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import TOLERANCE, TOLERANCE_COREML, compare
from silero import BLOCK, load


def main():
    jit_path, prefix = sys.argv[1:3]
    coreml = "--coreml" in sys.argv[3:]
    vulkan = "--vulkan" in sys.argv[3:]
    model = load(jit_path)
    inputs = (torch.zeros(1, 1, BLOCK), torch.zeros(1, model.state_size))
    edge = to_edge_transform_and_lower(torch.export.export(model, inputs, strict=False),
                                       partitioner=[partitioner(coreml, vulkan)])
    program = edge.to_executorch()
    path = f"{prefix}_coreml.pte" if coreml else f"{prefix}_vulkan.pte" if vulkan else f"{prefix}.pte"
    with open(path, "wb") as f:
        f.write(program.buffer)
    if vulkan:
        print(vulkan_summary(edge))
        return
    method = Runtime.get().load_program(path).load_method("forward")
    compare("ExecuTorch Core ML" if coreml else "ExecuTorch", f"{prefix}.onnx", model.state_size, method.execute,
            TOLERANCE_COREML if coreml else TOLERANCE)


def partitioner(coreml, vulkan=False):
    if vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return VulkanPartitioner()
    if not coreml:
        return XnnpackPartitioner()
    return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.CPU_AND_GPU))


def vulkan_summary(edge):
    """Python's ExecuTorch runtime has no Vulkan delegate: a Vulkan program is counted, not run."""
    graph = edge.exported_program().graph
    calls = [n for n in graph.nodes if n.op == "call_function" and "getitem" not in str(n.target)]
    delegated = sum(1 for n in calls if "executorch_call_delegate" in str(n.target))
    return (f"Vulkan: {delegated} delegated subgraphs, {len(calls) - delegated} operators left to the portable "
            "CPU kernels; not run here")


if __name__ == "__main__":
    main()
