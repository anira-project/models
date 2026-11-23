# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports the four hand tracking models to ExecuTorch (.pte, XNNPACK: every CPU platform).

The models exist as TFLite (MediaPipe) and our ONNX conversions only; ExecuTorch takes a PyTorch
module, which onnx_to_torch.py makes from each ONNX model. It is exported with torch.export, lowered,
and checked against ONNX Runtime on the same random input. ExecuTorch 1.3.1 is the runtime
anira's backends ship; its Python bindings are built against torch 2.12.0, hence the pin.

    uv run export_executorch.py <weights dir laid out like the HF repo> [--coreml | --vulkan]

writes <weights dir>/executorch/<model>.pte, lowered to XNNPACK (every CPU platform), or with
--coreml <model>_coreml.pte, lowered to Core ML (Apple platforms) in float16 for the Neural Engine,
checked against ONNX Runtime relative to each output's peak, or with --vulkan <model>_vulkan.pte,
lowered to Vulkan (GPUs with Vulkan), exported but not run: Python's ExecuTorch runtime has no
Vulkan delegate.
"""

import os
import sys

import numpy as np
import onnxruntime
import torch
import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
# onnx_to_torch.py is the generic converter in tools/export.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools", "export"))
from onnx_to_torch import MODELS, torch_module

TOLERANCE = 1e-3
TOLERANCE_COREML = 1e-2  # float16, relative to the output's peak


def export(weights, name, coreml, vulkan=False):
    onnx_path = os.path.join(weights, "onnx", f"{name}.onnx")
    module, shapes = torch_module(onnx_path)
    generator = torch.Generator().manual_seed(0)
    inputs = tuple(torch.rand(*dims, generator=generator) for dims in shapes.values())

    edge = to_edge_transform_and_lower(torch.export.export(module, inputs),
                                       partitioner=[partitioner(coreml, vulkan)])
    program = edge.to_executorch()
    os.makedirs(os.path.join(weights, "executorch"), exist_ok=True)
    suffix = "_coreml" if coreml else "_vulkan" if vulkan else ""
    pte_path = os.path.join(weights, "executorch", f"{name}{suffix}.pte")
    with open(pte_path, "wb") as f:
        f.write(program.buffer)
    if vulkan:
        print(f"{name}: {vulkan_summary(edge)}")
        return

    session = onnxruntime.InferenceSession(onnx_path)
    expected = session.run(None, {v.name: x.numpy() for v, x in zip(session.get_inputs(), inputs)})
    method = Runtime.get().load_program(pte_path).load_method("forward")
    actual = method.execute(list(inputs))
    if coreml:
        worst = max(float(np.max(np.abs(e - a.numpy()))) / (1.0 + float(np.max(np.abs(e))))
                    for e, a in zip(expected, actual))
    else:
        worst = max(float(np.max(np.abs(e - a.numpy()))) for e, a in zip(expected, actual))
    engine = "ExecuTorch Core ML" if coreml else "ExecuTorch"
    print(f"{name}: {len(program.buffer)} bytes, max |{engine} - ONNX Runtime| = {worst:.2e}")
    if worst > (TOLERANCE_COREML if coreml else TOLERANCE):
        sys.exit(f"{name}: the export differs from ONNX Runtime by {worst:.2e}")


def partitioner(coreml, vulkan=False):
    if vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return VulkanPartitioner()
    if not coreml:
        return XnnpackPartitioner()
    return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
        compute_precision=ct.precision.FLOAT16, compute_unit=ct.ComputeUnit.ALL))


def main():
    coreml = "--coreml" in sys.argv[2:]
    vulkan = "--vulkan" in sys.argv[2:]
    for name in MODELS:
        export(sys.argv[1], name, coreml, vulkan)


def vulkan_summary(edge):
    """Python's ExecuTorch runtime has no Vulkan delegate: a Vulkan program is counted, not run."""
    graph = edge.exported_program().graph
    calls = [n for n in graph.nodes if n.op == "call_function" and "getitem" not in str(n.target)]
    delegated = sum(1 for n in calls if "executorch_call_delegate" in str(n.target))
    return (f"Vulkan: {delegated} delegated subgraphs, {len(calls) - delegated} operators left to the portable "
            "CPU kernels; not run here")


if __name__ == "__main__":
    main()
