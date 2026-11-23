# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools", "onnx2torch", "onnxsim", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports an ONNX model to ExecuTorch from the PyTorch module sources.py gives, lowered to XNNPACK
(every CPU platform), Core ML (--coreml, Apple platforms, float32) or Vulkan (--vulkan, GPUs with
Vulkan), and checks it against ONNX Runtime over consecutive calls with the state fed back. Python's
ExecuTorch runtime has no Vulkan delegate: a Vulkan program is exported and its delegated part
counted, not run. ExecuTorch 1.3.1 is the runtime anira's backends ship; its Python bindings are
built against torch 2.12.0, hence the pin.

    uv run --no-project export_executorch.py <model.onnx> <out.pte> [<description.model.json>]
                                             [--guitar-lstm <GuitarLSTM-dynamic.pt>] [--coreml | --vulkan]
"""

import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import TOLERANCE, TOLERANCE_COREML, compare
from sources import load, options


def partitioner(target):
    if target == "coreml":
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        return CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.CPU_AND_GPU))
    if target == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return VulkanPartitioner()
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    return XnnpackPartitioner()


def main():
    onnx_path, out_path = sys.argv[1:3]
    args = sys.argv[3:]
    target = "coreml" if "--coreml" in args else "vulkan" if "--vulkan" in args else "xnnpack"
    module, input_names, output_names, shapes = load(onnx_path, *options(args))
    example = tuple(torch.zeros(*shapes[n]) for n in input_names)
    edge = to_edge_transform_and_lower(torch.export.export(module, example, strict=False), partitioner=[partitioner(target)])
    program = edge.to_executorch()
    with open(out_path, "wb") as f:
        f.write(program.buffer)
    if target == "vulkan":
        graph = edge.exported_program().graph
        delegated = sum(1 for n in graph.nodes if n.op == "call_function" and "executorch_call_delegate" in str(n.target))
        rest = sum(1 for n in graph.nodes if n.op == "call_function" and "executorch_call_delegate" not in str(n.target)
                   and "getitem" not in str(n.target))
        print(f"Vulkan: {delegated} delegated subgraphs, {rest} operators left to the portable CPU kernels; not run here")
        return
    method = Runtime.get().load_program(out_path).load_method("forward")
    compare("ExecuTorch Core ML" if target == "coreml" else "ExecuTorch", onnx_path, input_names, output_names, shapes,
            lambda values: method.execute([torch.from_numpy(v) for v in values]),
            TOLERANCE_COREML if target == "coreml" else TOLERANCE)


if __name__ == "__main__":
    main()
