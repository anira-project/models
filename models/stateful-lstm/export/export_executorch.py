# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools", "onnxruntime", "numpy"]
# ///
"""Exports lstm.py's explicit-state LSTM to ExecuTorch and checks it against the ONNX export over
consecutive blocks. ExecuTorch 1.3.1 is the runtime anira's backends ship; its Python bindings are
built against torch 2.12.0, hence the pin.

    uv run --no-project export_executorch.py <stateful-lstm-dynamic.pt> <prefix> [--coreml | --vulkan]

reads <prefix>.onnx and writes <prefix>.pte for the CPU (XNNPACK), or with --coreml <prefix>_coreml.pte,
lowered to Core ML (Apple platforms) in float32, or with --vulkan <prefix>_vulkan.pte, lowered to
Vulkan (GPUs with Vulkan), exported but not run: Python's ExecuTorch runtime has no Vulkan delegate.

torch.export unrolls torch.nn.LSTM over the block's 2048 steps, thousands of small operators that the
delegates lower in gigabytes and run slowly. The CPU and Vulkan programs take lstm.py's ScanLstm instead,
which ExecuTorch runs as a loop over one step: the delegate takes the input projection and the output
layer, the loop runs on ExecuTorch's CPU kernels (a 20-unit step is too small for a GPU). The Core ML
program keeps the LSTM as one operator, foundation::lstm_layer, which converts to Core ML's lstm.
"""

import sys
import time

import coremltools as ct
import torch
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _add_mil_lstm, _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from executorch.runtime import Runtime

from check import TOLERANCE, TOLERANCE_COREML, compare
from lstm import BLOCK, ScanLstm, load


@torch.library.custom_op("foundation::lstm_layer", mutates_args=())
def lstm_layer(x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor, w_ih: torch.Tensor, w_hh: torch.Tensor,
               b_ih: torch.Tensor, b_hh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One LSTM layer over a sequence, (time, batch, input) with h0, c0 (1, batch, hidden)."""
    y, h, c = torch._VF.lstm(x, (h0, c0), (w_ih, w_hh, b_ih, b_hh), True, 1, 0.0, False, False, False)
    return y, h, c


@lstm_layer.register_fake
def _(x, h0, c0, w_ih, w_hh, b_ih, b_hh):
    return x.new_empty(x.shape[0], x.shape[1], w_hh.shape[1]), torch.empty_like(h0), torch.empty_like(c0)


# Core ML's lstm for foundation::lstm_layer, by the names the partitioner and the converter look up.
@register_torch_op(torch_alias=["foundation.lstm_layer", "foundation::lstm_layer"])
def foundation_lstm_layer(context, node):
    x, h0, c0, w_ih, w_hh, b_ih, b_hh = _get_inputs(context, node, expected=7)
    y, h, c = _add_mil_lstm(x, h0, c0, [w_ih, w_hh, b_ih, b_hh], True, False, node.name)
    context.add((y, mb.expand_dims(x=h, axes=[0]), mb.expand_dims(x=c, axes=[0])), torch_name=node.name)


class LayerLstm(torch.nn.Module):
    """ExplicitStateLstm with its LSTM as foundation::lstm_layer. Single layer."""

    def __init__(self, model):
        super().__init__()
        assert model.lstm.num_layers == 1
        self.model = model

    def forward(self, audio_in, state_in):
        model, hidden = self.model, self.model.hidden
        lstm = model.lstm
        h = state_in[:, :hidden].reshape(1, 1, hidden)
        c = state_in[:, hidden:].reshape(1, 1, hidden)
        y, h, c = torch.ops.foundation.lstm_layer(audio_in, h, c, lstm.weight_ih_l0, lstm.weight_hh_l0,
                                                  lstm.bias_ih_l0, lstm.bias_hh_l0)
        y = model.lin(y)
        if model.skip:
            y = y + audio_in[:, :, 0:model.skip]
        return y, torch.cat([h.reshape(1, -1), c.reshape(1, -1)], 1)


def main():
    ts_path, prefix = sys.argv[1:3]
    coreml = "--coreml" in sys.argv[3:]
    vulkan = "--vulkan" in sys.argv[3:]
    model = load(ts_path)
    inputs = (torch.zeros(BLOCK, 1, 1), torch.zeros(1, model.state_size))
    if coreml:
        exported = LayerLstm(model).eval()
        partitioner = CoreMLPartitioner(compile_specs=CoreMLBackend.generate_compile_specs(
            compute_precision=ct.precision.FLOAT32, compute_unit=ct.ComputeUnit.CPU_AND_GPU,
            minimum_deployment_target=ct.target.macOS13), lower_full_graph=True)
    elif vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        exported, partitioner = ScanLstm(model).eval(), VulkanPartitioner()
    else:
        exported, partitioner = ScanLstm(model).eval(), XnnpackPartitioner()
    edge = to_edge_transform_and_lower(torch.export.export(exported, inputs, strict=False), partitioner=[partitioner],
                                       compile_config=EdgeCompileConfig(_check_ir_validity=not coreml))
    program = edge.to_executorch()
    path = f"{prefix}_coreml.pte" if coreml else f"{prefix}_vulkan.pte" if vulkan else f"{prefix}.pte"
    with open(path, "wb") as f:
        f.write(program.buffer)
    if vulkan:
        print(vulkan_summary(edge))
        return
    method = Runtime.get().load_program(path).load_method("forward")
    method.execute(list(inputs))
    start = time.perf_counter()
    for _ in range(10):
        method.execute(list(inputs))
    print(f"{(time.perf_counter() - start) * 100:.2f} ms per 2048-sample call")
    compare("ExecuTorch Core ML" if coreml else "ExecuTorch", f"{prefix}.onnx", model.state_size, method.execute,
            TOLERANCE_COREML if coreml else TOLERANCE)


def vulkan_summary(edge):
    """Python's ExecuTorch runtime has no Vulkan delegate: a Vulkan program is counted, not run."""
    graph = edge.exported_program().graph
    calls = [n for n in graph.nodes if n.op == "call_function" and "getitem" not in str(n.target)]
    delegated = sum(1 for n in calls if "executorch_call_delegate" in str(n.target))
    return (f"Vulkan: {delegated} delegated subgraphs, {len(calls) - delegated} operators left to the portable "
            "CPU kernels (the scan loop); not run here")


if __name__ == "__main__":
    main()
