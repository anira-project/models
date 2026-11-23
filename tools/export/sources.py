# SPDX-License-Identifier: Apache-2.0
"""The PyTorch module an exporter takes for an ONNX model, with the ONNX model's input names and
static shapes: onnx_to_torch.py's conversion, or for GuitarLSTM (an LSTM, which onnx2torch does not
convert) its rebuild from the TorchScript model. A description gives the input shapes where the ONNX
model's are dynamic (matched by position, as anira binds tensors whose names differ)."""

import json

import onnxruntime
import torch
import torch.nn.functional as F

from onnx_to_torch import torch_module


class GuitarLstm(torch.nn.Module):
    """GuitarLSTM (libtorch/GuitarLSTM-dynamic.pt): pad, two strided convolutions, an LSTM from a zero
    state over each window, the last step through a linear layer; stateless between calls."""

    def __init__(self, ts):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(ts.conv1.weight.shape[1], ts.conv1.weight.shape[0], ts.conv1.weight.shape[2], stride=12)
        self.conv2 = torch.nn.Conv1d(ts.conv2.weight.shape[1], ts.conv2.weight.shape[0], ts.conv2.weight.shape[2], stride=12)
        self.lstm = torch.nn.LSTM(ts.lstm.weight_ih_l0.shape[1], ts.lstm.weight_hh_l0.shape[1], batch_first=True)
        self.linear = torch.nn.Linear(ts.linear.weight.shape[1], ts.linear.weight.shape[0])
        for name in ("conv1", "conv2", "lstm", "linear"):
            getattr(self, name).load_state_dict({k: v.detach().clone() for k, v in getattr(ts, name).state_dict().items()})

    def forward(self, x):
        x = self.conv2(F.pad(self.conv1(F.pad(x, (12, 12))), (12, 12))).permute(0, 2, 1)
        return self.linear(self.lstm(x)[0][:, -1, :])


def load(onnx_path, description=None, guitar_lstm=None):
    """(module, input names, output names, shapes by input name)."""
    torch.set_grad_enabled(False)
    session = onnxruntime.InferenceSession(onnx_path)
    input_names = [v.name for v in session.get_inputs()]
    output_names = [v.name for v in session.get_outputs()]
    overrides = {}
    if description:
        for tensor, name in zip(json.load(open(description))["inputs"], input_names):
            overrides[name] = [n for _, n in tensor["axes"]]
    if guitar_lstm:
        module, shapes = GuitarLstm(torch.jit.load(guitar_lstm).eval()).eval(), {}
    else:
        module, shapes = torch_module(onnx_path, overrides)
    shapes.update(overrides)
    return module, input_names, output_names, shapes


def options(args):
    """The description and --guitar-lstm <pt> among the command line's optional arguments."""
    description = next((a for a in args if a.endswith(".model.json")), None)
    guitar_lstm = args[args.index("--guitar-lstm") + 1] if "--guitar-lstm" in args else None
    return description, guitar_lstm
