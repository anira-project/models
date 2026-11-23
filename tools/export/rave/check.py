# SPDX-License-Identifier: CC-BY-NC-4.0
"""Shared by export_litert.py and export_executorch.py: loads the re-implementation, and compares an
export with the ONNX one over several blocks with the same inputs, each feeding its own state back."""

import sys

import numpy as np
import onnxruntime
import torch

from export import StreamingRave, models

BLOCKS = 8
TOLERANCE = 1e-4
TOLERANCE_COREML = 2e-3  # Core ML on the GPU, float32


def load(ts_path):
    torch.set_grad_enabled(False)
    rave = StreamingRave(torch.jit.load(ts_path).eval()).eval()
    # The TorchScript model is only read from (weights, constants): out of the module tree, its
    # tensors are constants of the exported program rather than parameters of a ScriptModule, which
    # torch.export's verifier refuses.
    object.__setattr__(rave, "ts", rave._modules.pop("ts"))  # pylint: disable=protected-access
    return rave, models(rave)


def compare(engine, name, onnx_path, make_inputs, run, tolerance=TOLERANCE):
    """The largest difference relative to 1 + the output's peak (latents reach tens)."""
    session = onnxruntime.InferenceSession(onnx_path)
    names = [i.name for i in session.get_inputs()]
    torch.manual_seed(0)
    onnx_state = other_state = make_inputs()[1]
    worst = 0.0
    for _ in range(BLOCKS):
        fresh = make_inputs()
        feed = {n: t.numpy() for n, t in zip(names, fresh)}
        feed["state_in"] = onnx_state.numpy()
        out_onnx, onnx_state = session.run(None, feed)
        onnx_state = torch.from_numpy(onnx_state)
        out_other, other_state = run([fresh[0], other_state, *fresh[2:]])
        other_state = torch.as_tensor(np.asarray(other_state))
        out_other = np.asarray(out_other)
        worst = max(worst, float(np.abs(out_onnx - out_other).max()) / (1.0 + float(np.abs(out_onnx).max())))
    print(f"{name}: max |{engine} - ONNX Runtime| over {BLOCKS} blocks = {worst:.2e}", flush=True)
    if worst > tolerance:
        sys.exit(f"{name}: the export differs from ONNX Runtime by {worst:.2e}")
