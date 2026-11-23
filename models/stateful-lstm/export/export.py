# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.12.0", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports lstm.py's explicit-state LSTM to ONNX and TorchScript (traced, for LibTorch), after checking
it against the original TorchScript model, which keeps its state inside, over consecutive blocks.

    uv run --no-project export.py <stateful-lstm-dynamic.pt> <prefix>

writes <prefix>.onnx and <prefix>.pt.
"""

import sys

import numpy as np
import onnxruntime
import torch

from lstm import BLOCK, load, signal

BLOCKS = 20
TOLERANCE = 1e-5


def stream(run, state_size):
    state, out = torch.zeros(1, state_size), []
    for block in signal(BLOCKS):
        y, state = run(block, state)
        out.append(np.asarray(y).reshape(-1))
    return np.stack(out)


def main():
    ts_path, prefix = sys.argv[1:3]
    model = load(ts_path)
    original = torch.jit.load(ts_path).eval()
    expected = np.stack([original(block).numpy().reshape(-1) for block in signal(BLOCKS)])
    eager = stream(model, model.state_size)
    print(f"explicit state vs the original over {BLOCKS} blocks: {np.abs(eager - expected).max():.2e} "
          f"(output peak {np.abs(expected).max():.2f})")
    if np.abs(eager - expected).max() > TOLERANCE:
        sys.exit("the rebuild differs from the original")

    inputs = (torch.zeros(BLOCK, 1, 1), torch.zeros(1, model.state_size))
    torch.onnx.export(model, inputs, f"{prefix}.onnx", input_names=["audio_in", "state_in"],
                      output_names=["audio_out", "state_out"], opset_version=17, dynamo=False)
    torch.jit.trace(model, inputs).save(f"{prefix}.pt")

    session = onnxruntime.InferenceSession(f"{prefix}.onnx")
    onnx = stream(lambda x, s: [torch.from_numpy(o) for o in session.run(None, {"audio_in": x.numpy(), "state_in": s.numpy()})],
                  model.state_size)
    traced = torch.jit.load(f"{prefix}.pt").eval()
    ts = stream(traced, model.state_size)
    print(f"ONNX vs the original {np.abs(onnx - expected).max():.2e}, TorchScript {np.abs(ts - expected).max():.2e}")
    if max(np.abs(onnx - expected).max(), np.abs(ts - expected).max()) > TOLERANCE:
        sys.exit("an export differs from the original")


if __name__ == "__main__":
    main()
