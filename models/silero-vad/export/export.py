# SPDX-License-Identifier: MIT
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.12.0", "onnx", "onnxruntime", "numpy"]
# ///
"""Exports silero.py's explicit-state 16 kHz model to ONNX and TorchScript (traced, for LibTorch),
after checking it against upstream: the TorchScript model's 16 kHz path (its state inside) and the
upstream ONNX model (its context and state passed by the caller, as upstream's wrapper does).

    uv run --no-project export.py <silero_vad.jit> <silero_vad.onnx> <prefix>

writes <prefix>.onnx and <prefix>.pt.
"""

import sys

import numpy as np
import onnxruntime
import torch

from silero import BLOCK, load, voiced

TOLERANCE = 1e-5
AUDIO = voiced()


def stream(run):
    return np.array([float(np.asarray(run(block)).reshape(-1)[0]) for block in AUDIO])


def main():
    jit_path, onnx_path, prefix = sys.argv[1:4]
    model = load(jit_path)

    def ours(module):
        state = torch.zeros(1, model.state_size)
        def run(audio):
            nonlocal state
            probability, state = module(audio, state)
            return probability
        return run

    jit = torch.jit.load(jit_path).eval()
    jit.reset_states()
    upstream_jit = stream(lambda audio: jit(audio.reshape(1, BLOCK), 16000))

    session = onnxruntime.InferenceSession(onnx_path)
    onnx_state, onnx_context = np.zeros((2, 1, 128), np.float32), np.zeros((1, 64), np.float32)
    def upstream_onnx(audio):
        nonlocal onnx_state, onnx_context
        x = np.concatenate([onnx_context, audio.reshape(1, BLOCK).numpy()], 1)
        out, onnx_state = session.run(None, {"input": x, "state": onnx_state, "sr": np.array(16000, np.int64)})
        onnx_context = x[:, -64:]
        return out
    upstream_onnx = stream(upstream_onnx)

    eager = stream(ours(model))
    print(f"eager vs upstream TorchScript {np.abs(eager - upstream_jit).max():.2e}, "
          f"vs upstream ONNX {np.abs(eager - upstream_onnx).max():.2e} "
          f"(probabilities {upstream_onnx.min():.3f}..{upstream_onnx.max():.3f})")
    if max(np.abs(eager - upstream_jit).max(), np.abs(eager - upstream_onnx).max()) > TOLERANCE:
        sys.exit("the rebuild differs from upstream")

    inputs = (torch.zeros(1, 1, BLOCK), torch.zeros(1, model.state_size))
    torch.onnx.export(model, inputs, f"{prefix}.onnx", input_names=["audio_in", "state_in"],
                      output_names=["probability", "state_out"], opset_version=17, dynamo=False)
    traced = torch.jit.trace(model, inputs)
    traced.save(f"{prefix}.pt")

    ours_onnx = onnxruntime.InferenceSession(f"{prefix}.onnx")
    state = np.zeros((1, model.state_size), np.float32)
    def exported_onnx(audio):
        nonlocal state
        out, state = ours_onnx.run(None, {"audio_in": audio.numpy(), "state_in": state})
        return out
    exported_onnx = stream(exported_onnx)
    exported_ts = stream(ours(torch.jit.load(f"{prefix}.pt").eval()))
    print(f"ONNX vs eager {np.abs(exported_onnx - eager).max():.2e}, "
          f"TorchScript vs eager {np.abs(exported_ts - eager).max():.2e}")
    if max(np.abs(exported_onnx - eager).max(), np.abs(exported_ts - eager).max()) > TOLERANCE:
        sys.exit("an export differs from the rebuild")


if __name__ == "__main__":
    main()
