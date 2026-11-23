# SPDX-License-Identifier: Apache-2.0
"""Shared by the exporters: streams lstm.py's test audio through an export and the ONNX export over
consecutive blocks, each feeding its own state back."""

import sys

import numpy as np
import onnxruntime
import torch

from lstm import signal

BLOCKS = 12
TOLERANCE = 1e-5
TOLERANCE_COREML = 2e-3  # Core ML on the GPU, float32


def compare(engine, onnx_path, state_size, run, tolerance=TOLERANCE):
    session = onnxruntime.InferenceSession(onnx_path)
    onnx_state = np.zeros((1, state_size), np.float32)
    other_state = torch.zeros(1, state_size)
    worst = 0.0
    for block in signal(BLOCKS):
        expected, onnx_state = session.run(None, {"audio_in": block.numpy(), "state_in": onnx_state})
        got, other_state = run([block, other_state])
        other_state = torch.as_tensor(np.asarray(other_state))
        worst = max(worst, float(np.abs(np.asarray(got).reshape(expected.shape) - expected).max()))
    print(f"max |{engine} - ONNX Runtime| over {BLOCKS} blocks = {worst:.2e}", flush=True)
    if worst > tolerance:
        sys.exit(f"the export differs from ONNX Runtime by {worst:.2e}")
