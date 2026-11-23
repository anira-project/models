# SPDX-License-Identifier: Apache-2.0
"""Shared by the exporters: streams random input through an export and the ONNX model over
consecutive calls, feeding back the state pairs a model has (an input named <x>state_in beside an
output <x>state_out), relative to each output's peak."""

import sys

import numpy as np
import onnxruntime
import torch

BLOCKS = 8
TOLERANCE = 1e-4
TOLERANCE_COREML = 2e-3  # Core ML on the GPU, float32


def state_pairs(input_names, output_names):
    return {i: i[:-2] + "out" for i in input_names if i.endswith("state_in") and i[:-2] + "out" in output_names}


def compare(engine, onnx_path, input_names, output_names, shapes, run, tolerance=TOLERANCE):
    """run(list of numpy inputs in input_names order) -> list of outputs in output_names order."""
    session = onnxruntime.InferenceSession(onnx_path)
    pairs = state_pairs(input_names, output_names)
    out_index = {n: k for k, n in enumerate(output_names)}
    state_onnx = {i: np.zeros(shapes[i], np.float32) for i in pairs}
    state_other = dict(state_onnx)
    generator = torch.Generator().manual_seed(1)
    worst = 0.0
    for _ in range(BLOCKS):
        feed = {n: (torch.randn(*shapes[n], generator=generator) * 0.1).numpy() for n in input_names if n not in pairs}
        expected = session.run(None, {**feed, **state_onnx})
        actual = [np.asarray(a) for a in run([{**feed, **state_other}[n] for n in input_names])]
        for e, a in zip(expected, actual):
            worst = max(worst, float(np.max(np.abs(e - a.reshape(e.shape)))) / (1.0 + float(np.max(np.abs(e)))))
        state_onnx = {i: expected[out_index[o]] for i, o in pairs.items()}
        state_other = {i: actual[out_index[o]].reshape(shapes[i]) for i, o in pairs.items()}
    print(f"max |{engine} - ONNX Runtime| over {BLOCKS} calls (state pairs {list(pairs) or 'none'}) = {worst:.2e}", flush=True)
    if worst > tolerance:
        sys.exit(f"the export differs from ONNX Runtime by {worst:.2e}")
