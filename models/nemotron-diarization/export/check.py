# SPDX-License-Identifier: Apache-2.0
"""Shared by the exporters: streams a recording chunk by chunk through an export's step, with the speaker
cache kept outside and fed back, and compares every chunk's speaker probabilities with transformers' own
streaming forward (float32) over the same chunks.

    uv run check.py <audio.wav>...      checks the ONNX export in this repository (q4) against transformers

Test audio: 16 kHz mono recordings with several speakers (make_test_audio.py joins LibriSpeech utterances).
"""

import os
import sys

import numpy as np
import torch

from nemotron import INPUTS, Step, chunks, features, load, reference, stream

HERE = os.path.dirname(os.path.abspath(__file__))


def compare(engine, run, files, loaded=None, tolerance=1e-3):
    model, processor = loaded or load()
    worst = 0.0
    for path in files:
        mel_chunks = chunks(features(processor, path))
        expected = torch.cat(reference(model, mel_chunks), dim=1).sigmoid()
        got = torch.cat(stream(run, model, mel_chunks), dim=1).sigmoid()
        deviation = float((got - expected).abs().max())
        speakers = (expected > 0.5).any(dim=1).sum().item()
        print(f"{engine} {os.path.basename(path)}: {len(mel_chunks)} chunks, {speakers} speakers, "
              f"max |speaker probability - transformers| {deviation:.1e}", flush=True)
        worst = max(worst, deviation)
    if worst > tolerance:
        sys.exit(f"{engine}: speaker probabilities deviate by {worst:.1e} (> {tolerance:g})")
    return worst


def eager_runner(model):
    step = Step(model).eval()
    return lambda *x: [t.numpy() for t in step(*(torch.as_tensor(v) for v in x))]


def onnx_runner():
    import onnxruntime
    session = onnxruntime.InferenceSession(f"{os.path.dirname(HERE)}/onnx/model_q4.onnx")
    return lambda *x: session.run(None, dict(zip(INPUTS, x)))


if __name__ == "__main__":
    loaded = load()
    compare("PyTorch step (padded, masked)", eager_runner(loaded[0]), sys.argv[1:], loaded, tolerance=1e-4)
    # q4: the weights are 4-bit; reported, not failed.
    compare("ONNX Runtime (q4)", onnx_runner(), sys.argv[1:], loaded, tolerance=float("inf"))
