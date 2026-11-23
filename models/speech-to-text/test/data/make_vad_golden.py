# SPDX-License-Identifier: Apache-2.0
"""Writes vad_golden.json: Silero VAD's speech probability per 512-sample chunk of the test signal
(VadSignal.h, written the same way here), from ONNX Runtime with the state fed back.

    python make_vad_golden.py <silero-vad repository>/onnx/silero_vad_16k.onnx
"""

import json
import os
import sys

import numpy as np
import onnxruntime

RATE = 16000


def signal():
    syllable, period = RATE // 4, RATE // 120
    out = []

    def silence(seconds):
        out.extend([0.0] * int(seconds * RATE))

    def phrase():
        for _ in range(4):
            x = np.zeros(syllable)
            x[::period] = 1.0
            for formant, bandwidth in ((700.0, 80.0), (1200.0, 90.0), (2600.0, 120.0)):
                r = np.exp(-np.pi * bandwidth / RATE)
                a1, a2 = -2.0 * r * np.cos(2.0 * np.pi * formant / RATE), r * r
                y1 = y2 = 0.0
                for i in range(syllable):
                    y0 = x[i] - (a1 * y1) - (a2 * y2)
                    y2, y1 = y1, y0
                    x[i] = y0
                x = x / np.abs(x).max()
            hann = 0.5 - (0.5 * np.cos(2.0 * np.pi * np.arange(syllable) / (syllable - 1)))
            out.extend((0.3 * x * hann).tolist())
            silence(0.15)

    silence(0.5)
    phrase()
    silence(1.0)
    phrase()
    silence(1.0)
    return np.asarray(out, np.float32)


def main():
    session = onnxruntime.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
    audio = signal()
    state = np.zeros((1, 320), np.float32)
    probabilities = []
    for start in range(0, len(audio) - 511, 512):
        probability, state = session.run(
            None, {"audio_in": audio[start:start + 512].reshape(1, 1, 512), "state_in": state})
        probabilities.append(round(float(probability.ravel()[0]), 7))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vad_golden.json")
    with open(path, "w") as f:
        json.dump({"samples": len(audio), "chunk": 512, "probabilities": probabilities}, f)
        f.write("\n")
    print(path, len(probabilities), "chunks")


if __name__ == "__main__":
    main()
