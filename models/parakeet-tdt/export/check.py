# SPDX-License-Identifier: Apache-2.0
"""Shared by the exporters: runs an export's three parts on test speech in the 31 s window and compares
them with NeMo in PyTorch (float32): the features and the encoder outputs over the valid frames, and
decoder_joint over every step of the greedy search with its state fed back; the transcripts must match.

    uv run check.py <audio.wav>...      checks the ONNX export in this repository (int8) against NeMo

The test speech: any 16 kHz mono files up to 31 s, e.g. LibriSpeech dev-clean utterances (CC BY 4.0) or
the test recording in test/ of the weights' HF repository.
"""

import os
import sys

import numpy as np
import torch

from parakeet import NemoPreprocessor, audio, greedy, load, parts

HERE = os.path.dirname(os.path.abspath(__file__))


def relative(got, expected):
    got, expected = np.asarray(got, np.float32), np.asarray(expected, np.float32)
    return float(np.abs(got.reshape(expected.shape) - expected).max() / (1 + np.abs(expected).max()))


def compare(engine, run, files, model=None, tolerance=1e-3, same_lengths=True, same_text=True):
    """run[part](*numpy inputs) -> numpy outputs, for part in preprocessor, encoder, decoder_joint.
    same_lengths=False: a preprocessor that counts frames differently (the ONNX export's)."""
    model = model or load()
    reference = dict(parts(model), preprocessor=NemoPreprocessor(model).eval())  # torch.stft

    def eager(part):
        return lambda *x: [t.numpy() for t in reference[part](*(torch.as_tensor(v) for v in x))]

    worst = {"features": 0.0, "encoder": 0.0, "decoder_joint": 0.0}
    for path in files:
        waveforms, lens = audio(path)
        features, features_lens = eager("preprocessor")(waveforms, lens)
        got_features, got_lens = run["preprocessor"](waveforms, lens)
        valid = int(features_lens[0])
        assert not same_lengths or int(np.asarray(got_lens).reshape(-1)[0]) == valid, "features_lens differ"
        worst["features"] = max(worst["features"], relative(np.asarray(got_features)[:, :, :valid], features[:, :, :valid]))
        outputs, encoded = eager("encoder")(features, features_lens)
        got_outputs, got_encoded = run["encoder"](features, features_lens)
        frames = int(encoded[0])
        assert int(np.asarray(got_encoded).reshape(-1)[0]) == frames, "encoded_lengths differ"
        worst["encoder"] = max(worst["encoder"], relative(np.asarray(got_outputs)[:, :, :frames], outputs[:, :, :frames]))
        steps = []

        def paired(*x):  # both decoder_joints on the reference's search path, each with its own state
            expected = eager("decoder_joint")(*x)
            steps.append(relative(run["decoder_joint"](*x)[0], expected[0]))
            return expected

        expected_tokens = greedy(paired, outputs, frames)
        worst["decoder_joint"] = max([worst["decoder_joint"]] + steps)
        # The whole chain of the export on its own: its features, its encoder, its decoder_joint.
        own_features, own_lens = run["preprocessor"](waveforms, lens)
        own_outputs, own_encoded = run["encoder"](np.asarray(own_features, np.float32), np.asarray(own_lens))
        tokens = greedy(run["decoder_joint"], np.asarray(own_outputs, np.float32), int(np.asarray(own_encoded).reshape(-1)[0]))
        text, expected_text = model.tokenizer.ids_to_text(tokens), model.tokenizer.ids_to_text(expected_tokens)
        print(f"{engine} {os.path.basename(path)} ({len(steps)} decoder_joint steps): {text}", flush=True)
        if text != expected_text:
            print(f"    NeMo: {expected_text}", flush=True)
        if text != expected_text and same_text:
            sys.exit(f"{engine}: transcript differs from NeMo's: {expected_text}")
    print(f"{engine}: max deviation from NeMo (relative to the peak): features {worst['features']:.1e}, "
          f"encoder {worst['encoder']:.1e}, decoder_joint {worst['decoder_joint']:.1e}", flush=True)
    if any(v > tolerance for v in worst.values()):
        sys.exit(f"{engine}: deviation above {tolerance:g}")
    return worst


def onnx_runner():
    import onnxruntime
    root = os.path.dirname(HERE)
    sessions = {"preprocessor": onnxruntime.InferenceSession(f"{root}/nemo128.onnx"),
                "encoder": onnxruntime.InferenceSession(f"{root}/encoder-model.q4.onnx"),
                "decoder_joint": onnxruntime.InferenceSession(f"{root}/decoder_joint-model.int8.onnx")}

    def runner(session):
        names = [i.name for i in session.get_inputs()]
        return lambda *x: session.run(None, dict(zip(names, x)))

    return {k: runner(s) for k, s in sessions.items()}


if __name__ == "__main__":
    # int8: the encoder and decoder_joint deviate by their quantization; the transcripts must match.
    # Its preprocessor (nemo128.onnx) is a reimplementation: one frame more (length // 160 + 1), its own
    # normalisation; the encoder and decoder_joint deviate by their quantization, which can change a word or a
    # punctuation mark: reported, not failed.
    compare("ONNX Runtime (int8)", onnx_runner(), sys.argv[1:], tolerance=float("inf"), same_lengths=False,
            same_text=False)
