# SPDX-License-Identifier: Apache-2.0
"""Shared by the exporters: streams test speech through an export's encoder and decoder as the C++ wrapper
does (0.5 s steps, a flush at the end of the segment, greedy decoding after every step and at the end), each
call fed the export's own state, and compares every call with moonshine.py in PyTorch (float32) on the same
inputs. The final transcripts must equal transformers' on the whole segment.

    uv run check.py <variant> <audio.wav>...      the upstream graphs (upstream/<variant>/, int8) against
                                                  transformers: transcripts only, reported

The test speech: 16 kHz mono files up to 16 s (macOS: say -o x.aiff "..." and
afconvert -f WAVE -d LEI16@16000 -c 1 x.aiff x.wav).
"""

import os
import sys

import numpy as np
import torch

import moonshine as m

HERE = os.path.dirname(os.path.abspath(__file__))


def relative(got, expected):
    got, expected = np.asarray(got, np.float32), np.asarray(expected, np.float32)
    return float(np.abs(got.reshape(expected.shape) - expected).max() / (1 + np.abs(expected).max()))


def compare(engine, run, variant, files, model=None, tokenizer=None, tolerance=1e-3):
    """run(graph, *numpy inputs) -> numpy outputs, for graph in encoder, decoder."""
    if model is None:
        model, tokenizer = m.load(variant)
    expected_run = m.torch_run(m.Encoder(model).eval(), m.Decoder(model).eval())
    worst = {"encoder": 0.0, "decoder": 0.0}
    calls = {"encoder": 0, "decoder": 0}

    def paired(graph, *inputs):
        got = run(graph, *inputs)
        expected = expected_run(graph, *inputs)
        worst[graph] = max([worst[graph]] + [relative(g, e) for g, e in zip(got, expected)])
        calls[graph] += 1
        return [np.asarray(g, np.float32) for g in got]

    for path in files:
        samples = m.audio(path)
        tokens, _, _ = m.stream(model, paired, samples, decode_every=True)
        _, expected = m.reference(model, samples)
        text, expected_text = tokenizer.decode(tokens), tokenizer.decode(expected)
        print(f"{engine} {variant} {os.path.basename(path)}: {text}", flush=True)
        if text != expected_text:
            sys.exit(f"{engine}: transcript differs from transformers': {expected_text}")
    print(f"{engine} {variant}: {calls['encoder']} encoder and {calls['decoder']} decoder calls, max deviation "
          f"(relative to the peak): encoder {worst['encoder']:.1e}, decoder {worst['decoder']:.1e}", flush=True)
    if any(v > tolerance for v in worst.values()):
        sys.exit(f"{engine}: deviation above {tolerance:g}")
    return worst


def upstream(variant, model, samples):
    """Moonshine's own graphs as its runtime chains them (core/moonshine-streaming-model.cpp) on a whole
    segment, from transformers' frontend features (the upstream frontend takes its weights as inputs)."""
    import onnxruntime

    folder = f"{os.path.dirname(HERE)}/upstream/{variant}"
    session = lambda name: onnxruntime.InferenceSession(f"{folder}/{name}.ort")
    encoder, adapter, cross, decoder = (session(n) for n in ("encoder", "adapter", "cross_kv", "decoder_kv"))
    padded = np.concatenate([samples, np.zeros(-len(samples) % (m.CHUNK * m.FRAME), np.float32)])
    with torch.no_grad():
        features, _ = model.model.encoder.embedder(torch.from_numpy(padded)[None])
    encoded = encoder.run(None, {"features": features.numpy()})[0]
    memory = adapter.run(None, {"encoded": encoded, "pos_offset": np.zeros(1, np.int64)})[0]
    k_cross, v_cross = cross.run(None, {"memory": memory})
    s = m.sizes(model)
    k_self = np.zeros((s["layers"], 1, s["heads"], 0, s["head_dim"]), np.float32)
    v_self = k_self
    limit = min(int(np.ceil(len(padded) / m.SAMPLE_RATE * m.TOKENS_PER_SECOND)), m.TOKENS - 1)
    token, tokens = m.BOS, []
    for _ in range(limit):
        logits, k_self, v_self = decoder.run(None, {
            "token": np.array([[token]], np.int64), "k_self": k_self, "v_self": v_self,
            "out_k_cross": k_cross, "out_v_cross": v_cross})[:3]
        token = int(np.argmax(logits[0, -1]))
        if token == m.EOS:
            break
        tokens.append(token)
    return tokens


if __name__ == "__main__":
    variant, files = sys.argv[1], sys.argv[2:]
    model, tokenizer = m.load(variant)
    for path in files:
        samples = m.audio(path)
        _, expected = m.reference(model, samples)
        text, expected_text = tokenizer.decode(upstream(variant, model, samples)), tokenizer.decode(expected)
        print(f"upstream (int8) {variant} {os.path.basename(path)}: {text}", flush=True)
        if text != expected_text:
            print(f"    transformers: {expected_text}", flush=True)
