# SPDX-License-Identifier: Apache-2.0
"""Exports moonshine.py's encoder and decoder to ONNX (opset 18, float32) and checks them (check.py).

    uv run export_onnx.py <folder> <variant> <audio.wav>...

writes <folder>/<variant>/{encoder,decoder}.onnx.
"""

import os
import sys
import time

import numpy as np
import onnxruntime
import torch

import moonshine as m
from check import compare


def main():
    folder, variant, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    model, tokenizer = m.load(variant)
    modules = {"encoder": m.Encoder(model).eval(), "decoder": m.Decoder(model).eval()}
    sessions = {}
    for graph, module in modules.items():
        start = time.perf_counter()
        path = f"{folder}/{variant}/{graph}.onnx"
        inputs, outputs = m.NAMES[graph]
        torch.onnx.export(module, m.examples(model)[graph], path, input_names=inputs, output_names=outputs,
                          opset_version=18, dynamo=True, external_data=False)
        print(f"{path}: {os.path.getsize(path):,} bytes, {time.perf_counter() - start:.0f} s", flush=True)
        sessions[graph] = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

    def run(graph, *values):
        names = [i.name for i in sessions[graph].get_inputs()]
        return sessions[graph].run(None, dict(zip(names, (np.asarray(v) for v in values))))

    for graph in modules:
        example = [x.numpy() for x in m.examples(model)[graph]]
        run(graph, *example)
        start = time.perf_counter()
        run(graph, *example)
        print(f"{graph}: {(time.perf_counter() - start) * 1e3:.2f} ms per call", flush=True)
    compare("ONNX Runtime", run, variant, files, model, tokenizer)


if __name__ == "__main__":
    main()
