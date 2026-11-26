# Export

From the TorchScript model (`lstm.py` rebuilds it with the state as tensors):

```bash
uv run --no-project export.py <stateful-lstm-dynamic.pt> <prefix>              # ONNX, LibTorch
uv run --no-project export_litert.py <stateful-lstm-dynamic.pt> <prefix>
uv run --no-project export_executorch.py <stateful-lstm-dynamic.pt> <prefix> [--coreml | --vulkan]
uv run --no-project export_coreml.py <stateful-lstm-dynamic.pt> <prefix>
```

Run from this directory.

## LiteRT: slow to load, fast to run

litert-torch unrolls `torch.nn.LSTM` over the block's 2048 steps: the `.tflite` has 32,779 operators
(5.5 MB). LiteRT's C API, which anira uses, takes about 5 to 7 s to read it (`LiteRtCreateModelFromFile`
and `FromBuffer`, almost all of it in `UnpackModel` → `LiteRtSubgraphT::EmplaceTensor`, which grows much
faster than the tensor count). That holds for LiteRT 2.1.5, the version anira ships, as well as 2.4.1.
A plugin with this model pays that once per activate, on the main thread, even when another engine is
chosen, because anira prepares every engine's plan. It runs fast: about 2.2 ms per block (46 ms of
audio at 44.1 kHz). We keep it that way: a long load does not hurt the audio, and the run is the fastest.

Alternatives, measured on 2026-11-26 (Apple Silicon, ai-edge-litert 2.4.1 interpreter for the run,
LiteRT 2.1.5 C API for the load). All of them agree with ONNX Runtime to 6.7e-7 over 12 blocks:

| Export | Size | Load | Run per 2048-sample block |
|---|---|---|---|
| litert-torch, unrolled (shipped) | 5.5 MB | 6.7 s | 2.2 ms |
| TensorFlow, a `tf.scan` over steps of 128 unrolled steps each | 450 KB | 0.02 s | 3.8 ms |
| same, 256 steps | 710 KB | 0.08 s | 4.1 ms |
| same, 512 steps | 1.2 MB | 0.34 s | 4.8 ms |
| TensorFlow, a `tf.scan` over single steps (a WHILE) | 180 KB | 0.00 s | 56 ms, slower than real time |

- **torch scan:** `lstm.py`'s `ScanLstm` does not convert, because litert-torch has no lowering for
  `scan`.
- **Fused LSTM:** TFLite's fused op (`UNIDIRECTIONAL_SEQUENCE_LSTM`) cannot return the final state, so
  the converter falls back to a WHILE.
- **Chunked export:** the TensorFlow model takes `lstm.py`'s weights as constants. It computes the input
  gates for the whole block in one matmul, then scans over chunks, with the steps inside a chunk written
  out in Python, and has a serving signature named `audio_in` / `state_in` → `audio_out` / `state_out`. The weights go to Hugging Face, anira-project/stateful-lstm, whose README names the
commit of this repository they were made with.

## Source

[vackva/stateful-lstm](https://github.com/vackva/stateful-lstm) @ `3ce7bec`, `models/model_0/`; the training and export notebooks are in that repository.
