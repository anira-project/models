# Export

From Silero's TorchScript model (`silero.py` rebuilds it with the state as tensors):

```bash
uv run --no-project export.py <silero_vad.jit> <silero_vad.onnx> <prefix>
uv run --no-project export_litert.py <silero_vad.jit> <prefix>
uv run --no-project export_executorch.py <silero_vad.jit> <prefix> [--coreml]

uv run --no-project export_executorch.py <silero_vad.jit> <prefix> [--coreml | --vulkan]
uv run --no-project export_coreml.py <silero_vad.jit> <prefix>
```

Run from this directory. The weights go to Hugging Face, anira-project/silero-vad, whose README names the
commit of this repository they were made with.
