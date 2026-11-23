# Export

From the TorchScript model (`lstm.py` rebuilds it with the state as tensors):

```bash
uv run --no-project export.py <stateful-lstm-dynamic.pt> <prefix>              # ONNX, LibTorch
uv run --no-project export_litert.py <stateful-lstm-dynamic.pt> <prefix>
uv run --no-project export_executorch.py <stateful-lstm-dynamic.pt> <prefix> [--coreml | --vulkan]
uv run --no-project export_coreml.py <stateful-lstm-dynamic.pt> <prefix>
```

Run from this directory. The weights go to Hugging Face, anira-project/stateful-lstm, whose README names the
commit of this repository they were made with.

## Source

[vackva/stateful-lstm](https://github.com/vackva/stateful-lstm) @ `3ce7bec`, `models/model_0/`; the training and export notebooks are in that repository.
