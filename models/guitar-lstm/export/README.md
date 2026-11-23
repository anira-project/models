# Export

The formats are derived from the ONNX model by the generic exporters in `tools/export/`
(GuitarLSTM's LSTM is rebuilt from the TorchScript model, `--guitar-lstm`):

```bash
uv run --no-project tools/export/export_litert.py <model.onnx> <out.tflite> [<description>] [--guitar-lstm <pt>]
uv run --no-project tools/export/export_executorch.py <model.onnx> <out.pte> [<description>] [--guitar-lstm <pt>] [--coreml | --vulkan]
uv run --no-project tools/export/export_coreml.py <model.onnx> <out.mlpackage> [<description>] [--guitar-lstm <pt>]
uv run --no-project tools/export/export_libtorch.py <model.onnx> <out.pt> [<description>] [--guitar-lstm <pt>]
```

Run from the repository root. The weights go to Hugging Face, anira-project/guitar-lstm, whose README names the
commit of this repository they were made with.

## Source

[faressc/GuitarLSTM](https://github.com/faressc/GuitarLSTM) @ `08c7418` (a fork of [GuitarML/GuitarLSTM](https://github.com/GuitarML/GuitarLSTM)), `pytorch-version/models/model_0/` and `tensorflow-version/models/model_0/`; the export code is in that repository.
