# Export

The formats are derived from the ONNX models by the generic exporters in `tools/export/`:

```bash
uv run --no-project tools/export/export_litert.py <model.onnx> <out.tflite> [<description>] [--guitar-lstm <pt>]
uv run --no-project tools/export/export_executorch.py <model.onnx> <out.pte> [<description>] [--guitar-lstm <pt>] [--coreml | --vulkan]
uv run --no-project tools/export/export_coreml.py <model.onnx> <out.mlpackage> [<description>] [--guitar-lstm <pt>]
uv run --no-project tools/export/export_libtorch.py <model.onnx> <out.pt> [<description>] [--guitar-lstm <pt>]
```

Run from the repository root. The weights go to Hugging Face, anira-project/steerable-nafx, whose README names the
commit of this repository they were made with.

## Source

[faressc/steerable-nafx](https://github.com/faressc/steerable-nafx) @ `e495467` (a fork of [csteinmetz1/steerable-nafx](https://github.com/csteinmetz1/steerable-nafx)), `models/model_0/`; the export code is in that repository.
