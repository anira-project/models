# Export

`export_executorch.py` and `export_coreml.py` here convert the four ONNX models (using
`tools/export/onnx_to_torch.py`):

```bash
uv run export_executorch.py <folder laid out like this repository> [--coreml]

uv run export_executorch.py <folder laid out like this repository> [--coreml | --vulkan]
uv run export_coreml.py <folder laid out like this repository>
```

Run from this directory. The weights go to Hugging Face, anira-project/mediapipe-gesture-recognizer, whose README names the
commit of this repository they were made with.
