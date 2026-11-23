# Export

`export_executorch.py` and `export_coreml.py` here convert the four ONNX models (using
`tools/export/onnx_to_torch.py`):

```bash
uv run export_executorch.py <folder laid out like this repository> [--coreml]

uv run export_executorch.py <folder laid out like this repository> [--coreml | --vulkan]
uv run export_coreml.py <folder laid out like this repository>
```

`make_test_images.py` writes the test images of `test/test_HandTracker.cpp` (MediaPipe's own, as
PPM, for `test/` of the Hugging Face repository) and prints MediaPipe's GestureRecognizer's result on
each, the test's reference values:

```bash
uv run make_test_images.py <out dir>
```

Run from this directory. The weights go to Hugging Face, anira-project/mediapipe-gesture-recognizer, whose README names the
commit of this repository they were made with.
