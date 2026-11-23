# Export

From transformers (`nemotron.py`), a uv project of its own (`pyproject.toml`, `uv.lock`):

```bash
uv run export_executorch.py <folder> [--coreml] <audio.wav>...
uv run export_coreml.py <folder> <audio.wav>...
uv run check.py <audio.wav>...        # the ONNX export against transformers
```

Run from this directory. The weights go to Hugging Face, anira-project/nemotron-3-diarization, whose README names the
commit of this repository they were made with.
