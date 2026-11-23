# Export

From transformers (`moonshine.py`), a uv project of its own (`pyproject.toml`, `uv.lock`):

```bash
uv run export_onnx.py <folder> <variant> <audio.wav>...
uv run export_executorch.py <folder> <variant> [--coreml] <audio.wav>...
uv run export_coreml.py <folder> <variant> <audio.wav>...
uv run check.py <variant> <audio.wav>...      # Moonshine's own graphs (upstream/) against transformers
```

Run from this directory. The weights go to Hugging Face, anira-project/moonshine-streaming, whose README names the
commit of this repository they were made with.
