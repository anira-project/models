# Export

The explicit-state exports are made by the shared RAVE exporter in `tools/export/rave/`
(the same for rave-magnetic-resonator-piano), from the RAVE TorchScript model:

```bash
uv run --project tools/export/rave tools/export/rave/export.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_litert.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_executorch.py <model.ts> <prefix> [--coreml]

uv run --no-project tools/export/rave/export_libtorch.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_executorch.py <model.ts> <prefix> [--coreml | --vulkan]
uv run --no-project tools/export/rave/export_coreml.py <model.ts> <prefix>
```

Run from the repository root. The weights go to Hugging Face, anira-project/rave-funk-drum, whose README names the
commit of this repository they were made with.

## Source

`libtorch/rave_funk_drum.ts` is unchanged from [anira-project/example-models](https://github.com/anira-project/example-models) @ `d1bc5cf`, `third-party/ircam-acids/RAVE/rave_funk_drum.ts` (the same file anira pins by SHA256). RAVE is by [acids-ircam](https://github.com/acids-ircam/RAVE).
