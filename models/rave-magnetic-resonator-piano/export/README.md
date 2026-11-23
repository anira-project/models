# Export

The explicit-state exports are made by the shared RAVE exporter in `tools/export/rave/`
(the same for rave-funk-drum), from the RAVE TorchScript model:

```bash
uv run --project tools/export/rave tools/export/rave/export.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_litert.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_executorch.py <model.ts> <prefix> [--coreml]

uv run --no-project tools/export/rave/export_libtorch.py <model.ts> <prefix>
uv run --no-project tools/export/rave/export_executorch.py <model.ts> <prefix> [--coreml | --vulkan]
uv run --no-project tools/export/rave/export_coreml.py <model.ts> <prefix>
```

Run from the repository root. The weights go to Hugging Face, anira-project/rave-magnetic-resonator-piano, whose README names the
commit of this repository they were made with.

## Source

`libtorch/mrp_strengjavera_b2048_r44100_z16.ts` is unchanged from [Intelligent-Instruments-Lab/rave-models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models) @ `c25a03d`.
