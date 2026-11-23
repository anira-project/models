# Test models

Dummy networks for testing anira and this repository, not models anyone runs for their output. Each
checks one thing exactly, with integer-valued or power-of-two arithmetic so every runtime agrees bit
for bit:

| Folder | Networks | Tests |
|---|---|---|
| `simple-gain/` | `audio_out = audio_in * gain` plus the peak; mono, stereo, 4 and 16 channels; a multi-method `.pte` | the plain path, channel counts, static inputs |
| `stateful-accumulator/` | a running sum as an explicit state pair; mono, stereo, and a negated twin | state fed back, switching models with the state kept |
| `typed-gain/` | the gain at int8, int16, float16; unplanned `.pte`; fixtures: the dtype matrix (10 ONNX) and a quantised int8 TFLite | tensor dtypes end to end |
| `variant-gain/` | one Conv1d, three weights (half, double, invert) | switching between models of one form |

The weights are in one Hugging Face repository,
[anira-project/test-models](https://huggingface.co/anira-project/test-models), laid out as
`<folder>/<format>/<file>`; each folder's `test-models.catalog.json` lists its files there.

## Regenerating

`export/` is one uv project for all four (torch, ExecuTorch, ONNX, TensorFlow). Each script exports
its network to TorchScript, ONNX, TFLite and ExecuTorch and checks every runtime against the
expected values with exact equality; `derived.sh` adds the Core ML ML programs and the ExecuTorch
programs lowered to Core ML and Vulkan of the float32 networks (`tools/export`). They write to
`export/out/` (`TEST_MODELS_OUT`), the layout of the Hugging Face repository:

```bash
cd test-models/export
for s in simple_gain_network multi_function_gain stateful_accumulator_network \
         typed_gain_network variant_gain_network; do uv run python $s.py; done
./derived.sh
```

Then upload `out/` with the repository's `LICENSE` and `README.md`, and write the catalogs from the
uploaded revision:

```bash
for f in simple-gain stateful-accumulator typed-gain variant-gain; do
    python3 tools/make_catalog.py <checkout> anira-project/test-models <revision> Apache-2.0 test-models/$f $f
done
```
