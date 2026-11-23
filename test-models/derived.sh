#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The formats derived from the ONNX models by tools/export (Core ML ML programs, ExecuTorch lowered
# to Core ML and to Vulkan), for the float32 networks: simple gain and the stateful accumulator.
# Run after the network scripts (<network>/export/); writes beside their output in ${TEST_MODELS_OUT:-out}.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${TEST_MODELS_OUT:-$here/out}"
tools="$here/../tools/export"
derive() {  # <network> <onnx stem> <description>
    local net=$1 stem=$2 desc="$here/$1/$3"
    local onnx="$out/$net/onnx/$stem.onnx"
    mkdir -p "$out/$net/coreml" "$out/$net/executorch"
    uv run --no-project "$tools/export_coreml.py" "$onnx" "$out/$net/coreml/$stem.mlpackage" "$desc"
    uv run --no-project "$tools/export_executorch.py" "$onnx" "$out/$net/executorch/${stem}_coreml.pte" "$desc" --coreml
    uv run --no-project "$tools/export_executorch.py" "$onnx" "$out/$net/executorch/${stem}_vulkan.pte" "$desc" --vulkan
}
for ch in mono stereo 4ch 16ch; do
    derive simple-gain "simple_gain_network_$ch" "simple-gain-$ch.model.json"
done
for ch in mono stereo; do
    derive stateful-accumulator "stateful_accumulator_network_$ch" "stateful-accumulator-$ch.model.json"
    derive stateful-accumulator "stateful_accumulator_network_negated_$ch" "stateful-accumulator-negated-$ch.model.json"
done
