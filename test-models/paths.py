# SPDX-License-Identifier: Apache-2.0
"""Where the export scripts write: the layout of the Hugging Face repository anira-project/test-models,
<out>/<network>/<format>/<file>, with <out> the TEST_MODELS_OUT environment variable (default: out/
beside this file)."""

import os

OUT = os.environ.get("TEST_MODELS_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))

_FORMATS = {".onnx": "onnx", ".tflite": "litert", ".pt": "libtorch", ".pte": "executorch"}


def out_path(network: str, name: str) -> str:
    """The path of <name> for <network>, its format folder created."""
    folder = os.path.join(OUT, network, _FORMATS[os.path.splitext(name)[1]])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)
