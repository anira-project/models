# SPDX-License-Identifier: Apache-2.0
"""The hand tracking models as PyTorch modules, from our ONNX conversions, for the exporters that take
one (ExecuTorch, Core ML): each ONNX model simplified to static shapes (onnxsim) and turned into a
PyTorch module (onnx2torch, with the converters below made traceable)."""

import onnx
import onnxsim
import torch
from onnx2torch import convert
from onnx2torch.node_converters import registry
from onnx2torch.node_converters.activations import OnnxPReLU
from onnx2torch.node_converters.pad import OnnxPadStatic
from onnx2torch.utils.common import OnnxMapping, OperationConverterResult

# onnx2torch's PReLU falls back to boolean-mask indexing for per-channel slopes in NHWC, which
# torch.export cannot trace (data-dependent shapes). The same function without it:
OnnxPReLU.forward = lambda self, x, slope: torch.where(x >= 0, x, x * slope)


class _Static(torch.nn.Module):
    """A function of the node's first input, its other inputs fixed at conversion time."""

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


# onnx2torch reads these operators' constant inputs (padding, scales, shapes, axes, indices) at run
# time, which torch.export cannot trace. With the inputs constant, each becomes a static function.
# A maker returns None where it does not apply, and the original converter runs.
def _pad(node, c):
    if c[0] is None:
        return None
    value = float(c[1].item()) if len(c) > 1 and c[1] is not None else 0.0
    return OnnxPadStatic.create_from_onnx_params(
        onnx_pads=c[0].tolist(), onnx_mode=node.attributes.get("mode", "constant"), constant_value=value)


def _resize(node, c):
    scales = c[1].tolist() if len(c) > 1 and c[1] is not None else []
    if (node.attributes.get("mode") != "linear" or len(scales) != 4 or scales[:2] != [1.0, 1.0]
            or node.attributes.get("coordinate_transformation_mode") != "half_pixel"):
        return None
    return _Static(lambda x: torch.nn.functional.interpolate(
        x, scale_factor=(scales[2], scales[3]), mode="bilinear", align_corners=False))


def _reshape(node, c):
    if c[0] is None:
        return None
    shape = [int(d) for d in c[0].tolist()]  # 0 copies the input's dimension, -1 is inferred
    return _Static(lambda x: torch.reshape(x, [x.shape[i] if d == 0 else d for i, d in enumerate(shape)]))


def _squeeze(node, c):
    if not c or c[0] is None:
        return None
    axes = tuple(int(a) for a in c[0].tolist())
    return _Static(lambda x: torch.squeeze(x, axes))


def _slice(node, c):
    if any(t is None for t in c):
        return None
    starts, ends = c[0].tolist(), c[1].tolist()
    axes = c[2].tolist() if len(c) > 2 else list(range(len(starts)))
    steps = c[3].tolist() if len(c) > 3 else [1] * len(starts)

    def function(x):
        index = [slice(None)] * x.dim()
        for start, end, axis, step in zip(starts, ends, axes, steps):
            index[axis] = slice(start, None if end >= x.shape[axis] else end, step)
        return x[tuple(index)]

    return _Static(function)


def _gather(node, c):
    if c[0] is None or c[0].dim() > 1:
        return None
    axis = node.attributes.get("axis", 0)
    if c[0].dim() == 0:
        position = int(c[0].item())
        return _Static(lambda x: torch.select(x, axis, position))
    positions = [int(i) for i in c[0].tolist()]
    return _Static(lambda x: torch.index_select(x, axis, torch.tensor(positions)))


_MAKERS = {"Pad": _pad, "Resize": _resize, "Reshape": _reshape, "Squeeze": _squeeze, "Slice": _slice,
           "Gather": _gather}


def _with_constant_inputs(original, maker):
    def converter(node, graph):
        inputs = node.input_values
        constants = [graph.initializers[i].to_torch() if i in graph.initializers else None
                     for i in inputs[1:]]
        module = maker(node, constants)
        if module is None:
            return original(node, graph)
        return OperationConverterResult(
            torch_module=module, onnx_mapping=OnnxMapping(inputs=(inputs[0],), outputs=node.output_values))

    return converter


_REGISTRY = registry._CONVERTER_REGISTRY  # pylint: disable=protected-access
for description in list(_REGISTRY):
    if description.operation_type in _MAKERS:
        _REGISTRY[description] = _with_constant_inputs(
            _REGISTRY[description], _MAKERS[description.operation_type])

MODELS = ["hand_detector", "hand_landmarks_detector", "gesture_embedder", "canned_gesture_classifier"]

def static_shapes(onnx_model):
    return {v.name: [d.dim_value if d.dim_value > 0 else 1 for d in v.type.tensor_type.shape.dim]
            for v in onnx_model.graph.input}


def torch_module(onnx_path, shapes=None):
    """The model as a PyTorch module, and its inputs' static shapes by name (dynamic dimensions 1,
    unless shapes gives them)."""
    onnx_model = onnx.load(onnx_path)
    shapes = {**static_shapes(onnx_model), **(shapes or {})}
    simplified, ok = onnxsim.simplify(onnx_model, overwrite_input_shapes=shapes)
    if not ok:
        raise RuntimeError(f"{onnx_path}: onnxsim could not check the simplified model")
    return convert(constants_as_initializers(simplified)).eval(), shapes


def constants_as_initializers(model):
    """Constant nodes as initializers: onnx2torch reads a Conv's weight from the initializers only."""
    for node in [n for n in model.graph.node if n.op_type == "Constant"]:
        value = next((a for a in node.attribute if a.name == "value"), None)
        if value is None:
            continue
        tensor = onnx.TensorProto()
        tensor.CopyFrom(value.t)
        tensor.name = node.output[0]
        model.graph.initializer.append(tensor)
        model.graph.node.remove(node)
    return model
