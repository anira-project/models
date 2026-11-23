# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["onnxruntime==1.32.0", "onnx", "onnx-ir", "numpy"]
# ///
"""The 4-bit ONNX encoder: the fp32 export (istupakov/parakeet-tdt-0.6b-v3-onnx, encoder-model.onnx
and its .data) with every MatMul's weights as 4-bit blocks (ONNX Runtime's MatMulNBits: round to
nearest, block 32, asymmetric), the rest fp32. The Conformer layers' pointwise convolutions (1x1, 48
of them) are matrix products first, so they are 4-bit too.

Why not the int8 encoder of that repository (onnxruntime's dynamic quantisation, the one most users
run, like sherpa-onnx's): on segments cut from a LibriVox reading (45 cuts of 2 to 31 s) it emitted
only blanks for 9 of them and missed 19 % of the words of fp32's transcripts (per-channel int8: 4 and
30 %); this one emits text for all and misses about 1 %, at fp32's speed and in under a fifth of
its memory.

    uv run quantize_encoder.py <folder with encoder-model.onnx and .data> <out.onnx>
"""

import sys

import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def pointwise_convolutions_as_matmuls(model):
    """Conv(x [N,C,T], W [O,C,1]) without bias, stride, padding or groups as
    Transpose(MatMul(Transpose(x), W^T)): the same values, and a MatMul the quantiser takes."""
    weights = {i.name: i for i in model.graph.initializer}
    nodes, used = [], set()
    for node in model.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        weight = weights.get(node.input[1]) if node.op_type == "Conv" and len(node.input) == 2 else None
        pointwise = (weight is not None and len(weight.dims) == 3 and list(attrs.get("kernel_shape", [])) == [1]
                     and attrs.get("group", 1) == 1 and set(attrs.get("strides", [1])) == {1}
                     and set(attrs.get("dilations", [1])) == {1} and set(attrs.get("pads", [0])) == {0})
        if not pointwise:
            nodes.append(node)
            continue
        out = node.output[0]
        transposed = numpy_helper.from_array(numpy_helper.to_array(weight)[:, :, 0].T.copy(), weight.name + "_t")
        model.graph.initializer.append(transposed)
        nodes += [helper.make_node("Transpose", [node.input[0]], [out + "_ntc"], perm=[0, 2, 1]),
                  helper.make_node("MatMul", [out + "_ntc", transposed.name], [out + "_mm"]),
                  helper.make_node("Transpose", [out + "_mm"], [out], perm=[0, 2, 1])]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for node in model.graph.node:
        used.update(node.input)
    kept = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return model


def main():
    folder, out = sys.argv[1:3]
    model = pointwise_convolutions_as_matmuls(onnx.load(f"{folder}/encoder-model.onnx"))
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=False)
    quantizer.process()
    onnx.save_model(quantizer.model.model, out, save_as_external_data=False)


if __name__ == "__main__":
    main()
