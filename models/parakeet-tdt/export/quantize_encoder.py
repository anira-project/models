# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["onnxruntime==1.32.0", "onnx", "onnx-ir", "numpy"]
# ///
"""The 4-bit ONNX encoder: the fp32 export (istupakov/parakeet-tdt-0.6b-v3-onnx, encoder-model.onnx
and its .data) with every MatMul's weights as 4-bit blocks (ONNX Runtime's MatMulNBits: round to
nearest, block 32, asymmetric), the rest fp32.

Why not the int8 encoder of that repository (onnxruntime's dynamic quantisation, the one most users
run, like sherpa-onnx's): on segments cut from a LibriVox reading (45 cuts of 2 to 31 s) it emitted
only blanks for 9 of them and missed 19 % of the words of fp32's transcripts; this one emits text for
all and misses 1.1 %, at fp32's speed and under half its memory.

    uv run quantize_encoder.py <folder with encoder-model.onnx and .data> <out.onnx>
"""

import sys

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main():
    folder, out = sys.argv[1:3]
    model = onnx.load(f"{folder}/encoder-model.onnx")
    quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=False)
    quantizer.process()
    onnx.save_model(quantizer.model.model, out, save_as_external_data=False)


if __name__ == "__main__":
    main()
