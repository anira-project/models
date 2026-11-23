# Nemotron 3.5 ASR Streaming 0.6B (int4)

NVIDIA's cache-aware streaming ASR (FastConformer encoder, RNNT decoder, 40 languages) as onnx-community's
int4 ONNX export, unchanged; the weights are mirrored on Hugging Face,
[anira-project/nemotron-3.5-asr-streaming-int4](https://huggingface.co/anira-project/nemotron-3.5-asr-streaming-int4)
(OpenMDW-1.1; its README names the source revisions). No export scripts: nothing is re-exported.

Three graphs, each one job under the Async contract: the encoder (one 560 ms step, 65 log-mel frames in,
7 encoder frames out, its caches as State pairs), and the RNNT loop's decoder (one token, the LSTM state
passed by the wrapper, which advances it only on a non-blank token) and joint network. The front end
(log-mel) and the greedy RNNT loop are the wrapper's.
