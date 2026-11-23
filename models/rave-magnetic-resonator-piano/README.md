# RAVE magnetic resonator piano

The Intelligent Instruments Lab's RAVE model of the magnetic resonator piano
(`mrp_strengjavera_b2048_r44100_z16.ts`, [Intelligent-Instruments-Lab/rave-models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)),
2048 samples per call at 44.1 kHz.

- `rave-magnetic-resonator-piano-{encoder,decoder,forward}.model.json`: exported with explicit state
  for ONNX Runtime, LiteRT, TensorFlow Lite and ExecuTorch. The decoder is a generator: 8 latents
  and 120 noise values per block (static inputs, the caller draws the noise); the encoder returns
  the variational mean; the state is fed back as a state pair.
- `rave-magnetic-resonator-piano-{encoder,decoder,forward}-libtorch.model.json`: the streaming
  TorchScript's `encode`, `decode` and `forward` for LibTorch, its state inside the model.

The export scripts live with the weights in
[anira-project/rave-magnetic-resonator-piano](https://huggingface.co/anira-project/rave-magnetic-resonator-piano) (`export/`).

Licence: CC BY-NC 4.0, model and export scripts alike (the scripts are derived from RAVE's code, which
is CC BY-NC 4.0). Non-commercial use only.
