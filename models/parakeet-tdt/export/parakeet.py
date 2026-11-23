# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's Parakeet TDT 0.6B v3 (NeMo) in the three parts and with the tensor names of the ONNX
export in this repository (istupakov's, NeMo's own export wrappers), for the formats with fixed shapes
(Core ML, ExecuTorch): one window of 31 s, the longest segment anira models' voice activity
detector cuts (30 s of speech and 0.4 s of padding on each side). Shorter audio is zero-padded; the
lengths mask the padding.

    preprocessor   waveforms [1,496000], waveforms_lens [1]      -> features [1,128,3101], features_lens [1]
    encoder        audio_signal [1,128,3101], length [1]         -> outputs [1,1024,388], encoded_lengths [1]
    decoder_joint  encoder_outputs [1,1024,1], targets [1,1], target_length [1],
                   input_states_1 [2,1,640], input_states_2 [2,1,640]
                                                                 -> outputs [1,1,1,8198], prednet_lengths [1],
                                                                    output_states_1, output_states_2

decoder_joint is one step of the TDT search: the previous token (the blank, 8192, at the start) and the
prediction network's state in, the logits of 8192 tokens, the blank and 5 durations (0 to 4 frames) and
the next state out. Lengths are int64 as in the ONNX export (int32 in Core ML, which has no int64),
targets and target_length int32.
"""

import numpy as np
import soundfile
import torch

RATE = 16000
WINDOW = 31 * RATE
MEL_FRAMES = WINDOW // 160 + 1
ENCODER_FRAMES = 388
STATE = (2, 1, 640)
BLANK = 8192
DURATIONS = [0, 1, 2, 3, 4]
NAMES = {
    "preprocessor": (["waveforms", "waveforms_lens"], ["features", "features_lens"]),
    "encoder": (["audio_signal", "length"], ["outputs", "encoded_lengths"]),
    "decoder_joint": (["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
                      ["outputs", "prednet_lengths", "output_states_1", "output_states_2"]),
}


REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"  # nvidia/parakeet-tdt-0.6b-v3


def load():
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download
    torch.set_grad_enabled(False)
    path = hf_hub_download("nvidia/parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v3.nemo", revision=REVISION)
    model = nemo_asr.models.ASRModel.restore_from(path, map_location="cpu").eval()
    model.decoder._rnnt_export = True  # NeMo's export mode: no start token, raw logits
    model.joint.log_softmax = False
    return model


class Preprocessor(torch.nn.Module):
    """NeMo's FilterbankFeatures (inference) with its STFT as a convolution over a DFT basis: the same
    frames, window and padding as torch.stft, without complex tensors, which Core ML cannot take and
    ExecuTorch's delegates do not lower."""

    def __init__(self, model):
        super().__init__()
        f = model.preprocessor.featurizer
        assert (not f.exact_pad and f.stft_pad_amount is None and f.frame_splicing == 1 and f.pad_to == 0
                and f.normalize == "per_feature" and f.log and f.log_zero_guard_type == "add")
        self.featurizer = f
        n_fft, win = f.n_fft, f.win_length
        window = torch.zeros(n_fft)
        window[(n_fft - win) // 2:(n_fft - win) // 2 + win] = f.window.float()  # torch.stft centres it
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        n = torch.arange(n_fft, dtype=torch.float64)[None, :]
        angle = 2 * torch.pi * k * n / n_fft
        basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * window.double()
        self.register_buffer("basis", basis.float().unsqueeze(1))  # [2 * bins, 1, n_fft]

    def forward(self, waveforms, waveforms_lens):
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch
        f = self.featurizer
        seq_len = f.get_seq_len(waveforms_lens)
        x = waveforms
        if f.preemph is not None:
            timemask = torch.arange(x.shape[1]).unsqueeze(0) < waveforms_lens.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - f.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)
        x = torch.nn.functional.pad(x.unsqueeze(1), (f.n_fft // 2, f.n_fft // 2))
        spectrum = torch.nn.functional.conv1d(x, self.basis, stride=f.hop_length)
        bins = self.basis.shape[0] // 2
        x = torch.sqrt(spectrum[:, :bins] ** 2 + spectrum[:, bins:] ** 2)
        if f.mag_power != 1.0:
            x = x.pow(f.mag_power)
        x = torch.matmul(f.fb.to(x.dtype), x)
        x = torch.log(x + f.log_zero_guard_value_fn(x))
        x, _, _ = normalize_batch(x, seq_len, normalize_type=f.normalize)
        mask = torch.arange(x.size(-1)).unsqueeze(0) >= seq_len.unsqueeze(1)
        return x.masked_fill(mask.unsqueeze(1), f.pad_value), seq_len


class NemoPreprocessor(torch.nn.Module):
    """NeMo's own, torch.stft: the reference Preprocessor is checked against."""

    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, waveforms, waveforms_lens):
        return self.preprocessor(input_signal=waveforms, length=waveforms_lens)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, audio_signal, length):
        return self.encoder(audio_signal=audio_signal, length=length)


class DecoderJoint(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder, self.joint = model.decoder, model.joint

    def forward(self, encoder_outputs, targets, target_length, input_states_1, input_states_2):
        g, state = self.decoder.predict(targets.long(), state=[input_states_1, input_states_2], add_sos=False)
        logits = self.joint.joint(encoder_outputs.transpose(1, 2), g)
        # prednet_lengths is target_length (NeMo's), computed: Core ML has no output that is an input.
        return logits, (target_length + 1) - 1, state[0], state[1]


def parts(model):
    return {"preprocessor": Preprocessor(model).eval(), "encoder": Encoder(model).eval(),
            "decoder_joint": DecoderJoint(model).eval()}


def examples():
    """Inputs of the fixed window, the shapes every export takes."""
    return {
        "preprocessor": (torch.zeros(1, WINDOW), torch.tensor([WINDOW])),
        "encoder": (torch.zeros(1, 128, MEL_FRAMES), torch.tensor([MEL_FRAMES - 1])),
        "decoder_joint": (torch.zeros(1, 1024, 1), torch.tensor([[BLANK]], dtype=torch.int32),
                          torch.tensor([1], dtype=torch.int32), torch.zeros(*STATE), torch.zeros(*STATE)),
    }


def audio(path):
    """A mono 16 kHz file, zero-padded to the window: (waveforms [1,WINDOW], waveforms_lens [1])."""
    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == RATE and samples.ndim == 1 and len(samples) <= WINDOW
    window = np.zeros((1, WINDOW), np.float32)
    window[0, :len(samples)] = samples
    return window, np.array([len(samples)], np.int64)


def greedy(run_decoder_joint, encoder_outputs, frames, max_symbols=10):
    """The TDT greedy search (anira models' TdtGreedyDecoder) over one format's decoder_joint:
    run_decoder_joint(encoder_outputs [1,1024,1], targets, target_length, state_1, state_2) -> numpy
    (logits, _, state_1, state_2). Returns the token ids. Every step feeds the state back."""
    state_1, state_2 = np.zeros(STATE, np.float32), np.zeros(STATE, np.float32)
    tokens, previous, frame, emitted = [], BLANK, 0, 0
    while frame < frames:
        logits, _, next_1, next_2 = run_decoder_joint(
            np.ascontiguousarray(encoder_outputs[:, :, frame:frame + 1]), np.array([[previous]], np.int32),
            np.array([1], np.int32), state_1, state_2)
        logits = np.asarray(logits).reshape(-1)
        token = int(logits[:BLANK + 1].argmax())
        duration = DURATIONS[int(logits[BLANK + 1:].argmax())]
        if token != BLANK:
            tokens.append(token)
            previous, state_1, state_2 = token, np.asarray(next_1), np.asarray(next_2)
            emitted += 1
        if duration == 0 and (token == BLANK or emitted >= max_symbols):
            duration = 1
        if duration > 0:
            frame += duration
            emitted = 0
    return tokens
