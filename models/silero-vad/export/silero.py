# SPDX-License-Identifier: MIT
"""Silero VAD's 16 kHz model with explicit state, rebuilt in eager PyTorch from the weights of the
upstream TorchScript model (silero_vad.jit, snakers4/silero-vad):

    (audio_in [1,1,512], state_in [1,320]) -> (probability [1,1,1], state_out [1,320])

512 samples at 16 kHz per call, the speech probability of the block. The state is the LSTM's h and c
(128 each) and the last 64 samples of the previous call, which upstream's wrapper keeps outside the
model and prepends to each block. A zero state is upstream's reset state.
"""

import torch
import torch.nn.functional as F

BLOCK = 512
HIDDEN = 128


class SileroVad16k(torch.nn.Module):
    def __init__(self, jit):
        super().__init__()
        model = jit._model  # pylint: disable=protected-access  (the 16 kHz model; _model_8k is 8 kHz)
        stft = model.stft
        self.context = int(model.context_size_samples)
        self.hop = int(stft.hop_length)
        self.bins = int(stft.filter_length) // 2 + 1
        self.reflect = tuple(stft.padding.padding)
        # A copy with standard strides: upstream's buffer has a zero stride on its size-1 axis, which
        # ExecuTorch's Vulkan delegate refuses.
        basis = stft.forward_basis_buffer.detach()
        self.register_buffer("basis", torch.empty(basis.shape).copy_(basis))
        self.strides = []
        for i, block in enumerate(model.encoder.children()):
            conv = block.reparam_conv
            self.register_buffer(f"conv{i}_weight", conv.weight.detach().clone())
            self.register_buffer(f"conv{i}_bias", conv.bias.detach().clone())
            self.strides.append((int(conv.stride[0]), int(conv.padding[0])))
        for name, value in model.decoder.rnn.named_parameters():  # weight_ih, weight_hh, bias_ih, bias_hh
            self.register_buffer(f"rnn_{name}", value.detach().clone())
        head = list(model.decoder.decoder.children())[2]
        self.register_buffer("head_weight", head.weight.detach().clone())
        self.register_buffer("head_bias", head.bias.detach().clone())

    @property
    def state_size(self):
        return 2 * HIDDEN + self.context

    def forward(self, audio_in, state_in):
        h, c = state_in[:, :HIDDEN], state_in[:, HIDDEN:2 * HIDDEN]
        x = torch.cat([state_in[:, 2 * HIDDEN:], audio_in.reshape(1, BLOCK)], 1)
        context = x[:, -self.context:]
        x = F.pad(x.unsqueeze(1), self.reflect, mode="reflect")
        spectrum = F.conv1d(x, self.basis, None, self.hop)
        real, imag = spectrum[:, :self.bins], spectrum[:, self.bins:]
        x = torch.sqrt(real * real + imag * imag)
        for i, (stride, padding) in enumerate(self.strides):
            x = F.relu(F.conv1d(x, getattr(self, f"conv{i}_weight"), getattr(self, f"conv{i}_bias"), stride, padding))
        h, c = self.lstm_cell(x.squeeze(-1), h, c)
        y = torch.sigmoid(F.conv1d(F.relu(h.unsqueeze(-1)), self.head_weight, self.head_bias))
        probability = y.squeeze(1).mean(1).reshape(1, 1, 1)
        return probability, torch.cat([h, c, context], 1)


    def lstm_cell(self, x, h, c):
        """torch.nn.LSTMCell, its gates (input, forget, cell, output) sliced explicitly: the ONNX
        exporter has no chunk of an LSTMCell."""
        gates = x @ self.rnn_weight_ih.T + self.rnn_bias_ih + h @ self.rnn_weight_hh.T + self.rnn_bias_hh
        i, f, g, o = (gates[:, k * HIDDEN:(k + 1) * HIDDEN] for k in range(4))
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


def load(jit_path):
    torch.set_grad_enabled(False)
    return SileroVad16k(torch.jit.load(jit_path).eval()).eval()


def voiced(seconds=12.0, rate=16000, seed=0):
    """Speech-like audio without recordings: a 120 Hz pulse train (with jitter) through three
    formant resonators, in syllables of about 4 Hz, with pauses and a noise floor, so that a check
    streams both speech-like and silent blocks."""
    generator = torch.Generator().manual_seed(seed)
    n = int(seconds * rate)
    period = rate / (120.0 * (1 + 0.1 * torch.randn(n // 160 + 1, generator=generator)).repeat_interleave(160)[:n])
    pulses = (torch.cumsum(1.0 / period, 0).floor().diff(prepend=torch.zeros(1)) > 0).float()
    signal = pulses
    for formant, bandwidth in ((700.0, 80.0), (1200.0, 90.0), (2600.0, 120.0)):
        r = torch.exp(torch.tensor(-torch.pi * bandwidth / rate))
        a1, a2 = -2 * r * torch.cos(torch.tensor(2 * torch.pi * formant / rate)), r * r
        y, y1, y2 = torch.zeros(n), 0.0, 0.0
        for i in range(n):  # a two-pole resonator
            y1, y2 = float(signal[i]) - float(a1) * y1 - float(a2) * y2, y1
            y[i] = y1
        signal = y / y.abs().max()
    t = torch.arange(n) / rate
    syllables = torch.sin(2 * torch.pi * 4.0 * t).clamp(min=0)
    phrases = (torch.sin(2 * torch.pi * 0.25 * t) > -0.3).float()
    audio = 0.3 * signal * syllables * phrases + 0.003 * torch.randn(n, generator=generator)
    return audio[: n // BLOCK * BLOCK].reshape(-1, 1, 1, BLOCK)
