# SPDX-License-Identifier: Apache-2.0
"""The stateful LSTM with its state as tensors, rebuilt from the TorchScript model
(libtorch/stateful-lstm-dynamic.pt), which keeps the state inside (self.hidden):

    (audio_in [2048,1,1], state_in [1,40]) -> (audio_out [2048,1,1], state_out [1,40])

time, batch, channel as the TorchScript model takes them; the state is the LSTM's h and c (20 each),
zero at the start of a stream, as the TorchScript model's.
"""

import torch
from torch._higher_order_ops.scan import scan

BLOCK = 2048


class ExplicitStateLstm(torch.nn.Module):
    def __init__(self, ts):
        super().__init__()
        lstm = ts.lstm
        self.hidden = int(lstm.hidden_size)
        self.lstm = torch.nn.LSTM(int(lstm.input_size), self.hidden, int(lstm.num_layers))
        self.lstm.load_state_dict({k: v.detach().clone() for k, v in lstm.state_dict().items()})
        self.lin = torch.nn.Linear(int(ts.lin.in_features), int(ts.lin.out_features))
        self.lin.load_state_dict({k: v.detach().clone() for k, v in ts.lin.state_dict().items()})
        self.skip = int(ts.skip)

    @property
    def state_size(self):
        return 2 * self.hidden * self.lstm.num_layers

    def forward(self, audio_in, state_in):
        layers = self.lstm.num_layers
        h = state_in[:, :self.state_size // 2].reshape(layers, 1, self.hidden)
        c = state_in[:, self.state_size // 2:].reshape(layers, 1, self.hidden)
        y, (h, c) = self.lstm(audio_in, (h, c))
        y = self.lin(y)
        if self.skip:
            y = y + audio_in[:, :, 0:self.skip]
        return y, torch.cat([h.reshape(1, -1), c.reshape(1, -1)], 1)


def load(ts_path):
    torch.set_grad_enabled(False)
    return ExplicitStateLstm(torch.jit.load(ts_path).eval()).eval()


def signal(blocks, seed=0):
    """Guitar-like test audio: decaying plucked tones and noise, consecutive blocks [2048,1,1]."""
    generator = torch.Generator().manual_seed(seed)
    n = blocks * BLOCK
    t = torch.arange(n) / 44100.0
    audio = torch.zeros(n)
    for start in range(0, n, 11025):
        f = 82.4 * 2 ** (torch.randint(0, 24, (1,), generator=generator).item() / 12)
        k = torch.arange(n - start) / 44100.0
        audio[start:] += 0.4 * torch.sin(2 * torch.pi * f * k) * torch.exp(-3 * k)
    audio += 0.01 * torch.randn(n, generator=generator)
    return audio.reshape(blocks, BLOCK, 1, 1)


class ScanLstm(torch.nn.Module):
    """ExplicitStateLstm with the recurrence as torch.scan, for ExecuTorch: torch.export unrolls
    torch.nn.LSTM over the block's 2048 steps, thousands of small operators; ExecuTorch runs a scan as a
    loop over one step. Single layer."""

    def __init__(self, model):
        super().__init__()
        assert model.lstm.num_layers == 1
        self.model = model

    def forward(self, audio_in, state_in):
        model, hidden = self.model, self.model.hidden
        lstm = model.lstm
        w_hh = lstm.weight_hh_l0.t().contiguous()
        gates_in = torch.nn.functional.linear(audio_in[:, 0, :], lstm.weight_ih_l0, lstm.bias_ih_l0 + lstm.bias_hh_l0)

        def step(carry, x):
            h, c = carry
            gates = x + h @ w_hh
            i = torch.sigmoid(gates[0:hidden])
            f = torch.sigmoid(gates[hidden:2 * hidden])
            g = torch.tanh(gates[2 * hidden:3 * hidden])
            o = torch.sigmoid(gates[3 * hidden:4 * hidden])
            c = f * c + i * g
            h = o * torch.tanh(c)
            return (h, c), h.clone()

        # One-dimensional h and c: scan wants the carry's strides to match the initial state's.
        (h, c), y = scan(step, (state_in[0, :hidden].clone(), state_in[0, hidden:].clone()), gates_in)
        y = model.lin(y.unsqueeze(1))
        if model.skip:
            y = y + audio_in[:, :, 0:model.skip]
        return y, torch.cat([h, c]).unsqueeze(0)
