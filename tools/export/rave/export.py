# SPDX-License-Identifier: CC-BY-NC-4.0
# Derived from RAVE (acids-ircam/RAVE, CC BY-NC 4.0); see LICENSE. Copied from tanh-lab/anira-pitch-demo
# examples/hand-synth/tools/rave-onnx @ 62dc52f.
"""Export the encoder, decoder and forward of a streaming RAVE TorchScript model (cached_conv,
CachedPQMF; RAVE v2's EncoderV2 and GeneratorV2 or v1's Encoder and Generator) to ONNX with explicit
state, so a stateless runtime can stream it:

  <prefix>_encoder.onnx  (audio_in[1,1,B], state_in[1,E])              -> (latents[1,L,1], state_out[1,E])
  <prefix>_decoder.onnx  (latents[1,L,1], state_in[1,D], noise[1,N,1]) -> (audio[1,1,B], state_out[1,D])
  <prefix>_forward.onnx  (audio_in[1,1,B], state_in[1,E+D], noise[1,N,1]) -> (audio_out[1,1,B], state_out[1,E+D])

The cached convolutions keep their padding caches as module buffers updated in place, which
ONNX cannot express. Here the decoder is re-run from the TorchScript module's weights with every
cache passed in and returned. As in RAVE's decode(), it un-projects the PCA latents and refills
the N cropped dims with noise. The decoder takes that noise as an input (standard normal, the caller draws it): the
decoder is deterministic, testable, and needs no RandomNormalLike op. A v1 Generator's noise
synthesis takes its noise as an input too, synth_noise (uniform in [-1, 1], one value per filtered
sample), after the noise inputs of decoder and forward; its FFT filter is an equivalent matmul.
The encoder returns the variational mean (RAVE's encode() samples around it), projected
onto the first L PCA dims; forward is encode then decode. The export is checked against the
TorchScript model (PyTorch) and ONNX Runtime.

<prefix>_decoder_reference.bin streams fixed latents and noise through the decoder from a zero
state (PyTorch), for the C++ test: int32 blocks, L, N, B; then per block latents[L] and noise[N]
(float32); then the audio of all blocks (float32).
"""
import argparse, os, re, tempfile
import numpy as np, torch
import torch.nn.functional as F


def conv_args(module, transpose):
    """stride / dilation (/ groups) baked into the TorchScript code of one conv instance."""
    pattern = r"conv_transpose1d\(x, weight, None, \[(\d+)\], \[0\], \[0\], 1, \[(\d+)\]\)" if transpose \
        else r"conv1d\(x1, weight, bias, \[(\d+)\], \[0\], \[(\d+)\](?:, (\d+))?\)"
    match = re.search(pattern, module.code)
    if match is None:
        raise RuntimeError(f"unsupported conv code:\n{module.code}")
    if transpose:
        return int(match.group(1)), int(match.group(2))
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 1)


def mod_sigmoid(x):  # rave.core.mod_sigmoid
    return 2 * torch.sigmoid(x) ** 2.3 + 1e-7


def impulse_response_matrix(bands, target_size):
    """rave.core.amp_to_impulse_response is linear in the band amplitudes: the matrix [bands, target]
    it applies, from the original (FFT) code run on the identity."""
    amp = torch.stack([torch.eye(bands, dtype=torch.float64), torch.zeros(bands, bands, dtype=torch.float64)], -1)
    amp = torch.fft.irfft(torch.view_as_complex(amp))
    size = amp.shape[-1]
    amp = torch.roll(amp, size // 2, -1) * torch.hann_window(size, dtype=amp.dtype)
    amp = F.pad(amp, (0, target_size - size))
    return torch.roll(amp, -size // 2, -1).float()


def causal_filter(signal, kernel):
    """rave.core.fft_convolve (signal and kernel of length T, zero-padded to 2T, second half of the
    product) is the causal convolution out[p] = sum_{m <= p} signal[m] * kernel[p - m]: a matmul."""
    size = signal.shape[-1]
    lag = torch.arange(size).unsqueeze(0) - torch.arange(size).unsqueeze(1)  # [m, p] -> p - m
    toeplitz = kernel[..., lag.clamp(min=0)] * (lag >= 0)
    return (signal.unsqueeze(-2) @ toeplitz).squeeze(-2)


class Interpreter:
    """Runs a TorchScript cached_conv module tree functionally; caches come from and go to a flat state."""

    def __init__(self):
        self.shapes, self.initial = [], []
        self.collect = False
        self.synth_noise_shape = None
        self.impulse = {}
        # Whether an AdaIN layer is active, per module: constant per model, read once (the dry run in
        # StreamingRave fills it) so that a traced export never reads the buffers.
        self.adain_active = {}

    def begin(self, state, collect=False, synth_noise=None):
        self.state, self.offset, self.new_state, self.collect = state, 0, [], collect
        self.synth_noise = synth_noise

    def end(self):
        return torch.cat(self.new_state, 1) if self.new_state else torch.zeros(1, 0)

    def take(self, shape, initial):
        if self.collect:
            self.shapes.append(tuple(shape))
            self.initial.append(initial[:1].detach().reshape(1, -1).clone())
            return initial[:1].clone()
        n = int(np.prod(shape))
        piece = self.state[:, self.offset:self.offset + n].reshape(1, *shape)
        self.offset += n
        return piece

    def put(self, tensor):
        self.new_state.append(tensor.reshape(1, -1))

    @property
    def size(self):
        return sum(int(np.prod(s)) for s in self.shapes)

    def initial_state(self):
        return torch.cat(self.initial, 1)

    def run(self, mod, x):
        kind = mod.original_name
        if kind in ("CachedSequential", "Sequential"):
            for _, child in mod.named_children():
                x = self.run(child, x)
            return x
        if kind in ("DilatedUnit", "Encoder", "EncoderV2", "UpsampleLayer", "ResidualBlock", "ResidualLayer"):
            return self.run(mod.net, x)
        if kind == "AlignBranches":
            return [self.run(branch, self.run(padding, x))
                    for branch, padding in zip(mod.branches.children(), mod.paddings.children())]
        if kind == "ResidualStack":
            branches = self.run(mod.net, x)
            out = branches[0]
            for branch in branches[1:]:
                out = out + branch
            return out
        if kind == "BatchNorm1d":
            return F.batch_norm(x, mod.running_mean, mod.running_var, mod.weight, mod.bias, False, 0.0, float(mod.eps))
        if kind == "NoiseGenerator":
            amp = mod_sigmoid(self.run(mod.net, x) - 5).permute(0, 2, 1)
            amp = amp.reshape(amp.shape[0], amp.shape[1], int(mod.data_size), -1)
            if id(mod) not in self.impulse:  # computed by the dry run, a constant when traced
                self.impulse[id(mod)] = impulse_response_matrix(int(amp.shape[-1]), int(mod.target_size))
            ir = amp @ self.impulse[id(mod)]
            if self.collect:
                self.synth_noise_shape = tuple(ir.shape)
                noise = torch.zeros_like(ir)
            else:
                noise = self.synth_noise
            noise = causal_filter(noise, ir).permute(0, 2, 1, 3)
            return noise.reshape(noise.shape[0], noise.shape[1], -1)
        if kind == "CachedPadding1d":
            padding = int(mod.padding)
            if padding == 0:
                return x
            pad = self.take((x.shape[1], padding), mod.pad)
            x = torch.cat([pad, x], -1)
            self.put(x[..., -padding:])
            return x[..., :-padding] if bool(mod.crop) else x
        if kind == "CachedConv1d":
            x = self.run(mod.downsampling_delay, x)
            x = self.run(mod.cache, x)
            stride, dilation, groups = conv_args(mod, False)
            return F.conv1d(x, mod.weight, mod.bias, stride, 0, dilation, groups)
        if kind == "CachedConvTranspose1d":
            stride, dilation = conv_args(mod, True)
            y = F.conv_transpose1d(x, mod.weight, None, stride, 0, 0, 1, dilation)
            c = int(mod.cache.shape[-1])
            cache = self.take((y.shape[1], c), mod.cache)
            head = y[..., :c] + cache
            self.put(y[..., -c:])
            y = torch.cat([head, y[..., c:]], -1)[..., :-c]
            return y + mod.bias.unsqueeze(-1) if mod.bias is not None else y
        if kind == "Snake":
            return x + torch.reciprocal(mod.alpha + 1e-9) * torch.sin(mod.alpha * x) ** 2
        if kind == "LeakyReLU":
            return F.leaky_relu(x, 0.2)
        if kind == "Identity":
            return x
        if kind == "AdaptiveInstanceNormalization":
            if id(mod) not in self.adain_active:
                self.adain_active[id(mod)] = bool(mod.num_update_x) and bool(mod.num_update_y)
            if self.adain_active[id(mod)]:
                raise RuntimeError("AdaIN style transfer is active; not supported")
            return x
        if kind == "Residual":
            net, res = self.run(mod.aligned, x)
            return net + res
        raise RuntimeError(f"unsupported module {kind}")


def reverse_half(x):  # rave.pqmf.reverse_half
    mask = torch.ones_like(x)
    mask[:, 1::2, ::2] = -1
    return x * mask


class StreamingRave(torch.nn.Module):
    def __init__(self, ts):
        super().__init__()
        self.ts = ts
        self.latent_size = int(ts.latent_size)
        self.full_latent_size = int(ts.full_latent_size)
        self.block = int(ts.encode_params[3])
        self.register_buffer("latent_pca", ts.latent_pca.detach().clone())
        self.register_buffer("latent_mean", ts.latent_mean.detach().clone())
        # One state slot per cache, in traversal order (filled by a dry run).
        self.enc, self.dec = Interpreter(), Interpreter()
        decoder = ts.decoder
        self.add_synth_noise = decoder.original_name == "Generator" and bool(decoder.use_noise) \
            and bool(decoder.warmed_up)
        with torch.no_grad():
            self.enc.begin(None, collect=True)
            self.encoder_raw(torch.zeros(1, 1, self.block))
            self.dec.begin(None, collect=True)
            self.decoder_raw(torch.zeros(1, self.full_latent_size, 1))

    # --- encode -----------------------------------------------------------------------
    def encoder_raw(self, x):
        pqmf = self.ts.pqmf
        if int(pqmf.n_band) > 1:
            x = reverse_half(self.enc.run(pqmf.forward_conv, x))
        return self.enc.run(self.ts.encoder.encoder.net, x)

    def encode(self, audio, state):
        self.enc.begin(state)
        z = self.encoder_raw(audio)
        mean = z[:, :z.shape[1] // 2]
        z = F.conv1d(mean - self.latent_mean.unsqueeze(-1), self.latent_pca.unsqueeze(-1))
        return z[:, :self.latent_size], self.enc.end()

    def forward_stream(self, audio, state, noise=None, synth_noise=None):
        z, enc_state = self.encode(audio, state[:, :self.enc.size])
        audio, dec_state = self.decode(z, state[:, self.enc.size:], noise, synth_noise)
        return audio, torch.cat([enc_state, dec_state], 1)

    @property
    def synth_noise_shape(self):
        """The v1 noise synthesis' noise per call, or None."""
        return self.dec.synth_noise_shape

    def draw_synth_noise(self):
        return torch.rand(self.synth_noise_shape) * 2 - 1 if self.synth_noise_shape else None

    def initial_state(self):
        return torch.cat([self.enc.initial_state(), self.dec.initial_state()], 1)

    # --- decode -----------------------------------------------------------------------
    def decoder_raw(self, z):
        decoder = self.ts.decoder
        x = self.dec.run(decoder.net, z)
        if decoder.original_name == "Generator":  # v1: waveform, loudness (, noise) branches
            branches = self.dec.run(decoder.synth, x)
            loudness = branches[1]
            if int(decoder.loud_stride) != 1:
                loudness = torch.repeat_interleave(loudness, int(decoder.loud_stride))
            y = torch.tanh(branches[0]) * mod_sigmoid(loudness.reshape(x.shape[0], 1, -1))
            if self.add_synth_noise:
                y = y + branches[2]
            return self.pqmf_inverse(y)
        if bool(decoder.amplitude_modulation):
            x, amplitude = torch.split(x, x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)
        return self.pqmf_inverse(torch.tanh(x))

    def pqmf_inverse(self, x):
        pqmf = self.ts.pqmf
        if int(pqmf.n_band) == 1:
            return x
        m = int(pqmf.hk.shape[0])
        x = self.dec.run(pqmf.inverse_conv, reverse_half(x)) * m
        x = torch.flip(x, [1]).permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1, m).permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def decode(self, z, state, noise=None, synth_noise=None):
        if noise is None:
            noise = torch.randn(1, self.full_latent_size - self.latent_size, z.shape[-1])
        if synth_noise is None:
            synth_noise = self.draw_synth_noise()
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1)) + self.latent_mean.unsqueeze(-1)
        self.dec.begin(state, synth_noise=synth_noise)
        audio = self.decoder_raw(z)
        return audio, self.dec.end()


class Encoder(torch.nn.Module):
    def __init__(self, rave): super().__init__(); self.rave = rave
    def forward(self, audio_in, state_in): return self.rave.encode(audio_in, state_in)


class Decoder(torch.nn.Module):
    def __init__(self, rave): super().__init__(); self.rave = rave
    def forward(self, latents, state_in, noise=None, synth_noise=None):
        return self.rave.decode(latents, state_in, noise, synth_noise)


class Forward(torch.nn.Module):
    def __init__(self, rave): super().__init__(); self.rave = rave
    def forward(self, audio_in, state_in, noise=None, synth_noise=None):
        return self.rave.forward_stream(audio_in, state_in, noise, synth_noise)


def check_against_torchscript(ts_path, blocks=40):
    """Streams the same audio, latents and noise through TorchScript's modules and the
    re-implementation: encoder (mean latents), decoder and forward (encoder then decoder)."""
    ref = torch.jit.load(ts_path).eval()
    ref_fwd = torch.jit.load(ts_path).eval()
    rave = StreamingRave(torch.jit.load(ts_path).eval()).eval()
    torch.manual_seed(0)
    L, N = rave.latent_size, rave.full_latent_size - rave.latent_size

    def ref_encode(ts, x):
        mean = ts.encoder(ts.pqmf(x)).chunk(2, 1)[0]
        return F.conv1d(mean - ts.latent_mean.unsqueeze(-1), ts.latent_pca.unsqueeze(-1))[:, :L]

    def ref_decode(ts, z, noise, seed):
        full = F.conv1d(torch.cat([z, noise], 1), ts.latent_pca.T.unsqueeze(-1)) + ts.latent_mean.unsqueeze(-1)
        torch.manual_seed(seed)  # a v1 noise synthesis draws torch.rand_like: the same as draw_synth_noise()
        return ts.pqmf.inverse(ts.decoder(full))

    enc_state, dec_state, fwd_state = rave.enc.initial_state(), rave.dec.initial_state(), rave.initial_state()
    enc_err = dec_err = fwd_err = 0.0
    with torch.no_grad():
        for block in range(blocks):
            audio = torch.randn(1, 1, rave.block) * 0.3
            z = torch.randn(1, L, 1) * 1.5
            noise = torch.randn(1, N, 1)
            torch.manual_seed(1000 + block)
            synth_noise = rave.draw_synth_noise()
            got, enc_state = rave.encode(audio, enc_state)
            enc_err = max(enc_err, float((got - ref_encode(ref, audio)).abs().max()))
            got, dec_state = rave.decode(z, dec_state, noise, synth_noise)
            dec_err = max(dec_err, float((got - ref_decode(ref, z, noise, 1000 + block)).abs().max()))
            got, fwd_state = rave.forward_stream(audio, fwd_state, noise, synth_noise)
            expected = ref_decode(ref_fwd, ref_encode(ref_fwd, audio), noise, 1000 + block)
            fwd_err = max(fwd_err, float((got - expected).abs().max()))
    return enc_err, dec_err, fwd_err


def export(module, args, path, input_names, output_names):
    with torch.no_grad():
        torch.onnx.export(module, args, path, input_names=input_names, output_names=output_names,
                          opset_version=17, dynamo=False)


def check_onnx(rave, module, make_inputs, names, blocks=30):
    """ONNX Runtime against PyTorch with identical noise (noise as extra inputs); the largest
    difference relative to 1 + the output's peak."""
    import onnxruntime as ort
    path = os.path.join(tempfile.mkdtemp(), "check.onnx")
    inputs = make_inputs()
    export(module, tuple(inputs), path, names, ["out", "state_out"])
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(path, options)
    state_t = inputs[1]
    state_o = state_t.numpy()
    worst = 0.0
    with torch.no_grad():
        for _ in range(blocks):
            fresh = make_inputs()
            out_t, state_t = module(fresh[0], state_t, *fresh[2:])
            feed = {n: t.numpy() for n, t in zip(names, fresh)}
            feed["state_in"] = state_o
            out_o, state_o = session.run(None, feed)
            # Scaled by the output's magnitude: latents reach tens, audio stays within 1.
            scale = 1.0 + float(np.abs(out_t.numpy()).max())
            worst = max(worst, float(np.abs(out_o - out_t.numpy()).max()) / scale)
    return worst


def write_decoder_reference(rave, path, blocks=32):
    """Fixed latents and noise streamed through the decoder from a zero state."""
    L, N, B = rave.latent_size, rave.full_latent_size - rave.latent_size, rave.block
    generator = torch.Generator().manual_seed(1)
    state = torch.zeros(1, rave.dec.size)
    inputs, audio = [], []
    with torch.no_grad():
        for _ in range(blocks):
            z = torch.randn(1, L, 1, generator=generator) * 1.5
            noise = torch.randn(1, N, 1, generator=generator)
            out, state = rave.decode(z, state, noise)
            inputs += [z.flatten(), noise.flatten()]
            audio.append(out.flatten())
    with open(path, "wb") as f:
        f.write(np.array([blocks, L, N, B], dtype=np.int32).tobytes())
        f.write(torch.cat(inputs).numpy().astype(np.float32).tobytes())
        f.write(torch.cat(audio).numpy().astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("prefix", help="output prefix: <prefix>_decoder.onnx, <prefix>_decoder_reference.bin")
    args = parser.parse_args()

    torch.manual_seed(0)
    ts = torch.jit.load(args.model).eval()
    rave = StreamingRave(ts).eval()
    L, B, E, D = rave.latent_size, rave.block, rave.enc.size, rave.dec.size
    N = rave.full_latent_size - L
    print(f"latents {L} (of {rave.full_latent_size}), block {B}, encoder state {E} floats "
          f"({len(rave.enc.shapes)} caches), decoder state {D} floats ({len(rave.dec.shapes)} caches)")

    errors = check_against_torchscript(args.model)
    print("streaming vs TorchScript: encoder {:.2e}, decoder {:.2e}, forward {:.2e}".format(*errors))
    if max(errors) > 1e-4:
        raise SystemExit("mismatch")

    for name, module, make_inputs, inputs, outputs in models(rave):
        onnx_err = check_onnx(rave, module, make_inputs, inputs)
        print(f"onnxruntime vs pytorch: {name} {onnx_err:.2e}")
        if onnx_err > 1e-4:
            raise SystemExit("onnx mismatch")
        export(module, tuple(torch.zeros_like(t) for t in make_inputs()), f"{args.prefix}_{name}.onnx", inputs, outputs)
    if rave.synth_noise_shape is None:  # its format has no synth noise
        write_decoder_reference(rave, f"{args.prefix}_decoder_reference.bin")
    print(f"ENCODER_STATE_SIZE={E} DECODER_STATE_SIZE={D} DECODER_NOISE_SIZE={N} SYNTH_NOISE={rave.synth_noise_shape}")


def models(rave):
    """(name, module, example inputs, input names, output names) of each export."""
    L, B, N = rave.latent_size, rave.block, rave.full_latent_size - rave.latent_size
    synth = [rave.draw_synth_noise()] if rave.synth_noise_shape else []
    synth_name = ["synth_noise"] if synth else []
    return [
        ("encoder", Encoder(rave).eval(), lambda: [torch.randn(1, 1, B) * 0.3, rave.enc.initial_state()],
         ["audio_in", "state_in"], ["latents", "state_out"]),
        ("decoder", Decoder(rave).eval(),
         lambda: [torch.randn(1, L, 1) * 1.5, rave.dec.initial_state(), torch.randn(1, N, 1), *(rave.draw_synth_noise() for _ in synth)],
         ["latents", "state_in", "noise", *synth_name], ["audio", "state_out"]),
        ("forward", Forward(rave).eval(),
         lambda: [torch.randn(1, 1, B) * 0.3, rave.initial_state(), torch.randn(1, N, 1), *(rave.draw_synth_noise() for _ in synth)],
         ["audio_in", "state_in", "noise", *synth_name], ["audio_out", "state_out"]),
    ]


if __name__ == "__main__":
    main()
