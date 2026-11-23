# SPDX-License-Identifier: Apache-2.0
# Moonshine streaming with every tensor at a fixed size, built from Moonshine AI's checkpoints in
# transformers. Two graphs per model:
#
#   encoder: 0.5 s of audio and the frontend and encoder state in, the cross-attention keys and values
#            of the 25 encoder frames that became final out (frontend, sliding-window encoder, adapter
#            and the decoder's cross-attention projections in one step)
#   decoder: one token, its position, the self-attention cache, the cross-attention keys and values of
#            the segment and its length in, the logits and the token's keys and values out
#
# The sizes are the limits Moonshine's own runtime works within (moonshine-ai/moonshine, core/):
# a segment is at most 15 s plus the voice activity detector's 0.512 s look-behind (800 frames at
# 50 Hz), decoding emits at most 6.5 tokens a second (128 positions), the encoder sees 16 frames of
# left context per layer and 16 frames of lookahead, and it is updated every 0.5 s.
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, MoonshineStreamingForConditionalGeneration

MODELS = {
    "tiny-en": ("moonshine-ai/moonshine-streaming-tiny", "f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a"),
    "small-en": ("moonshine-ai/moonshine-streaming-small", "2c036506f23a09c18df5a50057599ba6d9280999"),
    "tiny-de": ("moonshine-ai/moonshine-streaming-tiny-de", "928bfb925992902b7e187fda7921d72114142e1f"),
    "small-de": ("moonshine-ai/moonshine-streaming-small-de", "56473ba32b22b484fdfc7eb03d34c393ff494197"),
}

SAMPLE_RATE = 16000
FRAME = 320       # samples per encoder frame (50 Hz)
CHUNK = 25        # encoder frames per step (0.5 s)
LOOKAHEAD = 16    # frames an encoder frame waits for
MEMORY = 800      # encoder frames per segment (16 s)
TOKENS = 128      # decoder positions per segment
BOS, EOS = 1, 2
TOKENS_PER_SECOND = 6.5
MASKED = -1e4


def load(variant):
    repo, revision = MODELS[variant]
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(repo, revision=revision,
                                                                       attn_implementation="eager")
    return model.eval(), AutoTokenizer.from_pretrained(repo, revision=revision)


def windows(model):
    # Per encoder layer, the frames a frame sees behind and ahead of it, as Moonshine's runtime graphs have
    # them: both bounds inclusive, (16, 4) sees 16 frames behind and 4 ahead (moonshine_voice/lora/export.py;
    # transformers reads the same config one frame narrower on each side). The German checkpoints' configs
    # write these windows in transformers' reading, (17, 5) and (17, 1); their runtime graphs use (16, 4) and
    # (16, 0), the same as the English ones.
    configured = [tuple(w) for w in model.config.encoder_config.sliding_windows]
    if all(behind == 17 for behind, _ in configured):
        return [(behind - 1, ahead - 1) for behind, ahead in configured]
    return configured


def sizes(model):
    enc, dec = model.config.encoder_config, model.config
    reach = windows(model)
    assert sum(ahead for _, ahead in reach) == LOOKAHEAD
    return dict(context=sum(behind for behind, _ in reach), encoder_dim=enc.hidden_size, layers=dec.num_hidden_layers,
                heads=dec.num_key_value_heads, head_dim=dec.head_dim, vocab=dec.vocab_size)


# The graphs are written for XNNPACK (ExecuTorch's CPU backend), which has no layer_norm, no mean over the last
# axis and no 4-D batched matmul: each op it lacks splits a graph and copies the tensors in between (the
# decoder's cross-attention memory on every layer). So the means of LayerNorm are matrix products, attention
# runs as 3-D batched matmuls on slices of the caches and rotary embedding as a matrix product. The other
# formats run the same graphs.
def layer_norm(x, weight, eps, average):
    # eps inside the mean (the same, average sums to 1): coremltools' fuse_linear_bias fails on a constant
    # added after the matmul.
    centered = x - torch.matmul(x, average)
    return centered * torch.rsqrt(torch.matmul(centered * centered + eps, average)) * weight


def attend(attention, q, k, v, mask):
    """q [H,Q,K], k and v [H,S,K], mask [1,Q,S] -> attention.o_proj of the attention, [1,Q,H*K]."""
    w = nn.functional.softmax(torch.bmm(q, k.transpose(1, 2)) * attention.scaling + mask, dim=-1)
    out = torch.bmm(w, v)
    return attention.o_proj(out.transpose(0, 1).reshape(1, q.shape[1], -1))


def heads_of(projection, x, head_dim):
    """[1,Q,H*K] -> [H,Q,K]"""
    y = projection(x)
    return y.reshape(y.shape[1], -1, head_dim).transpose(0, 1)


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder
        s = sizes(model)
        self.context, self.heads, self.head_dim = s["context"], s["heads"], s["head_dim"]
        self.window = self.context + CHUNK + LOOKAHEAD
        self.reach = windows(model)
        offset = torch.arange(self.window)
        self.register_buffer("distance", offset[:, None] - offset[None, :])
        self.register_buffer("slot", offset)
        self.register_buffer("step", torch.arange(CHUNK))
        width = s["encoder_dim"]
        self.register_buffer("average", torch.full((width, 1), 1.0 / width))

    def norm(self, ln, x):  # MoonshineStreamingLayerNorm
        return layer_norm(x, ln.gamma + ln.unit_offset, ln.ln.eps, self.average)

    @staticmethod
    def causal_conv(conv, x, buffer):
        x = torch.cat([buffer, x], dim=-1)
        k = conv.kernel_size[0] - 1
        return nn.functional.conv1d(x, conv.weight, conv.bias, stride=conv.stride), x[..., -k:]

    def forward(self, audio, conv1_buffer, conv2_buffer, features, position, length):
        embedder = self.encoder.embedder
        x = embedder.cmvn(audio.reshape(1, -1, embedder.frame_len))
        # The asinh compression written out: Core ML has no asinh.
        x = torch.exp(embedder.comp.log_k) * x
        x = torch.sign(x) * torch.log(torch.abs(x) + torch.sqrt(x * x + 1))
        x = nn.functional.silu(embedder.linear(x)).transpose(1, 2)
        x, conv1_out = self.causal_conv(embedder.conv1, x, conv1_buffer)
        x, conv2_out = self.causal_conv(embedder.conv2, nn.functional.silu(x), conv2_buffer)
        h = torch.cat([features, x.transpose(1, 2)], dim=1)
        features_out = h[:, CHUNK:]

        # Frame index of each slot in the segment; frames before the segment and after the audio so far
        # are masked, every frame keeps itself so that no row is empty.
        frame = self.slot + (position - self.context)
        # (logical_and / logical_or: coremltools has no Python & and | on tensors)
        valid = torch.logical_and(frame >= 0, frame < length)
        for layer, (behind, ahead) in zip(self.encoder.layers, self.reach):
            seen = torch.logical_and(self.distance >= -ahead, self.distance <= behind)
            seen = torch.logical_or(torch.logical_and(seen, valid[None, :]), self.distance == 0)
            mask = torch.where(seen, 0.0, MASKED)[None]
            a = layer.self_attn
            x = self.norm(layer.input_layernorm, h)
            h = h + attend(a, heads_of(a.q_proj, x, a.head_dim), heads_of(a.k_proj, x, a.head_dim),
                           heads_of(a.v_proj, x, a.head_dim), mask)
            h = h + layer.mlp(self.norm(layer.post_attention_layernorm, h))
        h = self.norm(self.encoder.final_norm, h)[:, self.context:self.context + CHUNK]

        index = torch.clamp(self.step + position, 0, self.decoder.pos_emb.num_embeddings - 1)
        memory = self.decoder.proj(h + self.decoder.pos_emb(index)[None])
        keys, values = [], []
        for layer in self.decoder.layers:
            attention = layer.encoder_attn
            keys.append(heads_of(attention.k_proj, memory, self.head_dim)[None])
            values.append(heads_of(attention.v_proj, memory, self.head_dim)[None])
        return torch.stack(keys), torch.stack(values), conv1_out, conv2_out, features_out


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.head = model.proj_out
        s = sizes(model)
        self.heads, self.head_dim = s["heads"], s["head_dim"]
        self.register_buffer("cache_slot", torch.arange(TOKENS))
        self.register_buffer("memory_slot", torch.arange(MEMORY))
        width = model.config.hidden_size
        self.register_buffer("average", torch.full((width, 1), 1.0 / width))
        # Rotary embedding on the first `turned` dimensions of each head, pairs (2j, 2j+1) rotated
        # (transformers' rotate_half and apply_rotary_pos_emb).
        inv_freq = self.decoder.rotary_emb.inv_freq
        turned = 2 * inv_freq.numel()
        self.register_buffer("inv_freq", inv_freq.clone())
        self.register_buffer("pair", torch.clamp(torch.arange(self.head_dim) // 2, max=inv_freq.numel() - 1))
        self.register_buffer("turned", (torch.arange(self.head_dim) < turned).float())
        rotation = torch.zeros(self.head_dim, self.head_dim)
        for j in range(turned // 2):
            rotation[2 * j + 1, 2 * j] = -1.0
            rotation[2 * j, 2 * j + 1] = 1.0
        self.register_buffer("rotation", rotation)

    def norm(self, ln, x):  # nn.LayerNorm without bias
        return layer_norm(x, ln.weight, ln.eps, self.average)

    def forward(self, token, position, k_self, v_self, k_cross, v_cross, memory_length):
        H, K = self.heads, self.head_dim
        h = self.decoder.embed_tokens(token)
        angle = position.float() * self.inv_freq
        cos = torch.cos(angle)[self.pair] * self.turned + (1 - self.turned)
        sin = torch.sin(angle)[self.pair] * self.turned
        here = (self.cache_slot == position).to(h.dtype)[None, :, None]
        self_mask = torch.where(self.cache_slot <= position, 0.0, MASKED)[None, None]
        cross_mask = torch.where(self.memory_slot < memory_length, 0.0, MASKED)[None, None]
        new_keys, new_values = [], []
        for i, layer in enumerate(self.decoder.layers):
            a = layer.self_attn
            x = self.norm(layer.input_layernorm, h)
            q, k, v = (heads_of(p, x, K) for p in (a.q_proj, a.k_proj, a.v_proj))
            q = q * cos + torch.matmul(q, self.rotation) * sin
            k = k * cos + torch.matmul(k, self.rotation) * sin
            new_keys.append(k[None])
            new_values.append(v[None])
            keys = k_self.narrow(0, i, 1).reshape(H, TOKENS, K) * (1 - here) + k * here
            values = v_self.narrow(0, i, 1).reshape(H, TOKENS, K) * (1 - here) + v * here
            h = h + attend(a, q, keys, values, self_mask)

            c = layer.encoder_attn
            q = heads_of(c.q_proj, self.norm(layer.post_attention_layernorm, h), K)
            h = h + attend(c, q, k_cross.narrow(0, i, 1).reshape(H, MEMORY, K),
                           v_cross.narrow(0, i, 1).reshape(H, MEMORY, K), cross_mask)

            mlp = layer.mlp  # fc1, then the gated half (transformers' chunk(2) as two slices)
            x = mlp.fc1(self.norm(layer.final_layernorm, h))
            half = x.shape[-1] // 2
            h = h + mlp.fc2(mlp.activation_fn(x.narrow(-1, half, half)) * x.narrow(-1, 0, half))
        logits = self.head(self.norm(self.decoder.norm, h))[:, 0]
        return logits, torch.stack(new_keys), torch.stack(new_values)


NAMES = {
    "encoder": (["audio", "conv1_buffer", "conv2_buffer", "features", "position", "length"],
                ["k_cross", "v_cross", "conv1_buffer_out", "conv2_buffer_out", "features_out"]),
    "decoder": (["token", "position", "k_self", "v_self", "k_cross", "v_cross", "memory_length"],
                ["logits", "k_new", "v_new"]),
}


def examples(model):
    s = sizes(model)
    d, layers, heads, head_dim = s["encoder_dim"], s["layers"], s["heads"], s["head_dim"]
    i32 = lambda v: torch.tensor([v], dtype=torch.int32)
    encoder = (torch.zeros(1, CHUNK * FRAME), torch.zeros(1, d, 4), torch.zeros(1, 2 * d, 4),
               torch.zeros(1, s["context"] + LOOKAHEAD, d), i32(0), i32(0))
    decoder = (torch.tensor([[BOS]], dtype=torch.int32), i32(0), torch.zeros(layers, 1, heads, TOKENS, head_dim),
               torch.zeros(layers, 1, heads, TOKENS, head_dim), torch.zeros(layers, 1, heads, MEMORY, head_dim),
               torch.zeros(layers, 1, heads, MEMORY, head_dim), i32(0))
    return {"encoder": encoder, "decoder": decoder}


def audio(path):
    import soundfile

    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == SAMPLE_RATE and samples.ndim == 1
    return samples


class Segment:
    """One segment through the two graphs, as the C++ wrapper drives them. run(name, *inputs) -> outputs
    as numpy arrays."""

    def __init__(self, model, run):
        s = sizes(model)
        self.s, self.run = s, run
        d, layers, heads, head_dim = s["encoder_dim"], s["layers"], s["heads"], s["head_dim"]
        self.conv1 = np.zeros((1, d, 4), np.float32)
        self.conv2 = np.zeros((1, 2 * d, 4), np.float32)
        self.features = np.zeros((1, s["context"] + LOOKAHEAD, d), np.float32)
        self.k_cross = np.zeros((layers, 1, heads, MEMORY, head_dim), np.float32)
        self.v_cross = np.zeros_like(self.k_cross)
        self.frames = 0   # encoder frames computed from audio
        self.memory = 0   # frames in the cross-attention memory

    def push(self, chunk):
        """0.5 s of audio, or None at the end of the segment (flushes the lookahead)."""
        if chunk is not None:
            self.frames += CHUNK
        position = self.frames - (CHUNK if chunk is not None else 0) - LOOKAHEAD
        samples = np.zeros((1, CHUNK * FRAME), np.float32) if chunk is None else chunk.reshape(1, -1)
        i32 = lambda v: np.array([v], np.int32)
        k, v, self.conv1, self.conv2, self.features = self.run(
            "encoder", samples, self.conv1, self.conv2, self.features, i32(position), i32(self.frames))
        for j in range(CHUNK):
            frame = position + j
            if 0 <= frame < self.frames and frame == self.memory and frame < MEMORY:
                self.k_cross[:, :, :, frame] = k[:, :, :, j]
                self.v_cross[:, :, :, frame] = v[:, :, :, j]
                self.memory += 1

    def decode(self, seconds):
        s = self.s
        k_self = np.zeros((s["layers"], 1, s["heads"], TOKENS, s["head_dim"]), np.float32)
        v_self = np.zeros_like(k_self)
        limit = min(int(np.ceil(seconds * TOKENS_PER_SECOND)), TOKENS - 1)
        token, tokens = BOS, []
        i32 = lambda v: np.array([v], np.int32)
        for position in range(limit):
            logits, k, v = self.run("decoder", np.array([[token]], np.int32), i32(position), k_self, v_self,
                                    self.k_cross, self.v_cross, i32(self.memory))
            k_self[:, :, :, position] = k[:, :, :, 0]
            v_self[:, :, :, position] = v[:, :, :, 0]
            token = int(np.argmax(logits[0]))
            if token == EOS:
                break
            tokens.append(token)
        return tokens


def stream(model, run, samples, decode_every=False):
    """A segment through the two graphs: 0.5 s steps, a flush at the end, then greedy decoding (also after
    every step with decode_every, as the live transcript does). Returns the final tokens and the memory."""
    padded = np.concatenate([samples, np.zeros(-len(samples) % (CHUNK * FRAME), np.float32)])
    segment = Segment(model, run)
    partial = []
    for start in range(0, len(padded), CHUNK * FRAME):
        segment.push(padded[start:start + CHUNK * FRAME])
        if decode_every:
            partial.append(segment.decode(segment.memory * FRAME / SAMPLE_RATE))
    segment.push(None)
    return segment.decode(len(padded) / SAMPLE_RATE), segment, partial


def torch_run(encoder, decoder):
    graphs = {"encoder": encoder, "decoder": decoder}

    def run(name, *inputs):
        with torch.no_grad():
            return [o.numpy() for o in graphs[name](*[torch.from_numpy(np.asarray(i)) for i in inputs])]

    return run


def encode(model, values):
    """transformers' encoder on a whole segment with the runtime's windows."""
    encoder = model.model.encoder
    h, _ = encoder.embedder(values)
    offset = torch.arange(h.shape[1])
    distance = offset[:, None] - offset[None, :]
    for layer, (behind, ahead) in zip(encoder.layers, windows(model)):
        seen = (distance >= -ahead) & (distance <= behind)
        h = layer(h, attention_mask=torch.where(seen, 0.0, torch.finfo(h.dtype).min)[None, None])
    return encoder.final_norm(h)


def reference(model, samples):
    """transformers on the whole segment, padded like stream(): the encoder output and the greedy tokens."""
    from transformers.models.moonshine_streaming.modeling_moonshine_streaming import \
        MoonshineStreamingEncoderModelOutput as Output

    padded = np.concatenate([samples, np.zeros(-len(samples) % (CHUNK * FRAME), np.float32)])
    values = torch.from_numpy(padded)[None]
    seconds = len(padded) / SAMPLE_RATE
    limit = min(int(np.ceil(seconds * TOKENS_PER_SECOND)), TOKENS - 1)
    with torch.no_grad():
        encoded = encode(model, values)
        out = model.generate(encoder_outputs=Output(last_hidden_state=encoded.clone()),
                             decoder_input_ids=torch.tensor([[BOS]]), max_new_tokens=limit, do_sample=False,
                             num_beams=1)
    tokens = [t for t in out[0].tolist()[1:] if t != EOS]
    return encoded, tokens
