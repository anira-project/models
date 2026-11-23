# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's Nemotron 3 Diarization (transformers) as one streaming step with the tensor names of the
ONNX export in this repository (onnx-community's), for the formats with fixed shapes (Core ML, ExecuTorch):

    input_features [1,104,128]   one chunk: 9 encoder frames scored and 4 look-ahead (transformers'
                                 low-latency streaming mode), 8 mel frames each
    cached_embeds [1,192,512]    the speaker cache and FIFO the chunk attends to (anira models'
                                 SpeakerCache: 128 + 64 frames), padded at the front while they fill
    attention_mask [1,205]       1 for every real frame, cached and chunk, 0 for the padding
 -> logits [1,1640,8]            speaker logits of every frame, 8 per encoder frame
    chunk_embeds [1,13,512]      the chunk's encoder input frames, for the FIFO
    silence_embeds [512]         the learned silence frame of the cache

The state stays outside, as in the ONNX export: the caller keeps the speaker cache and FIFO
(transformers' Nemotron3DiarizationSpeakerCache, SpeakerCache in anira models) and feeds them
back as cached_embeds. Padding at the front leaves the real frames' rotary positions apart as they are
(attention depends on their distance only); the masked frames do not take part, and are zeroed before
the head's 3-frame upsampling convolution. The attention_mask is
int64 as in the ONNX export (int32 in Core ML).
"""

import copy

import numpy as np
import soundfile
import torch

CHUNK, LOOKAHEAD, SUBSAMPLING = 9, 4, 8
MEL_FRAMES = (CHUNK + LOOKAHEAD) * SUBSAMPLING
CACHE, FIFO, UPDATE_PERIOD = 128, 64, 53  # SpeakerCacheConfig's defaults
CACHED = CACHE + FIFO
FRAMES = CACHED + CHUNK + LOOKAHEAD
INPUTS = ["input_features", "cached_embeds", "attention_mask"]
OUTPUTS = ["logits", "chunk_embeds", "silence_embeds"]
REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"  # nvidia/Nemotron-3-Diarization


def load():
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    torch.set_grad_enabled(False)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "nvidia/Nemotron-3-Diarization", revision=REVISION, attn_implementation="eager").eval()
    return model, AutoProcessor.from_pretrained("nvidia/Nemotron-3-Diarization", revision=REVISION)


class Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, cached_embeds, attention_mask):
        model = self.model
        chunk_embeds = model.model.audio_tower.embedder(input_features)
        embeds = torch.cat([cached_embeds, chunk_embeds], dim=1)
        position_ids = torch.arange(embeds.shape[1])[None, :]
        tower = model.model.audio_tower(inputs_embeds=embeds, attention_mask=attention_mask.bool(),
                                        position_ids=position_ids).last_hidden_state
        # The head's upsampler convolves 3 frames: the padding frames are zeroed after the projection, as
        # the convolution's own zero padding is where transformers' input starts.
        projected = model.model.proj(tower) * attention_mask[..., None].to(tower.dtype)
        logits = model.classifier(model.model.upsampler(projected))
        # silence_embeds computed: Core ML has no output that is a constant.
        silence = model.silence_embeds + 0 * input_features[0, 0, 0]
        return logits, chunk_embeds, silence


def example():
    return (torch.zeros(1, MEL_FRAMES, 128), torch.zeros(1, CACHED, 512), torch.ones(1, FRAMES, dtype=torch.int64))


def speaker_cache(model):
    """transformers' speaker cache with anira models' sizes."""
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import Nemotron3DiarizationSpeakerCache
    config = copy.deepcopy(model.config.streaming_config)
    config.speaker_cache_length = CACHE
    return Nemotron3DiarizationSpeakerCache(config, fifo_length=FIFO, speaker_cache_update_period=UPDATE_PERIOD)


def features(processor, path):
    """The recording's mel frames [1, frames, 128] (the per-chunk streaming features match them frame for frame)."""
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == 16000 and audio.ndim == 1
    return processor(audio, sampling_rate=rate, return_tensors="pt")["input_features"]


def chunks(mel):
    """The full streaming chunks of a recording: chunk k holds mel frames [72 k, 72 k + 104)."""
    step = CHUNK * SUBSAMPLING
    return [mel[:, k:k + MEL_FRAMES] for k in range(0, mel.shape[1] - MEL_FRAMES + 1, step)]


def stream(run, model, mel_chunks):
    """Streams the chunks through one format's step, run(input_features, cached_embeds, attention_mask) ->
    (logits, chunk_embeds, silence_embeds) as arrays, keeping the speaker cache outside as the ONNX
    export's callers do. Returns the scored logits of every chunk [1, 72, 8]."""
    cache, scored = speaker_cache(model), []
    for chunk in mel_chunks:
        real = cache.get_embeds(model.model.audio_tower.embedder(chunk))  # [1, n, 512]
        n = real.shape[1]
        cached = torch.zeros(1, CACHED, 512)
        cached[:, CACHED - n:] = real
        mask = torch.zeros(1, FRAMES, dtype=torch.int64)
        mask[:, CACHED - n:] = 1
        logits, chunk_embeds, silence = (torch.as_tensor(np.asarray(x)) for x in run(np.ascontiguousarray(chunk.numpy()), cached.numpy(), mask.numpy()))
        logits = logits[:, (CACHED - n) * SUBSAMPLING:]  # the real frames
        cache.update(torch.cat([real, chunk_embeds], dim=1), logits, silence, CHUNK)
        scored.append(logits[:, n * SUBSAMPLING:(n + CHUNK) * SUBSAMPLING])
    return scored


def reference(model, mel_chunks):
    """transformers' own streaming forward, chunk by chunk with the same cache sizes."""
    cache, scored = speaker_cache(model), []
    for chunk in mel_chunks:
        scored.append(model(input_features=chunk, speaker_cache=cache, num_lookahead_frames=LOOKAHEAD).logits)
    return scored
