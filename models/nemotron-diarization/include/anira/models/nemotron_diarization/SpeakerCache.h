// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace anira::models::nemotron_diarization {

/// Sizes and the frame-scoring policy of the speaker cache (Nemotron 3 Diarization's
/// `streaming_config`, with smaller sizes for CPU: every chunk attends to all cached frames).
struct SpeakerCacheConfig {
    std::size_t m_cache_frames = 128;  ///< Arrival-Order Speaker Cache capacity (encoder frames)
    std::size_t m_fifo_frames = 64;    ///< FIFO of the most recent frames
    std::size_t m_update_period = 53;  ///< frames moved to the cache when the FIFO overflows
    std::size_t m_silence_frames_per_speaker = 1;
    float m_prediction_score_threshold = 0.25f;
    float m_latest_frames_score_boost = 0.05f;
    float m_min_positive_scores_rate = 0.5f;
    float m_strong_boost_rate = 0.75f;
    float m_weak_boost_rate = 1.5f;
};

/// The streaming state of Nemotron 3 Diarization (Streaming Sortformer): the Arrival-Order
/// Speaker Cache and the FIFO of the most recent encoder frames, which every chunk attends to
/// ahead of its own frames. A port of transformers' Nemotron3DiarizationSpeakerCache for one
/// stream. Internal to the diarization engine.
class SpeakerCache {
public:
    static constexpr std::size_t k_speakers = 8;
    static constexpr std::size_t k_hidden = 512;
    static constexpr std::size_t k_subsampling = 8;  ///< logit frames per encoder frame

    explicit SpeakerCache(SpeakerCacheConfig config = {});

    /// The frames the next chunk attends to: the cache, then the FIFO; [frames][k_hidden].
    [[nodiscard]] std::vector<float> embeds() const;
    [[nodiscard]] std::size_t frames() const noexcept { return m_cache_frames + m_fifo_frames; }
    [[nodiscard]] std::size_t cache_frames() const noexcept { return m_cache_frames; }
    [[nodiscard]] std::size_t fifo_frames() const noexcept { return m_fifo_frames; }

    /// After a step: `input_embeds` is what the model saw ([frames()][hidden] cached, then the
    /// chunk's own and look-ahead frames), `logits` its speaker logits for all of them
    /// ([input frames * k_subsampling][k_speakers]). The first `chunk_frames` chunk frames join
    /// the FIFO; when it overflows, its oldest frames move to the cache, which is compressed
    /// to its capacity when it overflows in turn.
    void update(std::span<const float> input_embeds,
                std::span<const float> logits,
                std::span<const float> silence_embeds,
                std::size_t chunk_frames);

    void reset();

private:
    using Rows = std::vector<float>;
    void compress(Rows& embeds, Rows& probs, std::span<const float> silence_embeds) const;
    [[nodiscard]] Rows frame_scores(const Rows& probs, std::size_t frames) const;

    SpeakerCacheConfig m_config;
    std::size_t m_min_positive_scores = 0;
    std::size_t m_strong_boosted = 0;
    std::size_t m_weak_boosted = 0;

    Rows m_cache;        // [m_cache_frames][k_hidden]
    Rows m_cache_probs;  // [m_cache_frames][k_speakers]
    Rows m_fifo;         // [m_fifo_frames][k_hidden]
    std::size_t m_cache_frames = 0;
    std::size_t m_fifo_frames = 0;
    bool m_compressed = false;
};

}  // namespace anira::models::nemotron_diarization
