// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/core/Vocabulary.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace anira::models::parakeet_tdt {

using core::Vocabulary;

inline constexpr double k_sample_rate = 16000.0;

/// The windows the preprocessor and encoder run at (anira fixes a model's shapes at prepare): a
/// segment runs in the smallest that holds it, zero-padded, its length masking the padding.
struct Window {
    std::size_t m_seconds;
    std::size_t m_samples;  ///< 16 kHz
    std::size_t m_mel_frames;
    std::size_t m_encoder_frames;  ///< 80 ms each
};
inline constexpr std::array<Window, 4> k_windows{
    {{.m_seconds = 4, .m_samples = 64000, .m_mel_frames = 401, .m_encoder_frames = 51},
     {.m_seconds = 8, .m_samples = 128000, .m_mel_frames = 801, .m_encoder_frames = 101},
     {.m_seconds = 16, .m_samples = 256000, .m_mel_frames = 1601, .m_encoder_frames = 201},
     {.m_seconds = 31, .m_samples = 496000, .m_mel_frames = 3101, .m_encoder_frames = 388}}};
/// The longest segment: the largest window.
inline constexpr std::size_t k_max_samples = k_windows.back().m_samples;

/// NVIDIA's Parakeet TDT 0.6B v3: accurate speech to text of a segment (25 European languages),
/// with the time of every word. Job-shaped: the segment's audio through the preprocessor and the
/// encoder, then the TDT search one step at a time (core::AsyncHandler, anira's Async contract
/// from M3). transcribe() waits for every step: call it from a thread of your own, never the
/// audio thread (a voice activity detector's segments, e.g. silero_vad::Vad, are its input).
class Parakeet {
public:
    Parakeet(anira::Context& context, const core::ModelSource& weights);
    ~Parakeet();
    Parakeet(const Parakeet&) = delete;
    Parakeet& operator=(const Parakeet&) = delete;
    Parakeet(Parakeet&&) = delete;
    Parakeet& operator=(Parakeet&&) = delete;

    /// Main thread: every window's and the decoder's contract.
    void prepare();

    /// One segment of 16 kHz audio, at most k_max_samples: its final text and words, positions in
    /// 16 kHz samples from `start_sample` on. Throws Error for a segment too long or a failed job.
    core::Segment transcribe(std::span<const float> audio,
                             std::int64_t start_sample = 0,
                             std::uint64_t id = 0);

    /// A recording at any rate, in segments of at most 31 s (cut there, without regard to words):
    /// the final segments, positions in the recording's samples.
    static std::vector<core::Segment> run(anira::Context& context,
                                          const core::ModelSource& weights,
                                          std::span<const float> mono,
                                          double sample_rate);

    [[nodiscard]] const Vocabulary& vocabulary() const noexcept { return m_vocabulary; }

private:
    class Joint;

    Vocabulary m_vocabulary;
    std::array<std::unique_ptr<core::AsyncHandler>, k_windows.size()> m_preprocessors;
    std::array<std::unique_ptr<core::AsyncHandler>, k_windows.size()> m_encoders;
    std::unique_ptr<Joint> m_joint;

    std::vector<float> m_waveform = std::vector<float>(k_max_samples);
    std::int64_t m_waveform_length = 0;
    std::vector<float> m_features;
    std::int64_t m_features_length = 0;
    std::vector<float> m_encoded;
    std::int64_t m_encoded_length = 0;
};

}  // namespace anira::models::parakeet_tdt
