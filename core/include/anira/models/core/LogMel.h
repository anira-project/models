// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace anira::models::core {

/// NeMo's log-mel front end (FilterbankFeatures; transformers'
/// NemotronAsrStreamingFeatureExtractor): 16 kHz, pre-emphasis 0.97, a 400-sample periodic Hann
/// window zero-padded to a 512-point FFT, hop 160 (10 ms), power spectrum, 128 Slaney-normalised
/// mel bands (0 to 8 kHz), log(x + 2^-24), no per-feature normalisation. Frame k is centred on
/// sample k * 160 (torch.stft's center=True, the left half window zeros instead of reflected: only
/// the first two frames differ). Streaming: process() takes any block and returns the frames it
/// completed. Used by the Nemotron models.
class LogMel {
public:
    static constexpr std::size_t k_mels = 128;
    static constexpr std::size_t k_fft = 512;
    static constexpr std::size_t k_window = 400;
    static constexpr std::size_t k_hop = 160;
    static constexpr float k_preemphasis = 0.97F;

    LogMel();
    /// Non-real-time: blocks of up to `max_block` samples.
    void prepare(std::size_t max_block);
    /// A new stream.
    void reset() noexcept;
    /// Real-time: `samples` (16 kHz, at most max_block) in; the completed frames, k_mels values
    /// each (frame-major), written to `frames` (room for max_frames(n) frames). Returns their
    /// number.
    std::size_t process(std::span<const float> samples, std::span<float> frames) noexcept;
    /// The most frames a block of `samples` completes.
    [[nodiscard]] static std::size_t max_frames(std::size_t samples) noexcept {
        return (samples / k_hop) + 1;
    }

private:
    void frame(const float* samples, float* out) noexcept;

    std::vector<float> m_window;   // k_fft: Hann(400) centred by zero-padding
    std::vector<float> m_melbank;  // k_mels x (k_fft / 2 + 1)
    std::vector<float> m_pending;  // pre-emphasised samples not yet consumed
    std::size_t m_pending_size = 0;
    std::vector<float> m_re;
    std::vector<float> m_im;
    std::vector<float> m_power;
    float m_previous = 0.0F;
    bool m_started = false;
};

}  // namespace anira::models::core
