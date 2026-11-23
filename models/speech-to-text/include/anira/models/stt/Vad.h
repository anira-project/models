// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/abi/status.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/SpscQueue.h>
#include <anira/models/stt/VadSegmenter.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::stt {

/// Silero VAD: the probability that someone speaks, one value per 32 ms (512 samples at 16 kHz),
/// and speech segments from it (VadSegmenter). A detector: it never changes the audio.
///
///   - For DSP: process(in, probability) writes the probability of every input sample, delayed by
///     get_latency() host samples (probability[i] belongs to the input get_latency() samples
///     earlier); delay your own audio by as much (core::Delay, or the host's latency compensation).
///   - For a UI: pop(Frame&) gives every probability with its host position, latest() the newest.
///   - For transcription: pop(SpeechSpan&) gives the segments, in host samples since reset().
///   - Offline: run(), or set_non_realtime(true) for a bounce faster than real time.
class Vad : public core::AudioModel {
public:
    Vad(anira::Context& context, const core::ModelSource& weights, VadConfig config = {});

    /// Online, on the message thread with no process() running: Silero's contract with the host's
    /// block range and rate.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Online with every contract field the caller's.
    anira::PlanReport prepare(const anira::Hard& host);
    /// A new stream: the clock, the model's state, the segments and the queues start over.
    void reset();

    /// Audio thread. `in`: [channels, n] or [n] float host audio (downmixed); `probability`, when
    /// given: [n] or [1, n] float, written. ANIRA_MISSED when anira filled a late value (zero).
    anira_status process(const anira::Tensor& in,
                         const anira::Tensor& probability = {},
                         std::size_t* delivered = nullptr) noexcept;
    anira_status process(std::span<const float> mono, std::span<float> probability = {}) noexcept;

    /// Reader thread: the oldest segment not taken yet, positions in host samples.
    bool pop(SpeechSpan& span) noexcept { return m_spans.pop(span); }
    /// Reader thread: the oldest probability not taken yet.
    bool pop(core::Frame& frame) noexcept { return m_frames.pop(frame); }
    /// Any thread: the newest probability.
    [[nodiscard]] core::Frame latest() const noexcept {
        return {.m_position = m_latest_position.load(std::memory_order_relaxed),
                .m_value = m_latest_value.load(std::memory_order_relaxed)};
    }

    /// Every segment of a recording, the same values as streaming it (anira waits for each one).
    static std::vector<SpeechSpan> run(anira::Context& context,
                                       const core::ModelSource& weights,
                                       std::span<const float> mono,
                                       double sample_rate,
                                       VadConfig config = {});

private:
    void size_buffers();
    anira_status pop_frames(std::size_t frames_due) noexcept;
    void write_aligned(std::int64_t block_start, float* out, std::size_t samples) noexcept;
    [[nodiscard]] SpeechSpan to_host(const SpeechSpan& span) const noexcept;

    VadConfig m_config;
    VadSegmenter m_segmenter;
    core::SpscQueue<SpeechSpan> m_spans;
    core::SpscQueue<core::Frame> m_frames;
    std::atomic<std::int64_t> m_latest_position{0};
    std::atomic<float> m_latest_value{0.0f};

    float m_value = 0.0f;  // anira writes the popped probability here
    std::array<float*, 1> m_value_plane{&m_value};
    anira::Tensor m_value_tensor{};
    std::vector<float> m_values;      // the last frames' probabilities, by frame index
    std::int64_t m_popped = 0;        // frames popped since reset(), anira's priming included
    std::int64_t m_known = 0;         // frames whose probability is known
    std::int64_t m_frame_limit = -1;  // offline: frames past the recording reach no segment
};

}  // namespace anira::models::stt
