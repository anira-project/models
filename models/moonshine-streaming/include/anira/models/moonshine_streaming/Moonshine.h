// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/Latest.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/core/SpscQueue.h>
#include <anira/models/moonshine_streaming/SegmentDecoder.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace anira::models::moonshine_streaming {

/// Moonshine streaming: words while someone speaks. A segment's text is revised every 0.5 s (a
/// provisional core::Segment) and given once more when it ends (final, the same id). A segment
/// ends where the caller says (end_segment, e.g. at the end of a voice activity detector's span)
/// or at Moonshine's limit of 15.5 s.
///
///   - Audio thread: process(audio) never waits (downmixed, resampled to 16 kHz, queued);
///     end_segment() after the segment's last block.
///   - Decoding: a thread of the wrapper's own runs the encoder and the token loop (each step a
///   job,
///     core::AsyncHandler).
///   - Reader thread: pop(segment) gives every revision in order, latest() the newest; positions in
///     host samples since reset().
///   - Offline: run() (segments cut at the limit).
class Moonshine {
public:
    Moonshine(anira::Context& context,
              const core::ModelSource& weights,
              Variant variant = Variant::TinyEn);
    ~Moonshine();
    Moonshine(const Moonshine&) = delete;
    Moonshine& operator=(const Moonshine&) = delete;
    Moonshine(Moonshine&&) = delete;
    Moonshine& operator=(Moonshine&&) = delete;

    /// Main thread, no process() running: the host's rate and largest block; starts decoding. The
    /// plan report (anira's prepare).
    anira::PlanReport prepare(double sample_rate, std::size_t max_block);
    /// Main thread, neither process() nor the reader running: a new stream (the clock, the
    /// segment and the queues start over).
    void reset();

    /// Audio thread. ANIRA_ERROR_CAPACITY when the decoder fell so far behind that audio was lost
    /// (counted in dropped_audio()).
    anira_status process(std::span<const float> mono) noexcept;
    /// Audio thread: [channels, n] or [n] float host audio (downmixed), as anira's handler takes
    /// it.
    anira_status process(const anira::Tensor& audio) noexcept;
    /// Audio thread: the current segment ends after the audio given so far.
    void end_segment() noexcept;

    /// Reader thread: the oldest revision not taken yet.
    bool pop(core::Segment& segment) { return m_segments.pop(segment); }
    /// Reader thread (one): the newest revision.
    [[nodiscard]] core::Segment latest() { return m_latest.read(); }
    /// Revisions the reader did not take in time.
    [[nodiscard]] std::size_t dropped() const noexcept { return m_dropped.load(); }
    /// Model samples the audio thread could not queue.
    [[nodiscard]] std::size_t dropped_audio() const noexcept { return m_dropped_audio.load(); }
    /// Segments lost to a failed job (anira's error; the next segment starts clean).
    [[nodiscard]] std::size_t failed() const noexcept { return m_failed.load(); }
    /// Whether the decoder has taken every queued block (for tests and offline use).
    [[nodiscard]] bool idle() const noexcept;
    /// anira's plan report and plan, for both graphs (one plan index: they list the same engines).
    [[nodiscard]] anira::PlanReport plan_report() const { return m_decoder.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept;

    /// Every final segment of a recording.
    static std::vector<core::Segment> run(anira::Context& context,
                                          const core::ModelSource& weights,
                                          std::span<const float> mono,
                                          double sample_rate,
                                          Variant variant = Variant::TinyEn);

private:
    static constexpr std::size_t k_block = 256;  // model samples per queued item
    struct Block {
        std::array<float, k_block> m_samples{};
        std::uint16_t m_count = 0;
        bool m_end = false;  // the segment ends after this block's samples
    };

    void decode_loop();
    void step(bool end);
    void emit(bool final);
    void stop();
    [[nodiscard]] std::int64_t to_host(std::int64_t model_sample) const noexcept;

    SegmentDecoder m_decoder;
    double m_host_rate = k_sample_rate;
    std::size_t m_max_block = 0;
    thl::dsp_basics::Resampler m_resampler;
    std::vector<float> m_resampled;
    std::vector<float> m_mono;  // a host tensor's downmix
    Block m_pending;            // audio thread: the block being filled

    core::SpscQueue<Block> m_audio;
    core::SpscQueue<core::Segment> m_segments;
    core::Latest<core::Segment> m_latest;
    std::atomic<std::size_t> m_dropped{0};
    std::atomic<std::size_t> m_dropped_audio{0};
    std::atomic<std::size_t> m_failed{0};
    std::atomic<std::uint64_t> m_queued{0};
    std::atomic<std::uint64_t> m_taken{0};

    // Decode thread.
    std::vector<float> m_step;
    std::size_t m_step_fill = 0;
    std::int64_t m_model_samples = 0;  // taken from the queue since reset()
    std::int64_t m_segment_start = 0;  // model sample
    std::uint64_t m_segment_id = 0;
    bool m_in_segment = false;
    std::thread m_thread;
    std::atomic<bool> m_running{false};
};

}  // namespace anira::models::moonshine_streaming
