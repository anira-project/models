// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Latest.h>
#include <anira/models/core/LogMel.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/core/SpscQueue.h>
#include <anira/models/nemotron_diarization/SpeakerCache.h>
#include <tanh/dsp/utils/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <vector>

namespace anira::models::nemotron_diarization {

inline constexpr double k_sample_rate = 16000.0;
inline constexpr std::size_t k_chunk = 9;      ///< encoder frames scored per step (720 ms)
inline constexpr std::size_t k_lookahead = 4;  ///< encoder frames a step looks ahead
inline constexpr std::size_t k_cached = 192;   ///< cache and FIFO frames a step attends to
inline constexpr std::size_t k_chunk_mels =
    (k_chunk + k_lookahead) * SpeakerCache::k_subsampling;                         // 104
inline constexpr std::size_t k_step_mels = k_chunk * SpeakerCache::k_subsampling;  // 72

/// Who speaks: every speaker's probability for one 10 ms frame.
struct SpeakerFrame {
    std::int64_t m_position = -1;  ///< the frame's first host sample since reset()
    std::array<float, SpeakerCache::k_speakers> m_probability{};
};

/// A speaker turn: the speaker's probability above `m_threshold` for at least `m_min_turn` seconds,
/// gaps below `m_min_gap` seconds bridged.
struct TurnConfig {
    float m_threshold = 0.5F;
    double m_min_turn = 0.25;
    double m_min_gap = 0.5;
};

/// Nemotron 3 Diarization (streaming Sortformer): up to 8 speakers, numbered in the order they
/// first speak, every 10 ms, 1.04 s after the audio (one 720 ms step and its 320 ms look-ahead).
/// Each step is a job (core::AsyncHandler, anira's Async contract from M3) on a thread of the
/// wrapper's own, with the speaker cache (SpeakerCache) in between. A detector: it never changes
/// the audio.
///
///   - Audio thread: process(audio) never waits (downmixed, resampled to 16 kHz, queued).
///   - Reader thread: pop(SpeakerFrame&) every 10 ms frame, pop(core::Segment&) every speaker turn
///     once it ended (m_speaker; no text), latest() the newest frame; positions in host samples.
///   - Offline: run().
class Diarizer {
public:
    Diarizer(anira::Context& context, const core::ModelSource& weights, TurnConfig turns = {});
    ~Diarizer();
    Diarizer(const Diarizer&) = delete;
    Diarizer& operator=(const Diarizer&) = delete;
    Diarizer(Diarizer&&) = delete;
    Diarizer& operator=(Diarizer&&) = delete;

    /// Main thread, no process() running: the host's rate and largest block; starts the steps. The
    /// plan report (anira's prepare).
    anira::PlanReport prepare(double sample_rate, std::size_t max_block);
    /// Main thread, neither process() nor the reader running: a new stream (the clock, the
    /// speakers and the queues).
    void reset();

    /// Audio thread. ANIRA_ERROR_CAPACITY when the steps fell so far behind that audio was lost.
    anira_status process(std::span<const float> mono) noexcept;
    /// Audio thread: [channels, n] or [n] float host audio (downmixed), as anira's handler takes
    /// it.
    anira_status process(const anira::Tensor& audio) noexcept;
    /// Audio thread: the stream ends here; the frames still in the look-ahead are scored.
    void flush() noexcept;

    bool pop(SpeakerFrame& frame) noexcept { return m_frames.pop(frame); }
    bool pop(core::Segment& turn) { return m_turns.pop(turn); }
    [[nodiscard]] SpeakerFrame latest() noexcept { return m_latest.read(); }
    [[nodiscard]] std::size_t dropped() const noexcept { return m_dropped.load(); }
    [[nodiscard]] std::size_t dropped_audio() const noexcept { return m_dropped_audio.load(); }
    [[nodiscard]] bool idle() const noexcept;
    /// anira's plan report and plan, for the model.
    [[nodiscard]] anira::PlanReport plan_report() const { return m_handler->plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept;

    /// The speaker turns of a recording.
    static std::vector<core::Segment> run(anira::Context& context,
                                          const core::ModelSource& weights,
                                          std::span<const float> mono,
                                          double sample_rate,
                                          TurnConfig turns = {});

private:
    static constexpr std::size_t k_block = 256;
    struct Block {
        std::array<float, k_block> m_samples{};
        std::uint16_t m_count = 0;
        bool m_end = false;
    };
    struct Open {
        bool m_active = false;
        std::int64_t m_start = 0;  // 10 ms frame index
        std::int64_t m_last = 0;   // the last frame above the threshold
    };

    void loop();
    void add_mels(std::span<const float> samples);
    void step(std::size_t scored_mels);
    void track(std::int64_t frame, const std::array<float, SpeakerCache::k_speakers>& probability);
    void close(std::size_t speaker, bool all);
    void stop();
    [[nodiscard]] std::int64_t to_host(std::int64_t frame) const noexcept;

    TurnConfig m_turn_config;
    std::unique_ptr<core::AsyncHandler> m_handler;
    SpeakerCache m_cache;
    core::LogMel m_mel;
    double m_host_rate = k_sample_rate;
    std::size_t m_max_block = 0;
    thl::dsp::utils::Resampler m_resampler;
    std::vector<float> m_resampled;
    std::vector<float> m_mono;  // a host tensor's downmix
    Block m_pending;

    core::SpscQueue<Block> m_audio;
    core::SpscQueue<SpeakerFrame> m_frames;
    core::SpscQueue<core::Segment> m_turns;
    core::Latest<SpeakerFrame> m_latest;
    std::atomic<std::size_t> m_dropped{0};
    std::atomic<std::size_t> m_dropped_audio{0};
    std::atomic<std::uint64_t> m_queued{0};
    std::atomic<std::uint64_t> m_taken{0};

    // Step thread.
    std::vector<float> m_mels;  // mel frames not yet stepped past, frame-major
    std::size_t m_mel_count = 0;
    std::vector<float> m_new_mels;
    std::int64_t m_scored = 0;          // 10 ms frames scored since reset()
    std::vector<float> m_features;      // [1, 104, 128]
    std::vector<float> m_cached;        // [1, 192, 512]
    std::vector<std::int64_t> m_mask;   // [1, 205]
    std::vector<float> m_logits;        // [1, 1640, 8]
    std::vector<float> m_chunk_embeds;  // [1, 13, 512]
    std::vector<float> m_silence;       // [512]
    std::vector<float> m_input_embeds;
    std::array<Open, SpeakerCache::k_speakers> m_open{};
    std::uint64_t m_turn_id = 0;
    std::thread m_thread;
    std::atomic<bool> m_running{false};
};

}  // namespace anira::models::nemotron_diarization
