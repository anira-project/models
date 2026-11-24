// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Latest.h>
#include <anira/models/core/LogMel.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/core/SpscQueue.h>
#include <anira/models/core/Vocabulary.h>
#include <anira/models/core/Words.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace anira::models::nemotron_asr_streaming {

inline constexpr double k_sample_rate = 16000.0;
inline constexpr std::size_t k_new_mels = 56;   ///< mel frames a step adds (560 ms)
inline constexpr std::size_t k_cache_mels = 9;  ///< the previous step's last frames in front
inline constexpr std::size_t k_step_mels = k_cache_mels + k_new_mels;  // 65
inline constexpr std::size_t k_step_frames = 7;  ///< encoder frames a step gives (80 ms each)

/// The language-prompt index of a target language: a locale ("en-US", "de-DE"), a bare code
/// ("de") or "auto" (the model detects the language). The model's prompt dictionary
/// (nvidia/nemotron-3.5-asr-streaming-0.6b, processor_config.json); nullopt for an unknown one.
[[nodiscard]] std::optional<std::int64_t> language_id(std::string_view language) noexcept;

struct AsrConfig {
    std::string m_language = "auto";
    /// A segment ends at end_segment() or after this many seconds (the stream goes on; the text
    /// is cut at the step boundary).
    double m_max_segment = 30.0;
    /// The encoder's ONNX Runtime threads (core::OnnxEngine; 0: one per physical core).
    std::uint32_t m_threads = 0;
};

/// Nemotron 3.5 ASR Streaming 0.6B (cache-aware FastConformer, RNNT, 40 languages): text every
/// 560 ms, the encoder looking 480 ms ahead within each step (no further look-ahead). Each
/// encoder step, decoder and joint call is a job (core::AsyncHandler) on a thread of the
/// wrapper's own; the log-mel front end (core::LogMel) and the greedy RNNT loop are the
/// wrapper's. A segment's text is revised every step (a provisional core::Segment) and given once
/// more when it ends (final, the same id), with word timings (80 ms resolution). Language tags
/// ("<en-US>", given in auto mode) are left out of the text.
///
///   - Audio thread: process(audio) never waits (downmixed, resampled to 16 kHz, queued);
///     end_segment() after the segment's last block (e.g. where a voice activity detector's span
///     ends): its last frames are scored and the model starts over for the next segment.
///   - Reader thread: pop(segment) gives every revision in order, latest() the newest; positions
///     in host samples since reset().
///   - Offline: run().
///
/// The weights must be a directory source: the ONNX files keep their weights in external
/// `.onnx.data` files, which ONNX Runtime resolves only from a path.
class NemotronAsr {
public:
    /// Throws Error for an unknown language.
    NemotronAsr(anira::Context& context, const core::ModelSource& weights, AsrConfig config = {});
    ~NemotronAsr();
    NemotronAsr(const NemotronAsr&) = delete;
    NemotronAsr& operator=(const NemotronAsr&) = delete;
    NemotronAsr(NemotronAsr&&) = delete;
    NemotronAsr& operator=(NemotronAsr&&) = delete;

    /// Main thread, no process() running: the host's rate and largest block; starts decoding. The
    /// plan report (anira's prepare).
    anira::PlanReport prepare(double sample_rate, std::size_t max_block);
    /// Main thread, neither process() nor the reader running: a new stream (the clock, the model
    /// and the queues).
    void reset();

    /// Audio thread. ANIRA_ERROR_CAPACITY when decoding fell so far behind that audio was lost.
    anira_status process(std::span<const float> mono) noexcept;
    /// Audio thread: [channels, n] or [n] float host audio (downmixed), as anira's handler takes
    /// it.
    anira_status process(const anira::Tensor& audio) noexcept;
    /// Audio thread: the current segment ends after the audio given so far.
    void end_segment() noexcept;

    bool pop(core::Segment& segment) { return m_segments.pop(segment); }
    [[nodiscard]] core::Segment latest() { return m_latest.read(); }
    [[nodiscard]] std::size_t dropped() const noexcept { return m_dropped.load(); }
    [[nodiscard]] std::size_t dropped_audio() const noexcept { return m_dropped_audio.load(); }
    /// Steps lost to a failed job (anira's error).
    [[nodiscard]] std::size_t failed() const noexcept { return m_failed.load(); }
    [[nodiscard]] bool idle() const noexcept;
    /// anira's plan report and plan, for the three graphs (one plan index: they list the same
    /// engines).
    [[nodiscard]] anira::PlanReport plan_report() const { return m_encoder->plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept;

    /// Every final segment of a recording (one, or one per m_max_segment).
    static std::vector<core::Segment> run(anira::Context& context,
                                          const core::ModelSource& weights,
                                          std::span<const float> mono,
                                          double sample_rate,
                                          AsrConfig config = {});

private:
    static constexpr std::size_t k_block = 256;
    struct Block {
        std::array<float, k_block> m_samples{};
        std::uint16_t m_count = 0;
        bool m_end = false;  // the segment ends after this block's samples
    };

    void loop();
    void take(std::span<const float> samples);
    void step(std::size_t new_mels);
    void decode(std::size_t frames);
    void run_decoder(std::int64_t token);
    void finish_segment(bool end);
    void emit(bool final, bool at_end = false);
    void start_over();
    void stop();
    [[nodiscard]] std::int64_t to_host(std::int64_t model_sample) const noexcept;

    AsrConfig m_config;
    std::int64_t m_language = 0;
    core::Vocabulary m_vocabulary;
    std::unique_ptr<core::AsyncHandler> m_encoder;
    std::unique_ptr<core::AsyncHandler> m_decoder;
    std::unique_ptr<core::AsyncHandler> m_joint;
    core::LogMel m_mel;
    double m_host_rate = k_sample_rate;
    std::size_t m_max_block = 0;
    thl::dsp_basics::Resampler m_resampler;
    std::vector<float> m_resampled;
    std::vector<float> m_mono;  // a host tensor's downmix
    Block m_pending;

    core::SpscQueue<Block> m_audio;
    core::SpscQueue<core::Segment> m_segments;
    core::Latest<core::Segment> m_latest;
    std::atomic<std::size_t> m_dropped{0};
    std::atomic<std::size_t> m_dropped_audio{0};
    std::atomic<std::size_t> m_failed{0};
    std::atomic<std::uint64_t> m_queued{0};
    std::atomic<std::uint64_t> m_taken{0};

    // Decode thread.
    std::vector<float> m_mels;  // the 9 cached frames, then the frames not yet stepped, frame-major
    std::size_t m_mel_count = 0;  // frames after the cached ones
    std::vector<float> m_new_mels;
    std::vector<float> m_features;  // [1, 65, 128]
    std::int64_t m_length = 0;
    std::vector<float> m_encoded;  // [1, 7, 1024]
    std::int64_t m_encoded_length = 0;
    std::int64_t m_target = 0;
    std::vector<float> m_h;  // [2, 1, 640], the committed LSTM state
    std::vector<float> m_c;
    std::vector<float> m_h_next;
    std::vector<float> m_c_next;
    std::vector<float> m_decoder_out;   // [1, 640, 1]
    std::vector<float> m_logits;        // [1, 1, 1, 13088]
    std::int64_t m_model_samples = 0;   // taken since reset()
    std::int64_t m_base = 0;            // model sample of frame 0 since the last start over
    std::int64_t m_frames = 0;          // encoder frames since the last start over
    std::int64_t m_segment_frame = 0;   // the open segment's first encoder frame
    std::vector<core::Token> m_tokens;  // the open segment's, frames relative to m_segment_frame
    std::uint64_t m_segment_id = 0;
    bool m_start_over = false;  // a failed start over, retried at the next block
    std::thread m_thread;
    std::atomic<bool> m_running{false};
};

}  // namespace anira::models::nemotron_asr_streaming
