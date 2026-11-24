// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/HostAudio.h>
#include <anira/models/core/LogMel.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/nemotron_diarization/Descriptions.h>
#include <anira/models/nemotron_diarization/Diarizer.h>
#include <anira/models/nemotron_diarization/SpeakerCache.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace anira::models::nemotron_diarization {

namespace {

constexpr std::size_t k_speakers = SpeakerCache::k_speakers;
constexpr std::size_t k_hidden = SpeakerCache::k_hidden;
constexpr std::size_t k_sub = SpeakerCache::k_subsampling;
constexpr std::size_t k_frames = k_cached + k_chunk + k_lookahead;  // 205
constexpr std::size_t k_audio_queue = 64 * 16000 / 256;
constexpr std::size_t k_frame_queue = std::size_t{100} * 60 * 2;  // two minutes of 10 ms frames
constexpr std::size_t k_turn_queue = 1024;
constexpr auto k_poll = std::chrono::milliseconds(5);
constexpr double k_frame_seconds = 0.01;

anira::Tensor host(void* data, anira_dtype dtype, std::initializer_list<std::int64_t> shape) {
    return anira::Tensor::from_host(data,
                                    dtype,
                                    std::span<const std::int64_t>(shape.begin(), shape.size()));
}

}  // namespace

Diarizer::Diarizer(anira::Context& context, const core::ModelSource& weights, TurnConfig turns)
    : m_turn_config(turns)
    , m_handler(
          std::make_unique<core::AsyncHandler>(context, k_nemotron_diarization_model_json, weights))
    , m_mels(16 * k_chunk_mels * core::LogMel::k_mels)
    , m_features(k_chunk_mels * core::LogMel::k_mels)
    , m_cached(k_cached * k_hidden)
    , m_mask(k_frames)
    , m_logits(k_frames * k_sub * k_speakers)
    , m_chunk_embeds((k_chunk + k_lookahead) * k_hidden)
    , m_silence(k_hidden) {
    m_audio.reserve(k_audio_queue);
    m_frames.reserve(k_frame_queue);
    m_turns.reserve(k_turn_queue);
}

Diarizer::~Diarizer() {
    stop();
}

void Diarizer::stop() {
    m_running.store(false);
    if (m_thread.joinable()) { m_thread.join(); }
}

anira::PlanReport Diarizer::prepare(double sample_rate, std::size_t max_block) {
    stop();
    anira::PlanReport report = m_handler->prepare(anira::Async{.max_in_flight = 1});
    m_handler->bind_output(
        0,
        host(m_logits.data(), ANIRA_DTYPE_F32, {1, k_frames * k_sub, k_speakers}));
    m_handler->bind_output(
        1,
        host(m_chunk_embeds.data(), ANIRA_DTYPE_F32, {1, k_chunk + k_lookahead, k_hidden}));
    m_handler->bind_output(2, host(m_silence.data(), ANIRA_DTYPE_F32, {k_hidden}));
    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_mono.assign(max_block, 0.0F);
    m_resampler.prepare(sample_rate, k_sample_rate, max_block);
    m_resampled.assign(m_resampler.max_output(max_block), 0.0F);
    m_mel.prepare(k_block);
    m_new_mels.assign(core::LogMel::max_frames(k_block) * core::LogMel::k_mels, 0.0F);
    reset();
    return report;
}

void Diarizer::reset() {
    stop();
    m_resampler.reset();
    m_mel.reset();
    m_cache.reset();
    m_pending = {};
    m_audio.clear();
    m_frames.clear();
    m_turns.clear();
    m_latest.reset();
    m_dropped.store(0);
    m_dropped_audio.store(0);
    m_queued.store(0);
    m_taken.store(0);
    m_mel_count = 0;
    m_scored = 0;
    m_open = {};
    m_turn_id = 0;
    m_running.store(true);
    m_thread = std::thread([this] { loop(); });
}

anira_status Diarizer::process(const anira::Tensor& audio) noexcept {
    const std::size_t n = core::samples(audio);
    if (n > m_max_block || !core::downmix(audio, m_mono)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    return process(std::span<const float>(m_mono.data(), n));
}

anira_status Diarizer::set_plan(std::uint32_t plan) noexcept {
    return m_handler->set_plan(plan);
}

anira_status Diarizer::process(std::span<const float> mono) noexcept {
    if (mono.size() > m_max_block) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::size_t produced = m_resampler.process(mono, m_resampled);
    anira_status status = ANIRA_OK;
    for (std::size_t i = 0; i < produced; ++i) {
        m_pending.m_samples[m_pending.m_count++] = m_resampled[i];
        if (m_pending.m_count == k_block) {
            if (m_audio.push(m_pending)) {
                m_queued.fetch_add(1, std::memory_order_release);
            } else {
                m_dropped_audio.fetch_add(k_block, std::memory_order_relaxed);
                status = ANIRA_ERROR_CAPACITY;
            }
            m_pending.m_count = 0;
        }
    }
    return status;
}

void Diarizer::flush() noexcept {
    m_pending.m_end = true;
    if (m_audio.push(m_pending)) { m_queued.fetch_add(1, std::memory_order_release); }
    m_pending = {};
}

bool Diarizer::idle() const noexcept {
    return m_taken.load(std::memory_order_acquire) == m_queued.load(std::memory_order_acquire);
}

std::int64_t Diarizer::to_host(std::int64_t frame) const noexcept {
    return std::llround(static_cast<double>(frame) * k_frame_seconds * m_host_rate);
}

void Diarizer::loop() {
    Block block;
    while (m_running.load()) {
        if (!m_audio.pop(block)) {
            std::this_thread::sleep_for(k_poll);
            continue;
        }
        try {
            add_mels(std::span<const float>(block.m_samples.data(), block.m_count));
            while (m_mel_count >= k_chunk_mels) { step(k_step_mels); }
            if (block.m_end) {
                // The frames centred near the audio's end: the right half window of zeros.
                const std::array<float, core::LogMel::k_fft / 2> silence{};
                add_mels(silence);
                while (m_mel_count >= k_chunk_mels) { step(k_step_mels); }
                // The frames still waiting for their look-ahead: padded with silence, scored.
                const float floor = std::log(std::ldexp(1.0F, -24));
                while (m_mel_count > 0) {
                    std::fill(m_mels.begin() +
                                  static_cast<std::ptrdiff_t>(m_mel_count * core::LogMel::k_mels),
                              m_mels.begin() +
                                  static_cast<std::ptrdiff_t>(k_chunk_mels * core::LogMel::k_mels),
                              floor);
                    step(std::min(m_mel_count, k_step_mels));
                }
                for (std::size_t s = 0; s < k_speakers; ++s) { close(s, true); }
            }
        } catch (const std::exception&) {
            // A failed step: its frames are lost, the clock goes on past them.
            m_scored += static_cast<std::int64_t>(m_mel_count);
            m_mel_count = 0;
            if (block.m_end) {
                for (std::size_t s = 0; s < k_speakers; ++s) { close(s, true); }
            }
        }
        m_taken.fetch_add(1, std::memory_order_release);
    }
}

void Diarizer::add_mels(std::span<const float> samples) {
    const std::size_t made = m_mel.process(samples, m_new_mels);
    std::copy_n(m_new_mels.begin(),
                made * core::LogMel::k_mels,
                m_mels.begin() + static_cast<std::ptrdiff_t>(m_mel_count * core::LogMel::k_mels));
    m_mel_count += made;
}

// One chunk: the next 104 mel frames with the speaker cache in front; scores `scored_mels` frames.
void Diarizer::step(std::size_t scored_mels) {
    std::copy_n(m_mels.begin(), k_chunk_mels * core::LogMel::k_mels, m_features.begin());
    const std::vector<float> real = m_cache.embeds();  // the cache, then the FIFO
    const std::size_t n = m_cache.frames();
    std::ranges::fill(m_cached, 0.0F);
    std::ranges::copy(real,
                      m_cached.begin() + static_cast<std::ptrdiff_t>((k_cached - n) * k_hidden));
    std::ranges::fill(m_mask, 0);
    std::fill(m_mask.begin() + static_cast<std::ptrdiff_t>(k_cached - n), m_mask.end(), 1);
    const std::array<anira::Tensor, 3> inputs{
        host(m_features.data(), ANIRA_DTYPE_F32, {1, k_chunk_mels, core::LogMel::k_mels}),
        host(m_cached.data(), ANIRA_DTYPE_F32, {1, k_cached, k_hidden}),
        host(m_mask.data(), ANIRA_DTYPE_I64, {1, k_frames})};
    core::Ticket job = m_handler->submit(inputs);
    if (job.wait() == core::TicketStatus::Failed) {
        throw Error("Diarizer: the step failed (" + std::to_string(job.error()) + ")");
    }

    // The real frames' logits: the cache's and FIFO's, then the chunk's and the look-ahead's.
    const std::size_t first = (k_cached - n) * k_sub;
    const std::span<const float> logits(m_logits.data() + (first * k_speakers),
                                        (n + k_chunk + k_lookahead) * k_sub * k_speakers);
    m_input_embeds.assign(real.begin(), real.end());
    m_input_embeds.insert(m_input_embeds.end(), m_chunk_embeds.begin(), m_chunk_embeds.end());
    m_cache.update(m_input_embeds, logits, m_silence, k_chunk);

    // The chunk's own frames, 10 ms each, as probabilities.
    for (std::size_t f = 0; f < scored_mels; ++f) {
        SpeakerFrame frame;
        frame.m_position = to_host(m_scored);
        for (std::size_t s = 0; s < k_speakers; ++s) {
            const float logit = logits[(((n * k_sub) + f) * k_speakers) + s];
            frame.m_probability[s] = 1.0F / (1.0F + std::exp(-logit));
        }
        m_latest.write(frame);
        if (!m_frames.push(frame)) { m_dropped.fetch_add(1, std::memory_order_relaxed); }
        track(m_scored, frame.m_probability);
        ++m_scored;
    }
    // Step past the chunk's frames.
    if (m_mel_count > k_step_mels) {
        std::copy(m_mels.begin() + static_cast<std::ptrdiff_t>(k_step_mels * core::LogMel::k_mels),
                  m_mels.begin() + static_cast<std::ptrdiff_t>(m_mel_count * core::LogMel::k_mels),
                  m_mels.begin());
        m_mel_count -= k_step_mels;
    } else {
        m_mel_count = 0;
    }
}

void Diarizer::track(std::int64_t frame, const std::array<float, k_speakers>& probability) {
    const auto gap = static_cast<std::int64_t>(m_turn_config.m_min_gap / k_frame_seconds);
    for (std::size_t s = 0; s < k_speakers; ++s) {
        Open& open = m_open[s];
        if (probability[s] > m_turn_config.m_threshold) {
            if (!open.m_active) { open = {.m_active = true, .m_start = frame, .m_last = frame}; }
            open.m_last = frame;
        } else if (open.m_active && frame - open.m_last > gap) {
            close(s, false);
        }
    }
}

void Diarizer::close(std::size_t speaker, bool /*all*/) {
    Open& open = m_open[speaker];
    if (!open.m_active) { return; }
    open.m_active = false;
    const std::int64_t frames = open.m_last + 1 - open.m_start;
    if (static_cast<double>(frames) * k_frame_seconds < m_turn_config.m_min_turn) { return; }
    core::Segment turn;
    turn.m_id = m_turn_id++;
    turn.m_start_sample = to_host(open.m_start);
    turn.m_end_sample = to_host(open.m_last + 1);
    turn.m_speaker = static_cast<int>(speaker);
    turn.m_is_final = true;
    if (!m_turns.push(turn)) { m_dropped.fetch_add(1, std::memory_order_relaxed); }
}

std::vector<core::Segment> Diarizer::run(anira::Context& context,
                                         const core::ModelSource& weights,
                                         std::span<const float> mono,
                                         double sample_rate,
                                         TurnConfig turns) {
    constexpr std::size_t k_host_block = 4096;
    Diarizer diarizer(context, weights, turns);
    diarizer.prepare(sample_rate, k_host_block);
    std::vector<core::Segment> out;
    core::Segment turn;
    SpeakerFrame frame;
    const auto take = [&] {
        while (diarizer.pop(frame)) {}
        while (diarizer.pop(turn)) { out.push_back(turn); }
    };
    for (std::size_t start = 0; start < mono.size(); start += k_host_block) {
        diarizer.process(mono.subspan(start, std::min(k_host_block, mono.size() - start)));
        while (!diarizer.idle()) {
            take();
            std::this_thread::sleep_for(k_poll);
        }
    }
    diarizer.flush();
    while (!diarizer.idle()) {
        take();
        std::this_thread::sleep_for(k_poll);
    }
    take();
    std::ranges::sort(out, {}, &core::Segment::m_start_sample);
    return out;
}

}  // namespace anira::models::nemotron_diarization
