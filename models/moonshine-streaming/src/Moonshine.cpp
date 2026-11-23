// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/moonshine_streaming/Moonshine.h>
#include <anira/models/moonshine_streaming/SegmentDecoder.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <span>
#include <thread>
#include <vector>

namespace anira::models::moonshine_streaming {

namespace {

constexpr std::size_t k_audio_queue = 64 * 16000 / 256;  // 64 s of 16 kHz audio
constexpr std::size_t k_segment_queue = 1024;
constexpr auto k_poll = std::chrono::milliseconds(5);

}  // namespace

Moonshine::Moonshine(anira::Context& context, const core::ModelSource& weights, Variant variant)
    : m_decoder(context, weights, variant)
    , m_step(k_step, 0.0F) {
    m_audio.reserve(k_audio_queue);
    m_segments.reserve(k_segment_queue);
}

Moonshine::~Moonshine() { stop(); }

void Moonshine::stop() {
    m_running.store(false);
    if (m_thread.joinable()) { m_thread.join(); }
}

void Moonshine::prepare(double sample_rate, std::size_t max_block) {
    stop();
    m_decoder.prepare();
    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_resampler.prepare(sample_rate, k_sample_rate, max_block);
    m_resampled.assign(m_resampler.max_output(max_block), 0.0F);
    reset();
}

void Moonshine::reset() {
    stop();
    m_resampler.reset();
    m_pending = {};
    m_audio.clear();
    m_segments.clear();
    m_latest.reset();
    m_dropped.store(0);
    m_dropped_audio.store(0);
    m_queued.store(0);
    m_taken.store(0);
    m_step_fill = 0;
    m_model_samples = 0;
    m_segment_start = 0;
    m_segment_id = 0;
    m_in_segment = false;
    m_decoder.begin();
    m_running.store(true);
    m_thread = std::thread([this] { decode_loop(); });
}

anira_status Moonshine::process(std::span<const float> mono) noexcept {
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

void Moonshine::end_segment() noexcept {
    m_pending.m_end = true;
    if (m_audio.push(m_pending)) {
        m_queued.fetch_add(1, std::memory_order_release);
    } else {
        m_dropped_audio.fetch_add(m_pending.m_count, std::memory_order_relaxed);
    }
    m_pending = {};
}

bool Moonshine::idle() const noexcept {
    return m_taken.load(std::memory_order_acquire) == m_queued.load(std::memory_order_acquire);
}

std::int64_t Moonshine::to_host(std::int64_t model_sample) const noexcept {
    return std::llround(static_cast<double>(model_sample) * m_host_rate / k_sample_rate);
}

void Moonshine::decode_loop() {
    Block block;
    while (m_running.load()) {
        if (!m_audio.pop(block)) {
            std::this_thread::sleep_for(k_poll);
            continue;
        }
        try {
            for (std::size_t i = 0; i < block.m_count; ++i) {
                m_step[m_step_fill++] = block.m_samples[i];
                ++m_model_samples;
                if (m_step_fill == k_step) { step(false); }
            }
            if (block.m_end && (m_in_segment || m_step_fill > 0)) { step(true); }
        } catch (const std::exception&) {
            // A failed job: this segment is lost, the next starts clean.
            m_failed.fetch_add(1, std::memory_order_relaxed);
            m_step_fill = 0;
            m_in_segment = false;
        }
        m_taken.fetch_add(1, std::memory_order_release);
    }
}

// The step buffer into the segment: a full step, or at the end what is there, padded.
void Moonshine::step(bool end) {
    if (!m_in_segment) {
        m_decoder.begin();
        m_in_segment = true;
        m_segment_start = m_model_samples - static_cast<std::int64_t>(m_step_fill);
    }
    if (m_step_fill > 0) {
        std::fill(m_step.begin() + static_cast<std::ptrdiff_t>(m_step_fill), m_step.end(), 0.0F);
        if (!m_decoder.push(m_step)) {  // the segment is full: end it, this step starts the next
            emit(true);
            m_decoder.begin();
            m_in_segment = true;
            m_segment_start = m_model_samples - static_cast<std::int64_t>(m_step_fill);
            m_decoder.push(m_step);
        }
        m_step_fill = 0;
    }
    if (end) {
        emit(true);
        m_in_segment = false;
    } else {
        emit(false);
    }
}

void Moonshine::emit(bool final) {
    double seconds = static_cast<double>(m_decoder.memory() * k_frame) / k_sample_rate;
    if (final) {
        m_decoder.flush();
        seconds = static_cast<double>(m_decoder.steps() * k_step) / k_sample_rate;
    }
    const std::vector<std::int32_t> tokens = m_decoder.decode(seconds);
    core::Segment segment;
    segment.m_id = m_segment_id;
    segment.m_start_sample = to_host(m_segment_start);
    segment.m_end_sample =
        to_host(m_segment_start + static_cast<std::int64_t>(m_decoder.steps() * k_step));
    segment.m_text = m_decoder.tokenizer().text(tokens);
    segment.m_is_final = final;
    m_latest.write(segment);
    if (!m_segments.push(segment)) { m_dropped.fetch_add(1, std::memory_order_relaxed); }
    if (final) { ++m_segment_id; }
}

std::vector<core::Segment> Moonshine::run(anira::Context& context,
                                          const core::ModelSource& weights,
                                          std::span<const float> mono,
                                          double sample_rate,
                                          Variant variant) {
    constexpr std::size_t k_host_block = 4096;
    Moonshine moonshine(context, weights, variant);
    moonshine.prepare(sample_rate, k_host_block);
    std::vector<core::Segment> finals;
    core::Segment segment;
    const auto take = [&] {
        while (moonshine.pop(segment)) {
            if (segment.m_is_final) { finals.push_back(segment); }
        }
    };
    for (std::size_t start = 0; start < mono.size(); start += k_host_block) {
        moonshine.process(mono.subspan(start, std::min(k_host_block, mono.size() - start)));
        while (!moonshine.idle()) {  // offline: no audio is ever lost
            take();
            std::this_thread::sleep_for(k_poll);
        }
    }
    moonshine.end_segment();
    while (!moonshine.idle()) {
        take();
        std::this_thread::sleep_for(k_poll);
    }
    take();
    return finals;
}

}  // namespace anira::models::moonshine_streaming
