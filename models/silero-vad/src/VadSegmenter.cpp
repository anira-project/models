// SPDX-License-Identifier: Apache-2.0

#include <anira/models/silero_vad/VadSegmenter.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace anira::models::silero_vad {

VadSegmenter::VadSegmenter(VadConfig config)
    : m_config(config)
    , m_recent(static_cast<std::size_t>(std::max<std::int64_t>(
          0,
          config.m_split_window / std::max<std::int64_t>(1, config.m_chunk)))) {}

std::optional<SpeechSpan> VadSegmenter::push(float probability) {
    const auto chunk_start = m_position;
    m_position += m_config.m_chunk;
    if (!m_recent.empty()) {
        m_recent[m_recent_next] = probability;
        m_recent_next = (m_recent_next + 1) % m_recent.size();
        m_recent_count = std::min(m_recent_count + 1, m_recent.size());
    }

    if (!m_in_speech) {
        if (probability >= m_config.m_threshold) {
            m_in_speech = true;
            m_speech_start = chunk_start;
            m_silence_start.reset();
        }
        return std::nullopt;
    }

    if (probability < m_config.m_off_threshold) {
        if (!m_silence_start) { m_silence_start = chunk_start; }
        if (m_position - *m_silence_start >= m_config.m_min_silence) {
            return close(*m_silence_start, true);
        }
    } else {
        m_silence_start.reset();
    }

    if (m_position - m_speech_start >= m_config.m_max_segment) {
        // Split long speech: this part ends at the split, the next starts right there.
        const auto split = split_point();
        auto span = close(split, false);
        m_in_speech = true;
        m_speech_start = split;
        return span;
    }
    return std::nullopt;
}

std::optional<SpeechSpan> VadSegmenter::flush() {
    if (!m_in_speech) { return std::nullopt; }
    return close(m_silence_start.value_or(m_position), false);
}

std::int64_t VadSegmenter::split_point() const {
    // The latest of the least likely recent chunks, after the segment's first half.
    auto split = m_position;
    float lowest = 2.0f;
    for (std::size_t back = 0; back < m_recent_count; ++back) {
        const auto index = (m_recent_next + m_recent.size() - 1 - back) % m_recent.size();
        const auto start = m_position - (static_cast<std::int64_t>(back + 1) * m_config.m_chunk);
        if (start <= m_speech_start + (m_config.m_max_segment / 2)) { break; }
        if (m_recent[index] < lowest) {
            lowest = m_recent[index];
            split = start;
        }
    }
    return split;
}

void VadSegmenter::reset() {
    m_recent_next = 0;
    m_recent_count = 0;
    m_position = 0;
    m_in_speech = false;
    m_speech_start = 0;
    m_silence_start.reset();
    m_last_end = 0;
}

std::int64_t VadSegmenter::keep_from() const noexcept {
    const auto from = m_in_speech ? m_speech_start : m_position;
    return std::max(m_last_end, from - m_config.m_padding);
}

std::optional<SpeechSpan> VadSegmenter::close(std::int64_t speech_end, bool pad_end) {
    m_in_speech = false;
    m_silence_start.reset();
    if (speech_end - m_speech_start < m_config.m_min_speech) { return std::nullopt; }
    SpeechSpan span{
        .m_start = std::max(m_last_end, m_speech_start - m_config.m_padding),
        .m_end = pad_end ? std::min(m_position, speech_end + m_config.m_padding) : speech_end,
        .m_speech_start = m_speech_start,
        .m_speech_end = speech_end};
    m_last_end = span.m_end;
    return span;
}

}  // namespace anira::models::silero_vad
