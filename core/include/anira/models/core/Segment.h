// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace anira::models::core {

/// One word of a segment; positions in samples of the caller's audio (its rate; the wrapper that
/// delivers it says from when on).
struct Word {
    std::string m_text;
    std::int64_t m_start_sample = 0;
    std::int64_t m_end_sample = 0;

    friend bool operator==(const Word&, const Word&) = default;
};

/// What every engine delivers: a stretch of speech, provisional or final.
struct Segment {
    std::uint64_t m_id = 0;  ///< unique per stream; a revision reuses the id
    std::int64_t m_start_sample = 0;
    std::int64_t m_end_sample = 0;
    std::string m_text;
    std::vector<Word> m_words;  ///< empty when the engine has no word timing
    /// false: a provisional hypothesis that a later Segment with the same id replaces
    /// (streaming engines while a segment is open). Every id ends with one final Segment.
    bool m_is_final = true;
    /// Diarization engines: who spoke from m_start_sample to m_end_sample (speakers numbered
    /// 0.. in order of first appearance in the stream; no text). -1 for text segments.
    int m_speaker = -1;

    friend bool operator==(const Segment&, const Segment&) = default;
};

}  // namespace anira::models::core
