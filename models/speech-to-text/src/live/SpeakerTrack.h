// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/stt/Segment.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace anira::models::stt {

/// A stretch of a line one speaker said: from word `m_word` (index into the line's words, as
/// split at spaces) up to the next run's.
struct SpeakerRun {
    std::size_t m_word = 0;
    std::int64_t m_start_sample = 0;  ///< session time
    int m_speaker = 0;                ///< 0.., in order of first appearance

    friend bool operator==(const SpeakerRun&, const SpeakerRun&) = default;
};

/// Who spoke when, from a diarization engine's speaker turns, and who spoke which words of
/// each final transcript line once the diarization has decided its time. Session time
/// (16 kHz samples). Control thread only; pure logic.
///
/// A line's words each go to the speaker heard most at their onset (a word between turns to
/// the nearest turn within k_nearest); a lone short word inside another speaker's speech is
/// taken as noise, and a change right next to the end of a sentence moves to it. So a line in
/// which two people speak — the accurate engine heard no pause between them — comes out as
/// one run per speaker.
class SpeakerTrack {
public:
    /// A turn (Segment with m_speaker >= 0), or the next piece of one; turns of nobody
    /// (m_speaker < 0) only tell how far the diarization has decided.
    void add(const Segment& turn);
    /// The diarization's watermark: everything before `position` is decided.
    void set_decided_until(std::int64_t position);
    /// A final line to label once its time is decided: its words with their times, or just
    /// its text (the words are then spread over [start, end) by length).
    void await(std::uint64_t line,
               std::int64_t start,
               std::int64_t end,
               const std::string& text,
               std::vector<Word> words);
    /// The awaited lines decided since the last call, with who spoke them (lines nobody spoke
    /// are dropped).
    [[nodiscard]] std::vector<std::pair<std::uint64_t, std::vector<SpeakerRun>>> take_labels();
    /// A new session, or the diarization switched off: everything is forgotten.
    void clear();

    /// Up to where the diarization has decided.
    [[nodiscard]] std::int64_t decided_until() const noexcept { return m_decided_until; }
    /// The speaker of most of [start, end): nullopt while undecided, -1 when nobody spoke.
    [[nodiscard]] std::optional<int> speaker_of(std::int64_t start, std::int64_t end) const;
    /// Who spoke which of `words` (times in session samples) on a line starting at `start`:
    /// nullopt while the line's time is undecided, empty when nobody spoke.
    [[nodiscard]] std::optional<std::vector<SpeakerRun>>
        runs_of(std::int64_t start, std::int64_t end, const std::vector<Word>& words) const;

    /// A word with no turn on it goes to a turn at most this far away (16 kHz samples).
    static constexpr std::int64_t k_nearest = 8000;  // 0.5 s
    /// A single word shorter than this between two stretches of the same other speaker is
    /// taken as diarization noise (16 kHz samples).
    static constexpr std::int64_t k_min_run = 4800;  // 0.3 s

private:
    struct Turn {
        std::int64_t m_start = 0;
        std::int64_t m_end = 0;
        int m_speaker = -1;
    };
    struct Awaited {
        std::uint64_t m_line = 0;
        std::int64_t m_start = 0;
        std::int64_t m_end = 0;
        std::vector<Word> m_words;
    };

    /// The speaker of most of [start, end); -1 when no turn overlaps it.
    [[nodiscard]] int majority(std::int64_t start, std::int64_t end) const;
    /// ... else the speaker of the nearest turn within k_nearest; -1.
    [[nodiscard]] int speaker_near(std::int64_t start, std::int64_t end) const;

    std::vector<Turn> m_turns;
    std::vector<Awaited> m_awaited;
    std::int64_t m_decided_until = 0;
};

}  // namespace anira::models::stt
