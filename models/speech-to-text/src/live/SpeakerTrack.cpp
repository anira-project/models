// SPDX-License-Identifier: Apache-2.0

#include "SpeakerTrack.h"

#include <anira/models/stt/Segment.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace anira::models::stt {

namespace {

/// Turns that ended this long before every awaited line are no longer needed (16 kHz samples).
constexpr std::int64_t k_keep = std::int64_t{60} * 16000;
/// A word's speaker is the one heard at its onset: over its first k_onset (but at least
/// k_min_word) — a transducer may emit a word's last token (its punctuation) late, so its end
/// can reach into the next speaker's turn (16 kHz samples).
constexpr std::int64_t k_min_word = 1280;  // 80 ms
constexpr std::int64_t k_onset = 2560;     // 160 ms

/// The words of `text` spread over [start, end) by their length: a line without word times.
std::vector<Word> spread(const std::string& text, std::int64_t start, std::int64_t end) {
    std::vector<std::string> pieces;
    std::size_t letters = 0;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto stop = std::min(text.find(' ', begin), text.size());
        if (stop > begin) {
            pieces.push_back(text.substr(begin, stop - begin));
            letters += stop - begin;
        }
        begin = stop + 1;
    }
    std::vector<Word> words;
    words.reserve(pieces.size());
    std::size_t done = 0;
    const auto at = [&](std::size_t count) {
        return start + ((end - start) * static_cast<std::int64_t>(count) /
                        static_cast<std::int64_t>(std::max<std::size_t>(1, letters)));
    };
    for (auto& piece : pieces) {
        const auto from = at(done);
        done += piece.size();
        words.push_back(
            {.m_text = std::move(piece), .m_start_sample = from, .m_end_sample = at(done)});
    }
    return words;
}

bool ends_sentence(const std::string& word) {
    return !word.empty() && (word.back() == '.' || word.back() == '?' || word.back() == '!');
}

}  // namespace

void SpeakerTrack::add(const Segment& turn) {
    if (turn.m_speaker >= 0 && turn.m_end_sample > turn.m_start_sample) {
        if (!m_turns.empty() && m_turns.back().m_speaker == turn.m_speaker &&
            m_turns.back().m_end == turn.m_start_sample) {
            m_turns.back().m_end = turn.m_end_sample;  // the next piece of a turn going on
        } else {
            m_turns.push_back({.m_start = turn.m_start_sample,
                               .m_end = turn.m_end_sample,
                               .m_speaker = turn.m_speaker});
        }
    }
    m_decided_until = std::max(m_decided_until, turn.m_end_sample);
}

void SpeakerTrack::set_decided_until(std::int64_t position) {
    m_decided_until = std::max(m_decided_until, position);
}

void SpeakerTrack::await(std::uint64_t line,
                         std::int64_t start,
                         std::int64_t end,
                         const std::string& text,
                         std::vector<Word> words) {
    end = std::max(start, end);
    if (words.empty()) { words = spread(text, start, end); }
    m_awaited.push_back(
        {.m_line = line, .m_start = start, .m_end = end, .m_words = std::move(words)});
}

std::vector<std::pair<std::uint64_t, std::vector<SpeakerRun>>> SpeakerTrack::take_labels() {
    std::vector<std::pair<std::uint64_t, std::vector<SpeakerRun>>> labels;
    std::erase_if(m_awaited, [&](const Awaited& line) {
        auto runs = runs_of(line.m_start, line.m_end, line.m_words);
        if (!runs) { return false; }
        if (!runs->empty()) { labels.emplace_back(line.m_line, std::move(*runs)); }
        return true;
    });
    // Forget turns no awaited line can reach.
    auto oldest = m_decided_until - k_keep;
    for (const auto& line : m_awaited) { oldest = std::min(oldest, line.m_start); }
    std::erase_if(m_turns, [&](const Turn& turn) { return turn.m_end + k_nearest < oldest; });
    return labels;
}

void SpeakerTrack::clear() {
    m_turns.clear();
    m_awaited.clear();
    m_decided_until = 0;
}

std::optional<int> SpeakerTrack::speaker_of(std::int64_t start, std::int64_t end) const {
    if (end > m_decided_until) { return std::nullopt; }
    return majority(start, end);
}

int SpeakerTrack::majority(std::int64_t start, std::int64_t end) const {
    std::map<int, std::int64_t> overlap;
    for (const auto& turn : m_turns) {
        const auto from = std::max(start, turn.m_start);
        const auto to = std::min(end, turn.m_end);
        if (to > from) { overlap[turn.m_speaker] += to - from; }
    }
    int best = -1;
    std::int64_t most = 0;
    for (const auto& [speaker, samples] : overlap) {
        if (samples > most) {
            best = speaker;
            most = samples;
        }
    }
    return best;
}

int SpeakerTrack::speaker_near(std::int64_t start, std::int64_t end) const {
    if (const auto speaker = majority(start, end); speaker >= 0) { return speaker; }
    int nearest = -1;
    std::int64_t distance = k_nearest + 1;
    for (const auto& turn : m_turns) {
        const auto gap = turn.m_end <= start ? start - turn.m_end
                                             : std::max<std::int64_t>(0, turn.m_start - end);
        if (gap < distance) {
            distance = gap;
            nearest = turn.m_speaker;
        }
    }
    return nearest;
}

std::optional<std::vector<SpeakerRun>> SpeakerTrack::runs_of(std::int64_t start,
                                                             std::int64_t end,
                                                             const std::vector<Word>& words) const {
    if (std::max(start, end) > m_decided_until) { return std::nullopt; }
    if (words.empty()) {
        const auto speaker = speaker_near(start, std::max(end, start + k_min_word));
        if (speaker < 0) { return std::vector<SpeakerRun>{}; }
        return std::vector<SpeakerRun>{
            {.m_word = 0, .m_start_sample = start, .m_speaker = speaker}};
    }

    // Each word's speaker; words nobody is heard on go with their neighbours.
    std::vector<int> speaker(words.size(), -1);
    for (std::size_t i = 0; i < words.size(); ++i) {
        const auto from = words[i].m_start_sample;
        const auto length = std::clamp(words[i].m_end_sample - from, k_min_word, k_onset);
        speaker[i] = speaker_near(from, from + length);
    }
    const auto known = std::ranges::find_if(speaker, [](int s) { return s >= 0; });
    if (known == speaker.end()) { return std::vector<SpeakerRun>{}; }  // nobody spoke
    std::fill(speaker.begin(), known, *known);
    for (std::size_t i = 1; i < speaker.size(); ++i) {
        if (speaker[i] < 0) { speaker[i] = speaker[i - 1]; }
    }

    // A lone short word inside another speaker's speech is noise — unless it is a sentence of
    // its own.
    for (std::size_t i = 1; i + 1 < speaker.size(); ++i) {
        const bool lone = speaker[i] != speaker[i - 1] && speaker[i - 1] == speaker[i + 1];
        const bool short_word = words[i].m_end_sample - words[i].m_start_sample < k_min_run;
        const bool sentence = ends_sentence(words[i - 1].m_text) && ends_sentence(words[i].m_text);
        if (lone && short_word && !sentence) { speaker[i] = speaker[i - 1]; }
    }

    // A change one word away from the end of a sentence moves to it.
    for (std::size_t i = 1; i < speaker.size(); ++i) {
        if (speaker[i] == speaker[i - 1] || ends_sentence(words[i - 1].m_text)) { continue; }
        if (i >= 2 && ends_sentence(words[i - 2].m_text) && speaker[i - 2] == speaker[i - 1]) {
            speaker[i - 1] = speaker[i];  // the new speaker began a word earlier
        } else if (ends_sentence(words[i].m_text) && i + 1 < speaker.size() &&
                   speaker[i + 1] == speaker[i]) {
            speaker[i] = speaker[i - 1];  // ... or a word later
        }
    }

    std::vector<SpeakerRun> runs;
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (!runs.empty() && runs.back().m_speaker == speaker[i]) { continue; }
        runs.push_back({.m_word = i,
                        .m_start_sample = runs.empty() ? start : words[i].m_start_sample,
                        .m_speaker = speaker[i]});
    }
    return runs;
}

}  // namespace anira::models::stt
