// SPDX-License-Identifier: Apache-2.0

#include "TranscriptMerger.h"

#include <anira/models/stt/Segment.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace anira::models::stt {

namespace {

/// An accurate segment covering at least this many fast words must have at least
/// k_min_word_share of their count; with fewer it dropped speech (it happens on long spans
/// with several voices or languages), and the fast text is kept instead. Shorter segments
/// are where the accurate engine may take speech for another language.
constexpr std::size_t k_check_words = 8;
constexpr double k_min_word_share = 0.4;
/// An untimed fast utterance is covered when at least this share of its time is.
constexpr double k_min_time_share = 0.5;
/// Fast words the accurate text has nothing for count as lost by it where its words cover
/// less than this share of their time, each word counted for at most k_max_word (16 kHz
/// samples).
constexpr double k_heard_share = 0.3;
constexpr std::int64_t k_max_word = 16000;  // 1 s
/// The fast engine's lag is measured on at least this many words of a segment, and is never
/// taken to be more than k_max_lag (16 kHz samples).
constexpr std::size_t k_lag_words = 2;
constexpr std::int64_t k_max_lag = 16000;  // 1 s

std::size_t word_count(const std::string& text) {
    std::size_t count = 0;
    bool in_word = false;
    for (const char c : text) {
        const bool space = c == ' ';
        if (!space && !in_word) { ++count; }
        in_word = !space;
    }
    return count;
}

std::int64_t middle(std::int64_t start, std::int64_t end) {
    return start + ((end - start) / 2);
}

/// The share of [start, end) that lies within [from, to]; for an empty stretch 1 or 0,
/// whether its point does.
double share_within(std::int64_t start, std::int64_t end, std::int64_t from, std::int64_t to) {
    if (end <= start) { return start >= from && start <= to ? 1.0 : 0.0; }
    const auto inside = std::max<std::int64_t>(0, std::min(end, to) - std::max(start, from));
    return static_cast<double>(inside) / static_cast<double>(end - start);
}

/// The words of a text (split at spaces).
std::vector<std::string> words_of(const std::string& text) {
    std::vector<std::string> words;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = std::min(text.find(' ', begin), text.size());
        if (end > begin) { words.push_back(text.substr(begin, end - begin)); }
        begin = end + 1;
    }
    return words;
}

/// A word as the engines' texts are compared: lower case, without punctuation (the bytes of
/// multibyte UTF-8 characters are kept as they are).
std::string normalized(const std::string& word) {
    std::string out;
    for (const char c : word) {
        const auto byte = static_cast<unsigned char>(c);
        if (byte >= 0x80 || std::isalnum(byte) != 0 || c == '\'') {
            out += static_cast<char>(std::tolower(byte));
        }
    }
    return out;
}

/// A sound, not a word (normalized): the accurate engine leaves these out on purpose.
bool filler(const std::string& word) {
    static constexpr std::array<const char*, 16> k_fillers{"uh",
                                                           "um",
                                                           "uhm",
                                                           "umm",
                                                           "hm",
                                                           "hmm",
                                                           "mm",
                                                           "mhm",
                                                           "ah",
                                                           "eh",
                                                           "er",
                                                           "erm",
                                                           "äh",
                                                           "ähm",
                                                           "öh",
                                                           "hä"};
    return std::ranges::any_of(k_fillers, [&](const char* f) { return word == f; });
}

/// Aligns the words of `a` with those of `b` by the fewest edits (Levenshtein): the steps in
/// order, each (index in a, index in b) — a pair where the words are the same or one stands
/// for the other, -1 on one side where the other has a word of its own.
std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> align(const std::vector<std::string>& a,
                                                             const std::vector<std::string>& b) {
    const auto rows = a.size() + 1;
    const auto columns = b.size() + 1;
    std::vector<std::size_t> cost(rows * columns);
    const auto at = [&](std::size_t i, std::size_t j) -> std::size_t& {
        return cost[(i * columns) + j];
    };
    const auto differ = [&](std::size_t i, std::size_t j) -> std::size_t {
        return a[i - 1] == b[j - 1] ? 0 : 1;
    };
    for (std::size_t i = 0; i < rows; ++i) { at(i, 0) = i; }
    for (std::size_t j = 0; j < columns; ++j) { at(0, j) = j; }
    for (std::size_t i = 1; i < rows; ++i) {
        for (std::size_t j = 1; j < columns; ++j) {
            at(i, j) =
                std::min({at(i - 1, j - 1) + differ(i, j), at(i - 1, j) + 1, at(i, j - 1) + 1});
        }
    }
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> path;
    auto i = a.size();
    auto j = b.size();
    const auto index = [](std::size_t k) { return static_cast<std::ptrdiff_t>(k); };
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 && at(i, j) == at(i - 1, j - 1) + differ(i, j)) {
            path.emplace_back(index(i - 1), index(j - 1));
            --i;
            --j;
        } else if (i > 0 && at(i, j) == at(i - 1, j) + 1) {
            path.emplace_back(index(i - 1), -1);
            --i;
        } else {
            path.emplace_back(-1, index(j - 1));
            --j;
        }
    }
    std::ranges::reverse(path);
    return path;
}

/// The line each accurate word goes to: that of the fast word it pairs with. A run of words
/// the fast engine did not have goes with its neighbours; between two lines, the longest
/// pause in it decides.
std::vector<std::size_t> owners(const std::vector<Word>& words,
                                const std::vector<std::ptrdiff_t>& partner,
                                const std::vector<std::size_t>& fast_line) {
    constexpr auto k_none = static_cast<std::size_t>(-1);
    std::vector<std::size_t> owner(words.size(), k_none);
    for (std::size_t k = 0; k < words.size(); ++k) {
        if (partner[k] >= 0) { owner[k] = fast_line[static_cast<std::size_t>(partner[k])]; }
    }
    for (std::size_t k = 0; k < words.size();) {
        if (owner[k] != k_none) {
            ++k;
            continue;
        }
        auto run_end = k;
        while (run_end < words.size() && owner[run_end] == k_none) { ++run_end; }
        const auto before = k > 0 ? owner[k - 1] : k_none;
        const auto after = run_end < words.size() ? owner[run_end] : k_none;
        auto split = before == k_none ? k : run_end;  // [k, split) go with `before`
        if (before != k_none && after != k_none && before != after) {
            std::int64_t longest = -1;
            for (auto w = k - 1; w < run_end; ++w) {
                const auto pause = words[w + 1].m_start_sample - words[w].m_end_sample;
                if (pause > longest) {
                    longest = pause;
                    split = w + 1;
                }
            }
        }
        for (auto w = k; w < run_end; ++w) {
            owner[w] = w < split ? before : (after != k_none ? after : 0);
        }
        k = run_end;
    }
    return owner;
}

/// Writing systems, as far as telling languages apart needs them.
enum class Script : std::uint8_t { None, Latin, Greek, Cyrillic, Other };

Script script_of(char32_t c) {
    if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') ||
        (c >= 0xC0 && c <= 0x24F && c != 0xD7 && c != 0xF7)) {
        return Script::Latin;
    }
    if (c >= 0x370 && c <= 0x3FF) { return Script::Greek; }
    if (c >= 0x400 && c <= 0x52F) { return Script::Cyrillic; }
    const bool symbol = (c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) ||
                        (c >= 0xFE00 && c <= 0xFE6F) || (c >= 0xFF00 && c <= 0xFF20);
    return c >= 0x590 && !symbol ? Script::Other : Script::None;
}

/// The script most of the letters of UTF-8 text are in (None without letters).
Script main_script(const std::string& text) {
    std::array<std::size_t, 5> letters{};
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        if (lead >= 0xF0) {
            length = 4;
        } else if (lead >= 0xE0) {
            length = 3;
        } else if (lead >= 0xC0) {
            length = 2;
        }
        auto code = static_cast<char32_t>(length == 1 ? lead : lead & (0x7FU >> length));
        for (std::size_t k = 1; k < length && i + k < text.size(); ++k) {
            code = (code << 6U) | (static_cast<unsigned char>(text[i + k]) & 0x3FU);
        }
        ++letters[static_cast<std::size_t>(script_of(code))];
        i += length;
    }
    letters[static_cast<std::size_t>(Script::None)] = 0;
    const auto most = std::ranges::max_element(letters);
    return *most == 0 ? Script::None : static_cast<Script>(most - letters.begin());
}

/// Both texts have letters, in different writing systems.
bool differ_in_script(const std::string& a, const std::string& b) {
    const auto script_a = main_script(a);
    const auto script_b = main_script(b);
    return script_a != Script::None && script_b != Script::None && script_a != script_b;
}

}  // namespace

void TranscriptMerger::set_sources(bool fast, bool accurate) {
    m_fast = fast;
    m_accurate = accurate;
    if (!accurate) { m_accurate_start = -1; }
    reconcile();  // e.g. without the accurate engine, settled fast text is final now
}

void TranscriptMerger::set_accurate_watermark(std::int64_t position) {
    if (position <= m_watermark) { return; }
    m_watermark = position;
    reconcile();
}

void TranscriptMerger::set_accurate_start(std::int64_t position) {
    m_accurate_start = position;
}

void TranscriptMerger::end_fast() {
    // Its text will not change any more. Settled, it is resolved like any other: covered by
    // the accurate engine, or final once that is past it (at once without one).
    for (auto& [key, segment] : m_segments) {
        if (key.first == m_epoch) { segment.m_is_final = true; }
    }
    ++m_epoch;               // the next engine's segment ids start over
    m_lag_measured = false;  // ... and its lag is its own
    reconcile();
}

void TranscriptMerger::clear() {
    m_open.clear();
    m_segments.clear();
    m_done.clear();
    m_kept.clear();
    m_spans.clear();
    m_watermark = 0;
    m_accurate_start = -1;
    m_out.clear();
}

void TranscriptMerger::add_fast(const Segment& segment) {
    if (!m_fast) { return; }
    m_segments[{m_epoch, segment.m_id}] = segment;  // the latest revision
    reconcile();
}

void TranscriptMerger::add_accurate(const Segment& segment) {
    if (!m_accurate || !segment.m_is_final || segment.m_text.empty()) { return; }
    const auto start = segment.m_start_sample;
    const auto end = std::max(segment.m_end_sample, start);
    const auto pending = utterances();
    if (joined_mid_utterance(pending, start)) {
        return;  // the end of an utterance the fast engine heard whole: that text stays
    }

    // The fast text it covers.
    std::vector<const Utterance*> covered;
    std::size_t fast_words = 0;
    std::string fast_text;
    for (const auto& utterance : pending) {
        const auto words = covered_words(utterance, start, end);
        if (words == 0) { continue; }
        fast_words += words;
        fast_text += utterance.m_text;
        fast_text += ' ';
        covered.push_back(&utterance);
    }
    const auto accurate_words = word_count(segment.m_text);
    if (fast_words >= k_check_words &&
        static_cast<double>(accurate_words) < k_min_word_share * static_cast<double>(fast_words)) {
        return;  // it lost speech the fast engine heard: keep that instead
    }
    if (accurate_words < k_check_words && differ_in_script(segment.m_text, fast_text)) {
        return;  // a short segment taken for another language: keep the fast text
    }

    // It replaces the lines of what it covers; fast words it does not cover stay (reconcile()
    // gives them lines of their own).
    std::vector<Key> keys;
    std::vector<Line> lines;
    std::vector<FastWord> fast;
    for (const auto* utterance : covered) {
        const auto open = m_open.find(utterance->m_key);
        if (open == m_open.end()) { continue; }
        const auto line = lines.size();
        keys.push_back(open->first);
        lines.push_back(open->second);
        m_open.erase(open);
        const auto& segment_key = utterance->m_key.first;
        if (utterance->m_words.empty()) {
            for (const auto& word : words_of(utterance->m_text)) {
                fast.push_back({.m_text = normalized(word),
                                .m_word = {.m_text = word},
                                .m_timed = false,
                                .m_key = {segment_key, 0},
                                .m_line = line});
            }
        } else {
            for (std::size_t w = 0; w < utterance->m_words.size(); ++w) {
                const auto& word = utterance->m_words[w];
                fast.push_back({.m_text = normalized(word.m_text),
                                .m_word = word,
                                .m_timed = true,
                                .m_key = {segment_key, utterance->m_key.second + w},
                                .m_line = line});
            }
        }
    }
    m_spans.emplace_back(start, end);
    replace(keys, lines, fast, segment);
    reconcile();
}

std::vector<Line> TranscriptMerger::take() {
    return std::exchange(m_out, {});
}

std::int64_t TranscriptMerger::lag_of(const Utterance& utterance) const noexcept {
    return utterance.m_words.empty() ? 0 : m_lag;
}

std::size_t TranscriptMerger::covered_words(const Utterance& utterance,
                                            std::int64_t start,
                                            std::int64_t end) const {
    const auto from = start - k_tolerance;
    const auto to = end + k_tolerance;
    if (utterance.m_words.empty()) {
        return share_within(utterance.m_start, utterance.m_end, from, to) >= k_min_time_share
                   ? word_count(utterance.m_text)
                   : 0;
    }
    return static_cast<std::size_t>(std::ranges::count_if(utterance.m_words, [&](const Word& w) {
        const auto at = middle(w.m_start_sample, w.m_end_sample) - m_lag;
        return at >= from && at <= to;
    }));
}

bool TranscriptMerger::joined_mid_utterance(const std::vector<Utterance>& pending,
                                            std::int64_t start) const {
    if (m_accurate_start < 0 || start >= m_accurate_start + k_join_margin) { return false; }
    return std::ranges::any_of(pending, [&](const Utterance& utterance) {
        const auto lag = lag_of(utterance);
        return utterance.m_start - lag + k_join_lead <= m_accurate_start &&
               utterance.m_end - lag > start;
    });
}

void TranscriptMerger::learn_lag(const Segment& segment,
                                 const std::vector<FastWord>& fast,
                                 const std::vector<std::ptrdiff_t>& partner) {
    std::vector<std::int64_t> lags;
    for (std::size_t k = 0; k < partner.size(); ++k) {
        if (partner[k] < 0) { continue; }
        const auto& word = fast[static_cast<std::size_t>(partner[k])];
        const auto text = normalized(segment.m_words[k].m_text);
        if (!word.m_timed || text.empty() || word.m_text != text) { continue; }
        lags.push_back(word.m_word.m_start_sample - segment.m_words[k].m_start_sample);
    }
    if (lags.size() < k_lag_words) { return; }
    const auto median = lags.begin() + static_cast<std::ptrdiff_t>(lags.size() / 2);
    std::ranges::nth_element(lags, median);
    const auto measured = std::clamp<std::int64_t>(*median, 0, k_max_lag);
    m_lag = m_lag_measured ? ((3 * m_lag) + measured) / 4 : measured;
    m_lag_measured = true;
}

void TranscriptMerger::replace(const std::vector<Key>& keys,
                               const std::vector<Line>& lines,
                               const std::vector<FastWord>& fast,
                               const Segment& segment) {
    const auto start = segment.m_start_sample;
    const auto end = std::max(segment.m_end_sample, start);
    const auto final_line = [](std::uint64_t id,
                               std::int64_t from,
                               std::int64_t to,
                               std::string text,
                               std::vector<Word> words) {
        return Line{.m_id = id,
                    .m_start_sample = from,
                    .m_end_sample = to,
                    .m_text = std::move(text),
                    .m_is_final = true,
                    .m_source = LineSource::Accurate,
                    .m_words = std::move(words)};
    };
    if (lines.empty()) {
        emit(final_line(m_next_id++, start, end, segment.m_text, segment.m_words));
        return;
    }
    if (segment.m_words.empty()) {
        // No words to share out: the first line takes it all.
        emit(final_line(lines.front().m_id, start, end, segment.m_text, {}));
        for (std::size_t i = 1; i < lines.size(); ++i) { withdraw(lines[i]); }
        return;
    }

    // The two texts aligned; the lag measured on the words they agree on.
    std::vector<std::string> accurate;
    accurate.reserve(segment.m_words.size());
    for (const auto& word : segment.m_words) { accurate.push_back(normalized(word.m_text)); }
    std::vector<std::string> fast_texts;
    std::vector<std::size_t> fast_lines;
    for (const auto& word : fast) {
        fast_texts.push_back(word.m_text);
        fast_lines.push_back(word.m_line);
    }
    const auto path = align(accurate, fast_texts);
    std::vector<std::ptrdiff_t> partner(accurate.size(), -1);
    std::vector<bool> paired(fast.size(), false);
    for (const auto& [k, j] : path) {
        if (k < 0 || j < 0) { continue; }
        partner[static_cast<std::size_t>(k)] = j;
        paired[static_cast<std::size_t>(j)] = true;
    }
    learn_lag(segment, fast, partner);
    const auto owner = owners(segment.m_words, partner, fast_lines);

    // What the accurate engine lost: lines it has nothing for, and runs of fast words inside
    // a line it has something for — where it has no word at all.
    std::vector<bool> heard(lines.size(), false);   // the accurate text has words for it
    std::vector<bool> speech(lines.size(), false);  // the fast text has more than fillers
    for (std::size_t j = 0; j < fast.size(); ++j) {
        heard[fast[j].m_line] = heard[fast[j].m_line] || paired[j];
        speech[fast[j].m_line] = speech[fast[j].m_line] || !filler(fast[j].m_text);
    }
    // The accurate engine heard (next to) nothing in [from, to): its words cover less than
    // k_heard_share of it — each word counted for at most k_max_word, as its last one lasts
    // until its speech ended. (Where it heard something else, its words win: the fast engine
    // may have made its own up.)
    const auto silent = [&](std::int64_t from, std::int64_t to) {
        to = std::max(to, from + 1);
        std::int64_t heard_samples = 0;
        for (const auto& word : segment.m_words) {
            const auto word_end = std::min(word.m_end_sample, word.m_start_sample + k_max_word);
            heard_samples += std::max<std::int64_t>(
                0,
                std::min(word_end, to) - std::max(word.m_start_sample, from));
        }
        return static_cast<double>(heard_samples) < k_heard_share * static_cast<double>(to - from);
    };
    std::vector<bool> lost(fast.size(), false);
    for (std::size_t j = 0; j < fast.size();) {
        auto run_end = j;
        std::size_t words = 0;
        while (run_end < fast.size() && !paired[run_end] &&
               fast[run_end].m_line == fast[j].m_line) {
            words += filler(fast[run_end].m_text) ? 0 : 1;
            ++run_end;
        }
        if (run_end == j) {
            ++j;
            continue;
        }
        if (words >= k_lost_run && heard[fast[j].m_line] && fast[j].m_timed &&
            silent(fast[j].m_word.m_start_sample - m_lag,
                   fast[run_end - 1].m_word.m_end_sample - m_lag)) {
            for (auto w = j; w < run_end; ++w) { lost[w] = !filler(fast[w].m_text); }
        }
        j = run_end;
    }

    // Each line's words in the order of the alignment: the accurate ones, and the fast ones
    // the accurate engine lost (at their place, on the session clock).
    std::vector<std::vector<Word>> parts(lines.size());
    for (const auto& [k, j] : path) {
        if (k >= 0) {
            const auto word = static_cast<std::size_t>(k);
            parts[owner[word]].push_back(segment.m_words[word]);
        } else if (lost[static_cast<std::size_t>(j)]) {
            const auto& found = fast[static_cast<std::size_t>(j)];
            auto word = found.m_word;
            word.m_start_sample = std::max<std::int64_t>(0, word.m_start_sample - m_lag);
            word.m_end_sample = std::max<std::int64_t>(0, word.m_end_sample - m_lag);
            parts[found.m_line].push_back(std::move(word));
        }
    }

    // Lines the accurate engine lost as a whole: speech, and no accurate word in their time.
    std::vector<bool> kept(lines.size(), false);
    for (std::size_t i = 0; i < lines.size(); ++i) {
        kept[i] =
            parts[i].empty() && speech[i] && silent(lines[i].m_start_sample, lines[i].m_end_sample);
    }

    // The segment's bounds go to its first and last line — unless a line the accurate engine
    // lost comes before or after: that one keeps its own time.
    std::size_t first = parts.size();
    std::size_t last = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (parts[i].empty() && !kept[i]) { continue; }
        first = std::min(first, i);
        last = i;
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
        auto& part = parts[i];
        if (part.empty()) {
            if (!kept[i]) {
                withdraw(lines[i]);  // fillers it leaves out, or words it heard otherwise
                continue;
            }
            // The accurate engine lost the whole line: it stays as the fast engine heard it.
            m_open[keys[i]] = lines[i];
            for (const auto& word : fast) {
                if (word.m_line == i) { m_kept.insert(word.m_key); }
            }
            continue;
        }
        // A line the accurate engine lost after this one: this one ends where that begins (the
        // accurate engine's last word lasts until its speech ended, the lost line included).
        for (auto next = i + 1; next < lines.size(); ++next) {
            if (!parts[next].empty()) { break; }
            if (!kept[next]) { continue; }
            const auto limit = std::max(part.back().m_start_sample, lines[next].m_start_sample);
            part.back().m_end_sample = std::min(part.back().m_end_sample, limit);
            break;
        }
        std::string text;
        for (const auto& word : part) {
            if (!text.empty()) { text += ' '; }
            text += word.m_text;
        }
        const auto from =
            i == first ? std::min(start, part.front().m_start_sample) : part.front().m_start_sample;
        const auto to =
            i == last ? std::max(end, part.back().m_end_sample) : part.back().m_end_sample;
        emit(final_line(lines[i].m_id, from, to, std::move(text), std::move(part)));
    }
}

bool TranscriptMerger::covered(std::int64_t start, std::int64_t end) const {
    const auto at = middle(start, end);
    return std::ranges::any_of(m_spans, [&](const auto& span) {
        return at >= span.first - k_tolerance && at <= span.second + k_tolerance;
    });
}

double TranscriptMerger::covered_share(std::int64_t start, std::int64_t end) const {
    double share = 0.0;
    for (const auto& [span_start, span_end] : m_spans) {
        share += share_within(start, end, span_start - k_tolerance, span_end + k_tolerance);
    }
    return std::min(share, 1.0);
}

std::vector<TranscriptMerger::Utterance> TranscriptMerger::utterances() const {
    std::vector<Utterance> out;
    for (const auto& [id, segment] : m_segments) {
        if (segment.m_words.empty()) {
            // No word times (Moonshine): the segment is one utterance, covered by most of its
            // time.
            const Key key{id, 0};
            const auto end = std::max(segment.m_end_sample, segment.m_start_sample);
            if (segment.m_text.empty() || m_done.contains(key) ||
                (!m_kept.contains({id, 0}) &&
                 covered_share(segment.m_start_sample, end) >= k_min_time_share)) {
                continue;
            }
            out.push_back({.m_key = key,
                           .m_start = segment.m_start_sample,
                           .m_end = end,
                           .m_text = segment.m_text,
                           .m_words = {},
                           .m_settled = segment.m_is_final});
            continue;
        }
        // Timed words: the uncovered ones, cut at pauses and around covered ones.
        std::vector<Utterance> pieces;
        bool open = false;
        bool hidden = false;
        for (std::size_t w = 0; w < segment.m_words.size(); ++w) {
            const auto& word = segment.m_words[w];
            hidden = !m_kept.contains({id, w}) &&
                     covered(word.m_start_sample - m_lag, word.m_end_sample - m_lag);
            const bool pause = open && word.m_start_sample - pieces.back().m_end >= k_pause;
            if (hidden || pause) { open = false; }
            if (hidden) { continue; }
            if (!open) {
                pieces.push_back({.m_key = {id, w},
                                  .m_start = word.m_start_sample,
                                  .m_end = word.m_end_sample,
                                  .m_text = word.m_text,
                                  .m_words = {word},
                                  .m_settled = true});
                open = true;
            } else {
                auto& piece = pieces.back();
                piece.m_text += ' ';
                piece.m_text += word.m_text;
                piece.m_end = std::max(piece.m_end, word.m_end_sample);
                piece.m_words.push_back(word);
            }
        }
        // The fast engine only appends: a piece is settled once any word follows it (shown or
        // covered); the one that holds the segment's last word is once the segment is final.
        if (!pieces.empty() && !hidden) { pieces.back().m_settled = segment.m_is_final; }
        for (auto& piece : pieces) {
            if (!m_done.contains(piece.m_key)) { out.push_back(std::move(piece)); }
        }
    }
    std::ranges::sort(out,
                      [](const Utterance& a, const Utterance& b) { return a.m_start < b.m_start; });
    return out;
}

void TranscriptMerger::reconcile() {
    std::set<Key> seen;
    for (const auto& utterance : utterances()) {
        seen.insert(utterance.m_key);
        // Final once the fast engine will not revise it and, while an accurate engine runs,
        // once that is past it without covering it (by the time the fast engine reported:
        // the later one, to be safe).
        const bool final =
            utterance.m_settled && (!m_accurate || utterance.m_end + k_tolerance <= m_watermark);
        const auto open = m_open.find(utterance.m_key);
        const auto lag = lag_of(utterance);
        const auto earlier = [&](std::int64_t position) {
            return std::max<std::int64_t>(0, position - lag);
        };
        Line line{.m_id = open != m_open.end() ? open->second.m_id : 0,
                  .m_start_sample = earlier(utterance.m_start),
                  .m_end_sample = earlier(utterance.m_end),
                  .m_text = utterance.m_text,
                  .m_is_final = final,
                  .m_source = LineSource::Fast,
                  .m_words = utterance.m_words};
        for (auto& word : line.m_words) {
            word.m_start_sample = earlier(word.m_start_sample);
            word.m_end_sample = earlier(word.m_end_sample);
        }
        if (line.m_id == 0) { line.m_id = m_next_id++; }
        if (final) {
            m_done.insert(utterance.m_key);
            if (open != m_open.end()) { m_open.erase(open); }
            emit(line);
        } else if (open == m_open.end() || open->second != line) {
            m_open[utterance.m_key] = line;
            emit(line);
        }
    }
    // Lines whose text is gone: revised away by the fast engine (or covered by accurate text
    // that did not take them over).
    for (auto it = m_open.begin(); it != m_open.end();) {
        if (seen.contains(it->first)) {
            ++it;
            continue;
        }
        withdraw(it->second);
        it = m_open.erase(it);
    }
    // Forget what is resolved: final fast segments with nothing open, and accurate segments no
    // fast text can reach any more.
    for (auto it = m_segments.begin(); it != m_segments.end();) {
        const auto& key = it->first;
        const bool open = std::ranges::any_of(m_open, [&](const auto& entry) {
            return entry.first.first == key;
        });
        if (it->second.m_is_final && !open) {
            std::erase_if(m_done, [&](const Key& done) { return done.first == key; });
            std::erase_if(m_kept, [&](const WordKey& kept) { return kept.first == key; });
            it = m_segments.erase(it);
        } else {
            ++it;
        }
    }
    if (!m_spans.empty()) {
        std::int64_t oldest = m_watermark;
        for (const auto& [key, segment] : m_segments) {
            oldest = std::min(oldest, segment.m_start_sample - m_lag);
        }
        std::erase_if(m_spans,
                      [&](const auto& span) { return span.second + k_tolerance < oldest; });
    }
}

void TranscriptMerger::emit(Line line) {
    m_out.push_back(std::move(line));
}

void TranscriptMerger::withdraw(const Line& line) {
    emit({.m_id = line.m_id,
          .m_start_sample = line.m_start_sample,
          .m_end_sample = line.m_start_sample,
          .m_text = {},
          .m_is_final = false,
          .m_source = LineSource::Fast,
          .m_words = {}});
}

}  // namespace anira::models::stt
