// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/stt/Segment.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace anira::models::stt {

/// Which engine a line's text comes from.
enum class LineSource { Fast, Accurate };

/// One line of the merged transcript, in 16 kHz samples of session time. A later Line with the
/// same id replaces it, a final one never changes again, and an empty provisional one
/// withdraws it.
struct Line {
    std::uint64_t m_id = 0;
    std::int64_t m_start_sample = 0;  ///< where its speech starts
    std::int64_t m_end_sample = 0;    ///< ... and ends (provisional: as far as heard)
    std::string m_text;
    bool m_is_final = false;
    LineSource m_source = LineSource::Fast;
    /// The text's words with their times, when the engine gave them (empty otherwise).
    std::vector<Word> m_words;

    friend bool operator==(const Line&, const Line&) = default;
};

/// Merges the fast and the accurate slot into one transcript, utterance by utterance. All
/// positions are session time (16 kHz samples): every engine's segments must already be on
/// that one clock.
///
/// - Accurate only: every segment is a final line.
/// - Fast only: the fast text is cut into utterances at pauses (by word times; a segment
///   without them is one utterance); each is a line, provisional until its engine will not
///   revise it any more.
/// - Both: the fast utterances are provisional lines, and an accurate segment replaces the
///   fast text of its time in place, word by word: the two texts are aligned (fewest edits),
///   each accurate word goes to the line of the fast word it pairs with (words the fast engine
///   missed go with their neighbours), so every line firms up where it is. Where the two
///   disagree on a word, the accurate one is taken. Where the accurate text has nothing for
///   a stretch the fast engine heard — a whole line, or k_lost_run words or more that are not
///   fillers — and no words of its own at that time either, the accurate engine lost it (it
///   does, now and then, mostly where speakers change), and the fast words stay: a whole line
///   as it is, a stretch inside a line among the accurate words. (Where it heard other words
///   at that time, those stand: the fast engine may have made its own up.) Fast words after the
///   segment stay provisional on a line of their own. Fast text no accurate segment covers — speech
///   its VAD dropped — becomes a final line of its own once the accurate engine's watermark
///   (set_accurate_watermark) is past it (source Fast). So nothing that was shown disappears.
///
/// Coverage is by time: a timed fast word is covered when its middle lies within an accurate
/// segment widened by k_tolerance; a fast segment without word times is covered when most of
/// it does. A streaming engine reports its words late; that lag is measured on the words both
/// engines agree on and taken off the fast words' times (for coverage and for the lines).
///
/// The accurate text is not taken where it is less likely right than the fast text: when it
/// lost much of the speech the fast engine heard, or when a short segment came out in another
/// writing system (the accurate engine hears a short segment without context, and may take it
/// for another language).
///
/// Engines may join and leave at any time — nothing restarts. A fast engine that leaves is
/// ended with end_fast(): its text is settled and resolved like any other (covered by the
/// accurate engine, or kept). An accurate engine that joins while someone speaks starts
/// mid-utterance: its first fragment is dropped when the fast engine heard that utterance whole
/// (set_accurate_start).
///
/// Control thread only; pure logic, no engines.
class TranscriptMerger {
public:
    /// Which slots deliver segments.
    void set_sources(bool fast, bool accurate);
    /// The accurate engine will emit nothing more that starts before `position`.
    void set_accurate_watermark(std::int64_t position);
    /// Where the accurate engine started listening (it joined, or the session began).
    void set_accurate_start(std::int64_t position);
    /// The fast engine is gone (switched off or replaced): its text will not change any more,
    /// and a new engine's segment ids start over.
    void end_fast();
    /// A new session: everything is forgotten. Line ids keep counting up.
    void clear();

    void add_fast(const Segment& segment);
    void add_accurate(const Segment& segment);

    /// The lines added, revised, finished or withdrawn since the last call, in that order.
    [[nodiscard]] std::vector<Line> take();

    /// How late the fast engine reports its words, as measured so far (16 kHz samples).
    [[nodiscard]] std::int64_t fast_lag() const noexcept { return m_lag; }

    /// Fast words this far apart (16 kHz samples) start a new utterance: just under the
    /// accurate engine's VAD pause, so an utterance rarely spans two accurate segments.
    static constexpr std::int64_t k_pause = 7200;  // 0.45 s
    /// How far the engines' times for the same speech may differ (16 kHz samples).
    static constexpr std::int64_t k_tolerance = 4800;  // 0.3 s
    /// An accurate segment starting this close after the accurate engine started listening
    /// (16 kHz samples) may have begun mid-utterance.
    static constexpr std::int64_t k_join_margin = 8000;  // 0.5 s
    /// ... it did when the fast engine heard that utterance begin at least this much earlier.
    static constexpr std::int64_t k_join_lead = 1600;  // 0.1 s
    /// This many fast words in a row (fillers aside) that the accurate text has nothing for
    /// were lost by the accurate engine: they stay.
    static constexpr std::size_t k_lost_run = 3;

private:
    /// A fast segment: which engine (epoch) and its id there.
    using SegmentKey = std::pair<std::uint64_t, std::uint64_t>;
    /// An utterance: its segment, and the index of its first word in it.
    using Key = std::pair<SegmentKey, std::size_t>;

    /// A stretch of one fast segment between pauses (and between accurate segments). Times as
    /// the fast engine reported them.
    struct Utterance {
        Key m_key;
        std::int64_t m_start = 0;
        std::int64_t m_end = 0;
        std::string m_text;
        std::vector<Word> m_words;  // empty for a segment without word times
        bool m_settled = false;     // the fast engine will not revise it any more
    };

    /// A fast word: its segment, and its index there (0 for a segment without word times,
    /// which stands for all of it).
    using WordKey = std::pair<SegmentKey, std::size_t>;

    /// A word of the fast text an accurate segment covers.
    struct FastWord {
        std::string m_text;  // normalized: lower case, no punctuation
        Word m_word;         // as the fast engine reported it
        bool m_timed = false;
        WordKey m_key;
        std::size_t m_line = 0;  // the line it is on
    };

    /// The lag to take off an utterance's times (none without word times).
    [[nodiscard]] std::int64_t lag_of(const Utterance& utterance) const noexcept;
    /// How many of the utterance's words the accurate segment [start, end] covers.
    [[nodiscard]] std::size_t covered_words(const Utterance& utterance,
                                            std::int64_t start,
                                            std::int64_t end) const;
    /// The accurate segment begins mid-utterance: the accurate engine started listening just
    /// before it, while the fast engine already heard that utterance begin.
    [[nodiscard]] bool joined_mid_utterance(const std::vector<Utterance>& pending,
                                            std::int64_t start) const;
    /// Measures the fast engine's lag on the words the segment and the fast text agree on
    /// (`partner`: the fast word each of the segment's words pairs with, -1 for none).
    void learn_lag(const Segment& segment,
                   const std::vector<FastWord>& fast,
                   const std::vector<std::ptrdiff_t>& partner);
    /// Puts the accurate segment's text in place of the lines it covers (`keys`, `lines`: in
    /// time order; `fast`: their words), word by word (see the class comment).
    void replace(const std::vector<Key>& keys,
                 const std::vector<Line>& lines,
                 const std::vector<FastWord>& fast,
                 const Segment& segment);
    /// The middle of [start, end) lies within an accurate segment (widened).
    [[nodiscard]] bool covered(std::int64_t start, std::int64_t end) const;
    /// The share of [start, end) within accurate segments (widened).
    [[nodiscard]] double covered_share(std::int64_t start, std::int64_t end) const;
    /// The fast text no accurate segment covers, cut into utterances, in time order.
    [[nodiscard]] std::vector<Utterance> utterances() const;
    /// Brings the lines in line with the fast text: new, revised, finished, withdrawn.
    void reconcile();
    void emit(Line line);
    void withdraw(const Line& line);

    bool m_fast = false;
    bool m_accurate = false;
    std::uint64_t m_next_id = 1;
    std::vector<Line> m_out;

    std::map<SegmentKey, Segment> m_segments;  // fast segments not yet resolved
    std::uint64_t m_epoch = 0;                 // the current fast engine
    std::map<Key, Line> m_open;                // provisional lines, by utterance
    std::set<Key> m_done;                      // utterances already final as fast text
    std::vector<std::pair<std::int64_t, std::int64_t>> m_spans;  // accurate segments
    std::int64_t m_watermark = 0;
    std::int64_t m_accurate_start = -1;  // where the accurate engine started listening
    std::int64_t m_lag = 0;              // how late the fast engine reports its words
    bool m_lag_measured = false;         // m_lag is the current fast engine's
    std::set<WordKey> m_kept;            // fast words the accurate engine lost: never covered
};

}  // namespace anira::models::stt
