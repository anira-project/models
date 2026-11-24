// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/Segment.h>
#include <anira/models/core/Vocabulary.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core {

/// A token a transducer emitted and the encoder frame it was emitted at.
struct Token {
    std::int32_t m_id;
    std::size_t m_frame;

    friend bool operator==(const Token&, const Token&) = default;
};

/// 16 kHz samples per encoder frame of the NeMo FastConformers (Parakeet, Nemotron ASR): 10 ms
/// feature hop x subsampling factor 8.
inline constexpr std::int64_t k_samples_per_frame = 1280;

/// The words of a transcribed segment. The words are the decoded text
/// (`vocabulary.decode(token ids)`) split at its spaces, so joined with single spaces they
/// are that text without its leading space (which decode keeps, like onnx-asr, when a bare
/// word marker opens the segment), whatever the tokens: a bare marker, punctuation with or
/// without one, special tokens. A word starts at the frame of the token that produced its first
/// character, relative to `segment_start` (16 kHz samples), and ends with the frame of the token
/// that produced its last one — or where the next word starts, if that is earlier. So a pause
/// between two words shows as a gap. Times are clamped into the segment.
[[nodiscard]] std::vector<Word> words_from_tokens(std::span<const Token> tokens,
                                                  const Vocabulary& vocabulary,
                                                  std::int64_t segment_start,
                                                  std::int64_t segment_end);

}  // namespace anira::models::core
