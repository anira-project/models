// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/stt/Segment.h>
#include <anira/models/stt/TdtGreedyDecoder.h>
#include <anira/models/stt/Vocabulary.h>

#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::stt {

/// 16 kHz samples per Parakeet encoder frame: 10 ms feature hop x subsampling factor 8.
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

}  // namespace anira::models::stt
