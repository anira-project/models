// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace anira::models::core {

/// SentencePiece vocabulary of a NeMo transducer (Parakeet, Nemotron ASR): token strings indexed by
/// id, with the blank symbol as the last entry. Word starts are marked with "▁" (U+2581).
class Vocabulary {
public:
    /// `tokens[id]` is the token with that id; the last one is the blank.
    explicit Vocabulary(std::vector<std::string> tokens);

    /// Reads a "<token> <id>" per line file (scripts/export-parakeet writes vocab.txt).
    /// Throws std::runtime_error if the file is missing, malformed or ids are not 0..n-1.
    [[nodiscard]] static Vocabulary load(const std::filesystem::path& path);
    /// The same from the file's text (`what` names it in errors).
    [[nodiscard]] static Vocabulary parse(std::string_view text,
                                          const std::string& what = "vocab.txt");
    /// Reads a one-token-per-line file, the id being the line number from 0 (the
    /// onnx-community Nemotron vocab.txt). Throws std::runtime_error if missing or empty.
    [[nodiscard]] static Vocabulary load_lines(const std::filesystem::path& path);
    /// The same from the file's text.
    [[nodiscard]] static Vocabulary parse_lines(std::string_view text,
                                                const std::string& what = "vocab.txt");

    /// True for control tokens like "<unk>" or a language tag "<de-DE>", which are no text.
    [[nodiscard]] bool is_control(std::int32_t id) const;

    [[nodiscard]] std::size_t size() const noexcept { return m_tokens.size(); }

    [[nodiscard]] std::int32_t blank_id() const noexcept;

    /// Throws std::out_of_range for an id outside the vocabulary.
    [[nodiscard]] std::string_view token(std::int32_t id) const;

    /// Joins the tokens into text: "▁" becomes a space, a leading space is dropped and a
    /// space before punctuation is removed (same rule as onnx-asr, so the golden output
    /// of scripts/export-parakeet compares byte for byte).
    [[nodiscard]] std::string decode(std::span<const std::int32_t> ids) const;

private:
    std::vector<std::string> m_tokens;
};

}  // namespace anira::models::core
