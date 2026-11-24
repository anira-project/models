// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/ModelSource.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace anira::models::moonshine_streaming {

inline constexpr double k_sample_rate = 16000.0;
inline constexpr std::size_t k_frame = 320;               ///< samples per encoder frame (50 Hz)
inline constexpr std::size_t k_chunk = 25;                ///< encoder frames per step (0.5 s)
inline constexpr std::size_t k_step = k_frame * k_chunk;  ///< samples per step: 8000
inline constexpr std::size_t k_lookahead = 16;            ///< frames an encoder frame waits for
inline constexpr std::size_t k_memory = 800;              ///< encoder frames per segment (16 s)
inline constexpr std::size_t k_tokens = 128;              ///< decoder positions per segment
inline constexpr double k_tokens_per_second = 6.5;
inline constexpr std::int32_t k_bos = 1;
inline constexpr std::int32_t k_eos = 2;
/// The longest segment: 15 s plus the voice activity detector's look-behind of 0.512 s, whole
/// steps.
inline constexpr std::size_t k_max_steps = 31;

/// One of the four models of anira-project/moonshine-streaming.
enum class Variant { TinyEn, SmallEn, TinyDe, SmallDe };
std::string_view folder_of(Variant variant) noexcept;

/// Moonshine's tokenizer.bin: per token id its bytes (a length of one byte below 128, else two).
class Tokenizer {
public:
    explicit Tokenizer(std::span<const std::byte> data);
    /// The text of `tokens`: their bytes, special tokens (<...>) left out, "▁" as a space, trimmed.
    [[nodiscard]] std::string text(std::span<const std::int32_t> tokens) const;
    [[nodiscard]] std::size_t size() const noexcept { return m_tokens.size(); }

private:
    std::vector<std::string> m_tokens;
};

/// One segment through Moonshine's two graphs, as moonshine-streaming/export/moonshine.py's Segment
/// drives them: 0.5 s steps of 16 kHz audio into the encoder (its frames become the
/// cross-attention memory), a flush at the end, greedy decoding of the memory so far. Every call
/// submits a job and waits for it (core::AsyncHandler): a decoding thread's object, never the audio
/// thread's.
class SegmentDecoder {
public:
    SegmentDecoder(anira::Context& context, const core::ModelSource& weights, Variant variant);
    ~SegmentDecoder();
    SegmentDecoder(const SegmentDecoder&) = delete;
    SegmentDecoder& operator=(const SegmentDecoder&) = delete;
    SegmentDecoder(SegmentDecoder&&) = delete;
    SegmentDecoder& operator=(SegmentDecoder&&) = delete;

    /// Main thread: both graphs' Async contracts (no deadline: every step is the transcript's).
    /// The encoder's plan report (both graphs list the same engines: one plan index for both).
    anira::PlanReport prepare();
    /// Any thread: both graphs' plan.
    anira_status set_plan(std::uint32_t plan) noexcept;
    [[nodiscard]] anira::PlanReport plan_report() const { return m_encoder->plan_report(); }
    /// A new segment: the encoder's state and the memory start over.
    void begin();
    /// The next 0.5 s (k_step samples, 16 kHz). Returns false once the segment is full.
    bool push(std::span<const float> step);
    /// The end of the segment: the encoder's last frames (its lookahead).
    void flush();
    /// Greedy tokens for the memory so far, at most `seconds` x 6.5 of them.
    std::vector<std::int32_t> decode(double seconds);

    [[nodiscard]] std::size_t steps() const noexcept { return m_frames / k_chunk; }
    [[nodiscard]] std::size_t memory() const noexcept { return m_memory; }
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return *m_tokenizer; }

private:
    void encode(std::span<const float> samples, std::int32_t position);

    std::unique_ptr<Tokenizer> m_tokenizer;
    std::unique_ptr<core::AsyncHandler> m_encoder;
    std::unique_ptr<core::AsyncHandler> m_decoder;
    std::size_t m_layers = 0;
    std::size_t m_heads = 0;
    std::size_t m_head_dim = 0;
    std::size_t m_vocab = 0;

    // Encoder job.
    std::vector<float> m_audio = std::vector<float>(k_step);
    std::int32_t m_position = 0;
    std::int32_t m_length = 0;
    std::vector<float> m_k_new;  // [L,1,H,25,K]
    std::vector<float> m_v_new;
    // The segment's memory and the decoder job.
    std::vector<float> m_k_cross;  // [L,1,H,800,K]
    std::vector<float> m_v_cross;
    std::vector<float> m_k_self;  // [L,1,H,128,K]
    std::vector<float> m_v_self;
    std::int32_t m_token = 0;
    std::int32_t m_token_position = 0;
    std::int32_t m_memory_length = 0;
    std::vector<float> m_logits;
    std::vector<float> m_k_token;  // [L,1,H,1,K]
    std::vector<float> m_v_token;

    std::size_t m_frames = 0;  // encoder frames computed from audio
    std::size_t m_memory = 0;  // frames in the cross-attention memory
};

}  // namespace anira::models::moonshine_streaming
