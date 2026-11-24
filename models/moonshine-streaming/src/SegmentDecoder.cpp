// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/moonshine_streaming/Descriptions.h>
#include <anira/models/moonshine_streaming/SegmentDecoder.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace anira::models::moonshine_streaming {

namespace {

anira::Tensor host(void* data, anira_dtype dtype, std::initializer_list<std::int64_t> shape) {
    return anira::Tensor::from_host(data,
                                    dtype,
                                    std::span<const std::int64_t>(shape.begin(), shape.size()));
}

anira::Tensor none(anira_dtype dtype) {
    const std::array<std::int64_t, 1> shape{0};
    anira::Tensor tensor{};
    anira_tensor_init_host(&tensor, nullptr, dtype, 1, shape.data());
    return tensor;
}

struct Graphs {
    std::string_view m_encoder;
    std::string_view m_decoder;
};

Graphs graphs_of(Variant variant) {
    switch (variant) {
        case Variant::TinyEn:
            return {.m_encoder = k_moonshine_tiny_en_encoder_model_json,
                    .m_decoder = k_moonshine_tiny_en_decoder_model_json};
        case Variant::SmallEn:
            return {.m_encoder = k_moonshine_small_en_encoder_model_json,
                    .m_decoder = k_moonshine_small_en_decoder_model_json};
        case Variant::TinyDe:
            return {.m_encoder = k_moonshine_tiny_de_encoder_model_json,
                    .m_decoder = k_moonshine_tiny_de_decoder_model_json};
        case Variant::SmallDe:
            return {.m_encoder = k_moonshine_small_de_encoder_model_json,
                    .m_decoder = k_moonshine_small_de_decoder_model_json};
    }
    return {};
}

// The extents of a tensor of a description.
std::vector<std::int64_t> extents(std::string_view description,
                                  const char* side,
                                  const char* name) {
    const auto json = nlohmann::json::parse(description);
    for (const auto& tensor : json.at(side)) {
        if (tensor.at("name") == name) {
            std::vector<std::int64_t> out;
            for (const auto& axis : tensor.at("axes")) {
                out.push_back(axis.at(1).get<std::int64_t>());
            }
            return out;
        }
    }
    throw Error(std::string("Moonshine: no tensor ") + name);
}

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(' ');
    if (first == std::string::npos) { return {}; }
    const auto last = text.find_last_not_of(' ');
    return text.substr(first, last - first + 1);
}

}  // namespace

std::string_view folder_of(Variant variant) noexcept {
    switch (variant) {
        case Variant::TinyEn: return "tiny-en";
        case Variant::SmallEn: return "small-en";
        case Variant::TinyDe: return "tiny-de";
        case Variant::SmallDe: return "small-de";
    }
    return {};
}

// ---------------------------------------------------------------------------------------------
// Tokenizer

Tokenizer::Tokenizer(std::span<const std::byte> data) {
    std::size_t offset = 0;
    const auto byte = [&](std::size_t at) { return static_cast<std::uint8_t>(data[at]); };
    while (offset < data.size()) {
        std::size_t length = byte(offset++);
        if (length >= 128) {
            if (offset >= data.size()) { throw Error("Moonshine tokenizer: truncated length"); }
            length = (static_cast<std::size_t>(byte(offset++)) * 128) + length - 128;
        }
        if (length > data.size() - offset) { throw Error("Moonshine tokenizer: truncated token"); }
        m_tokens.emplace_back(reinterpret_cast<const char*>(data.data() + offset), length);
        offset += length;
    }
    if (m_tokens.empty()) { throw Error("Moonshine tokenizer: no tokens"); }
}

std::string Tokenizer::text(std::span<const std::int32_t> tokens) const {
    static constexpr std::string_view k_space = "\xE2\x96\x81";  // ▁
    std::string joined;
    for (const std::int32_t token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= m_tokens.size()) { continue; }
        const std::string& bytes = m_tokens[static_cast<std::size_t>(token)];
        if (bytes.size() > 2 && bytes.front() == '<' && bytes.back() == '>') { continue; }
        joined += bytes;
    }
    std::string text;
    for (std::size_t i = 0; i < joined.size();) {
        if (joined.compare(i, k_space.size(), k_space) == 0) {
            text += ' ';
            i += k_space.size();
        } else {
            text += joined[i++];
        }
    }
    return trim(text);
}

// ---------------------------------------------------------------------------------------------
// SegmentDecoder

SegmentDecoder::SegmentDecoder(anira::Context& context,
                               const core::ModelSource& weights,
                               Variant variant) {
    const Graphs graphs = graphs_of(variant);
    const std::string tokenizer =
        weights.read_text(std::string(folder_of(variant)) + "/tokenizer.bin");
    m_tokenizer = std::make_unique<Tokenizer>(
        std::as_bytes(std::span<const char>(tokenizer.data(), tokenizer.size())));
    m_encoder = std::make_unique<core::AsyncHandler>(context, graphs.m_encoder, weights);
    m_decoder = std::make_unique<core::AsyncHandler>(context, graphs.m_decoder, weights);

    const auto cache = extents(graphs.m_decoder, "inputs", "k_self");  // [L,1,H,128,K]
    m_layers = static_cast<std::size_t>(cache[0]);
    m_heads = static_cast<std::size_t>(cache[2]);
    m_head_dim = static_cast<std::size_t>(cache[4]);
    m_vocab = static_cast<std::size_t>(extents(graphs.m_decoder, "outputs", "logits")[1]);
    const std::size_t lanes = m_layers * m_heads * m_head_dim;
    m_k_new.assign(lanes * k_chunk, 0.0F);
    m_v_new.assign(lanes * k_chunk, 0.0F);
    m_k_cross.assign(lanes * k_memory, 0.0F);
    m_v_cross.assign(lanes * k_memory, 0.0F);
    m_k_self.assign(lanes * k_tokens, 0.0F);
    m_v_self.assign(lanes * k_tokens, 0.0F);
    m_logits.assign(m_vocab, 0.0F);
    m_k_token.assign(lanes, 0.0F);
    m_v_token.assign(lanes, 0.0F);
}

SegmentDecoder::~SegmentDecoder() = default;

anira::PlanReport SegmentDecoder::prepare() {
    const auto l = static_cast<std::int64_t>(m_layers);
    const auto h = static_cast<std::int64_t>(m_heads);
    const auto k = static_cast<std::int64_t>(m_head_dim);
    anira::PlanReport report = m_encoder->prepare(anira::Async{.max_in_flight = 1});
    m_encoder->bind_output(0, host(m_k_new.data(), ANIRA_DTYPE_F32, {l, 1, h, k_chunk, k}));
    m_encoder->bind_output(1, host(m_v_new.data(), ANIRA_DTYPE_F32, {l, 1, h, k_chunk, k}));
    m_decoder->prepare(anira::Async{.max_in_flight = 1});
    m_decoder->bind_output(
        0,
        host(m_logits.data(), ANIRA_DTYPE_F32, {1, static_cast<std::int64_t>(m_vocab)}));
    m_decoder->bind_output(1, host(m_k_token.data(), ANIRA_DTYPE_F32, {l, 1, h, 1, k}));
    m_decoder->bind_output(2, host(m_v_token.data(), ANIRA_DTYPE_F32, {l, 1, h, 1, k}));
    begin();
    return report;
}

anira_status SegmentDecoder::set_plan(std::uint32_t plan) noexcept {
    const anira_status encoder = m_encoder->set_plan(plan);
    const anira_status decoder = m_decoder->set_plan(plan);
    return ANIRA_FAILED(encoder) ? encoder : decoder;
}

void SegmentDecoder::begin() {
    m_encoder->reset();
    std::ranges::fill(m_k_cross, 0.0F);
    std::ranges::fill(m_v_cross, 0.0F);
    m_frames = 0;
    m_memory = 0;
}

bool SegmentDecoder::push(std::span<const float> step) {
    if (steps() >= k_max_steps || step.size() != k_step) { return false; }
    m_frames += k_chunk;
    encode(step,
           static_cast<std::int32_t>(m_frames) - static_cast<std::int32_t>(k_chunk + k_lookahead));
    return true;
}

void SegmentDecoder::flush() {
    const std::vector<float> silence(k_step, 0.0F);
    encode(silence, static_cast<std::int32_t>(m_frames) - static_cast<std::int32_t>(k_lookahead));
}

void SegmentDecoder::encode(std::span<const float> samples, std::int32_t position) {
    std::ranges::copy(samples, m_audio.begin());
    m_position = position;
    m_length = static_cast<std::int32_t>(m_frames);
    const std::array<anira::Tensor, 6> inputs{
        host(m_audio.data(), ANIRA_DTYPE_F32, {1, static_cast<std::int64_t>(k_step)}),
        none(ANIRA_DTYPE_F32),
        none(ANIRA_DTYPE_F32),
        none(ANIRA_DTYPE_F32),
        host(&m_position, ANIRA_DTYPE_I32, {1}),
        host(&m_length, ANIRA_DTYPE_I32, {1})};
    core::Ticket job = m_encoder->submit(inputs);
    if (job.wait() == core::TicketStatus::Failed) {
        throw Error("Moonshine: the encoder failed (" + std::to_string(job.error()) + ")");
    }
    // The frames that became final, in order, into the memory.
    for (std::size_t j = 0; j < k_chunk; ++j) {
        const std::int64_t frame = position + static_cast<std::int64_t>(j);
        if (frame < 0 || std::cmp_greater_equal(frame, m_frames) ||
            std::cmp_not_equal(frame, m_memory) || m_memory >= k_memory) {
            continue;
        }
        for (std::size_t lane = 0; lane < m_layers * m_heads; ++lane) {
            const std::size_t from = ((lane * k_chunk) + j) * m_head_dim;
            const std::size_t to = ((lane * k_memory) + m_memory) * m_head_dim;
            std::copy_n(m_k_new.begin() + static_cast<std::ptrdiff_t>(from),
                        m_head_dim,
                        m_k_cross.begin() + static_cast<std::ptrdiff_t>(to));
            std::copy_n(m_v_new.begin() + static_cast<std::ptrdiff_t>(from),
                        m_head_dim,
                        m_v_cross.begin() + static_cast<std::ptrdiff_t>(to));
        }
        ++m_memory;
    }
}

std::vector<std::int32_t> SegmentDecoder::decode(double seconds) {
    const auto limit =
        std::min<std::size_t>(static_cast<std::size_t>(std::ceil(seconds * k_tokens_per_second)),
                              k_tokens - 1);
    const auto l = static_cast<std::int64_t>(m_layers);
    const auto h = static_cast<std::int64_t>(m_heads);
    const auto k = static_cast<std::int64_t>(m_head_dim);
    std::ranges::fill(m_k_self, 0.0F);
    std::ranges::fill(m_v_self, 0.0F);
    m_memory_length = static_cast<std::int32_t>(m_memory);
    std::vector<std::int32_t> tokens;
    m_token = k_bos;
    for (std::size_t position = 0; position < limit; ++position) {
        m_token_position = static_cast<std::int32_t>(position);
        const std::array<anira::Tensor, 7> inputs{
            host(&m_token, ANIRA_DTYPE_I32, {1, 1}),
            host(&m_token_position, ANIRA_DTYPE_I32, {1}),
            host(m_k_self.data(), ANIRA_DTYPE_F32, {l, 1, h, k_tokens, k}),
            host(m_v_self.data(), ANIRA_DTYPE_F32, {l, 1, h, k_tokens, k}),
            host(m_k_cross.data(), ANIRA_DTYPE_F32, {l, 1, h, k_memory, k}),
            host(m_v_cross.data(), ANIRA_DTYPE_F32, {l, 1, h, k_memory, k}),
            host(&m_memory_length, ANIRA_DTYPE_I32, {1})};
        core::Ticket job = m_decoder->submit(inputs);
        if (job.wait() == core::TicketStatus::Failed) {
            throw Error("Moonshine: the decoder failed (" + std::to_string(job.error()) + ")");
        }
        for (std::size_t lane = 0; lane < m_layers * m_heads; ++lane) {
            const std::size_t to = ((lane * k_tokens) + position) * m_head_dim;
            std::copy_n(m_k_token.begin() + static_cast<std::ptrdiff_t>(lane * m_head_dim),
                        m_head_dim,
                        m_k_self.begin() + static_cast<std::ptrdiff_t>(to));
            std::copy_n(m_v_token.begin() + static_cast<std::ptrdiff_t>(lane * m_head_dim),
                        m_head_dim,
                        m_v_self.begin() + static_cast<std::ptrdiff_t>(to));
        }
        const auto best = std::ranges::max_element(m_logits) - m_logits.begin();
        m_token = static_cast<std::int32_t>(best);
        if (m_token == k_eos) { break; }
        tokens.push_back(m_token);
    }
    return tokens;
}

}  // namespace anira::models::moonshine_streaming
