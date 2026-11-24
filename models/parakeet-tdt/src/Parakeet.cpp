// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/parakeet_tdt/Descriptions.h>
#include <anira/models/parakeet_tdt/Parakeet.h>
#include <anira/models/parakeet_tdt/TdtGreedyDecoder.h>
#include <anira/models/core/Vocabulary.h>
#include <anira/models/core/Words.h>
#include <tanh/dsp-basics/Resampler.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace anira::models::parakeet_tdt {

namespace {

constexpr std::size_t k_mel_bins = 128;
constexpr std::size_t k_encoder_dim = 1024;
constexpr std::size_t k_state = 2 * 640;  // [2, 1, 640]
constexpr std::int32_t k_blank = 8192;
constexpr std::size_t k_logits = 8198;  // 8192 tokens, the blank, 5 durations

anira::Tensor host(void* data, anira_dtype dtype, std::initializer_list<std::int64_t> shape) {
    return anira::Tensor::from_host(
        data, dtype, std::span<const std::int64_t>(shape.begin(), shape.size()));
}

constexpr std::array<std::string_view, k_windows.size()> k_preprocessor_json{
    k_parakeet_preprocessor_4s_model_json,
    k_parakeet_preprocessor_8s_model_json,
    k_parakeet_preprocessor_16s_model_json,
    k_parakeet_preprocessor_31s_model_json};
constexpr std::array<std::string_view, k_windows.size()> k_encoder_json{
    k_parakeet_encoder_4s_model_json,
    k_parakeet_encoder_8s_model_json,
    k_parakeet_encoder_16s_model_json,
    k_parakeet_encoder_31s_model_json};

void wait(core::Ticket& job, const char* what) {
    if (job.wait() == core::TicketStatus::Failed) {
        throw Error(std::string("Parakeet: the ") + what + " failed (" +
                    std::to_string(job.error()) + ")");
    }
}

}  // namespace

// The TDT search's joint: one decoder_joint job per step, on the encoder frame the search is at.
class Parakeet::Joint : public TdtJoint {
public:
    Joint(anira::Context& context, const core::ModelSource& weights)
        : m_handler(context, k_parakeet_decoder_joint_model_json, weights) {}

    void prepare() {
        m_handler.prepare(anira::Async{.max_in_flight = 1});
        m_handler.bind_output(0, host(m_logits.data(), ANIRA_DTYPE_F32, {1, 1, 1, k_logits}));
        m_handler.bind_output(1, host(&m_prednet_length, ANIRA_DTYPE_I32, {1}));
        m_handler.bind_output(2, host(m_next_1.data(), ANIRA_DTYPE_F32, {2, 1, 640}));
        m_handler.bind_output(3, host(m_next_2.data(), ANIRA_DTYPE_F32, {2, 1, 640}));
    }

    /// A new segment: its encoder output [1, 1024, frames of the window] and the start state.
    void begin(std::span<const float> encoded, std::size_t window_frames) {
        m_encoded = encoded;
        m_window_frames = window_frames;
        std::ranges::fill(m_state_1, 0.0F);
        std::ranges::fill(m_state_2, 0.0F);
    }

    std::span<const float> evaluate(std::size_t frame, std::int32_t previous_token) override {
        for (std::size_t c = 0; c < k_encoder_dim; ++c) {
            m_frame[c] = m_encoded[(c * m_window_frames) + frame];
        }
        m_target = previous_token;
        const std::array<anira::Tensor, 5> inputs{
            host(m_frame.data(), ANIRA_DTYPE_F32, {1, k_encoder_dim, 1}),
            host(&m_target, ANIRA_DTYPE_I32, {1, 1}),
            host(&m_target_length, ANIRA_DTYPE_I32, {1}),
            host(m_state_1.data(), ANIRA_DTYPE_F32, {2, 1, 640}),
            host(m_state_2.data(), ANIRA_DTYPE_F32, {2, 1, 640})};
        core::Ticket job = m_handler.submit(inputs);
        wait(job, "decoder");
        return m_logits;
    }

    void accept_state() override {
        m_state_1 = m_next_1;
        m_state_2 = m_next_2;
    }

private:
    core::AsyncHandler m_handler;
    std::span<const float> m_encoded;
    std::size_t m_window_frames = 0;
    std::array<float, k_encoder_dim> m_frame{};
    std::int32_t m_target = k_blank;
    std::int32_t m_target_length = 1;
    std::array<float, k_state> m_state_1{};
    std::array<float, k_state> m_state_2{};
    std::array<float, k_logits> m_logits{};
    std::int32_t m_prednet_length = 0;
    std::array<float, k_state> m_next_1{};
    std::array<float, k_state> m_next_2{};
};

Parakeet::Parakeet(anira::Context& context, const core::ModelSource& weights)
    : m_vocabulary(Vocabulary::parse(weights.read_text("vocab.txt")))
    , m_joint(std::make_unique<Joint>(context, weights))
    , m_features(k_mel_bins * k_windows.back().m_mel_frames)
    , m_encoded(k_encoder_dim * k_windows.back().m_encoder_frames) {
    for (std::size_t w = 0; w < k_windows.size(); ++w) {
        m_preprocessors[w] = std::make_unique<core::AsyncHandler>(context, k_preprocessor_json[w], weights);
        m_encoders[w] = std::make_unique<core::AsyncHandler>(context, k_encoder_json[w], weights);
    }
}

Parakeet::~Parakeet() = default;

void Parakeet::prepare() {
    for (std::size_t w = 0; w < k_windows.size(); ++w) {
        const Window& window = k_windows[w];
        const auto mel = static_cast<std::int64_t>(window.m_mel_frames);
        const auto frames = static_cast<std::int64_t>(window.m_encoder_frames);
        m_preprocessors[w]->prepare(anira::Async{.max_in_flight = 1});
        m_preprocessors[w]->bind_output(0, host(m_features.data(), ANIRA_DTYPE_F32, {1, k_mel_bins, mel}));
        m_preprocessors[w]->bind_output(1, host(&m_features_length, ANIRA_DTYPE_I64, {1}));
        m_encoders[w]->prepare(anira::Async{.max_in_flight = 1});
        m_encoders[w]->bind_output(0, host(m_encoded.data(), ANIRA_DTYPE_F32, {1, k_encoder_dim, frames}));
        m_encoders[w]->bind_output(1, host(&m_encoded_length, ANIRA_DTYPE_I64, {1}));
    }
    m_joint->prepare();
}

core::Segment Parakeet::transcribe(std::span<const float> audio, std::int64_t start_sample, std::uint64_t id) {
    if (audio.size() > k_max_samples) { throw Error("Parakeet: a segment is at most 31 s"); }
    const auto fits = std::ranges::find_if(k_windows, [&](const Window& w) { return audio.size() <= w.m_samples; });
    const auto w = static_cast<std::size_t>(fits - k_windows.begin());
    const Window& window = *fits;

    std::ranges::copy(audio, m_waveform.begin());
    std::fill(m_waveform.begin() + static_cast<std::ptrdiff_t>(audio.size()),
              m_waveform.begin() + static_cast<std::ptrdiff_t>(window.m_samples), 0.0F);
    m_waveform_length = static_cast<std::int64_t>(audio.size());
    const std::array<anira::Tensor, 2> preprocessor_inputs{
        host(m_waveform.data(), ANIRA_DTYPE_F32, {1, static_cast<std::int64_t>(window.m_samples)}),
        host(&m_waveform_length, ANIRA_DTYPE_I64, {1})};
    core::Ticket features = m_preprocessors[w]->submit(preprocessor_inputs);
    wait(features, "preprocessor");

    const std::array<anira::Tensor, 2> encoder_inputs{
        host(m_features.data(), ANIRA_DTYPE_F32, {1, k_mel_bins, static_cast<std::int64_t>(window.m_mel_frames)}),
        host(&m_features_length, ANIRA_DTYPE_I64, {1})};
    core::Ticket encoded = m_encoders[w]->submit(encoder_inputs);
    wait(encoded, "encoder");

    const auto frames = static_cast<std::size_t>(std::clamp<std::int64_t>(
        m_encoded_length, 0, static_cast<std::int64_t>(window.m_encoder_frames)));
    m_joint->begin(std::span<const float>(m_encoded.data(), k_encoder_dim * window.m_encoder_frames),
                   window.m_encoder_frames);
    const TdtConfig config{.m_blank_id = k_blank,
                           .m_num_tokens = static_cast<std::size_t>(k_blank) + 1,
                           .m_durations = {0, 1, 2, 3, 4},
                           .m_max_symbols_per_step = 10};
    const std::vector<Token> tokens = tdt_greedy_decode(*m_joint, frames, config);

    std::vector<std::int32_t> ids;
    ids.reserve(tokens.size());
    for (const Token& token : tokens) { ids.push_back(token.m_id); }
    core::Segment segment;
    segment.m_id = id;
    segment.m_start_sample = start_sample;
    segment.m_end_sample = start_sample + static_cast<std::int64_t>(audio.size());
    segment.m_text = m_vocabulary.decode(ids);
    if (!segment.m_text.empty() && segment.m_text.front() == ' ') { segment.m_text.erase(0, 1); }
    segment.m_words = core::words_from_tokens(tokens, m_vocabulary, start_sample, segment.m_end_sample);
    segment.m_is_final = true;
    return segment;
}

std::vector<core::Segment> Parakeet::run(anira::Context& context,
                                         const core::ModelSource& weights,
                                         std::span<const float> mono,
                                         double sample_rate) {
    std::vector<float> audio(mono.begin(), mono.end());
    if (sample_rate != k_sample_rate) {
        thl::dsp_basics::Resampler resampler;
        resampler.prepare(sample_rate, k_sample_rate, mono.size());
        audio.resize(resampler.max_output(mono.size()));
        audio.resize(resampler.process(mono, audio));
    }
    Parakeet parakeet(context, weights);
    parakeet.prepare();
    std::vector<core::Segment> segments;
    const double to_host = sample_rate / k_sample_rate;
    for (std::size_t start = 0; start < audio.size(); start += k_max_samples) {
        const auto count = std::min(k_max_samples, audio.size() - start);
        core::Segment segment = parakeet.transcribe(std::span<const float>(audio).subspan(start, count),
                                                    static_cast<std::int64_t>(start),
                                                    segments.size());
        const auto scale = [&](std::int64_t s) { return std::llround(static_cast<double>(s) * to_host); };
        segment.m_start_sample = scale(segment.m_start_sample);
        segment.m_end_sample = scale(segment.m_end_sample);
        for (core::Word& word : segment.m_words) {
            word.m_start_sample = scale(word.m_start_sample);
            word.m_end_sample = scale(word.m_end_sample);
        }
        segments.push_back(std::move(segment));
    }
    return segments;
}

}  // namespace anira::models::parakeet_tdt
