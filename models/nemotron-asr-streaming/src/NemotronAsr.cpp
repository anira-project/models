// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/LogMel.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/core/Vocabulary.h>
#include <anira/models/core/Words.h>
#include <anira/models/nemotron_asr_streaming/Descriptions.h>
#include <anira/models/nemotron_asr_streaming/NemotronAsr.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace anira::models::nemotron_asr_streaming {

namespace {

constexpr std::size_t k_mels = core::LogMel::k_mels;
constexpr std::size_t k_hidden = 1024;
constexpr std::size_t k_decoder = 640;
constexpr std::size_t k_lstm = 2 * k_decoder;
constexpr std::size_t k_vocab = 13088;
constexpr std::int64_t k_blank = 13087;
constexpr int k_max_symbols = 10;  // per encoder frame
constexpr std::size_t k_audio_queue = 64 * 16000 / 256;
constexpr std::size_t k_segment_queue = 1024;
constexpr auto k_poll = std::chrono::milliseconds(5);

// nvidia/nemotron-3.5-asr-streaming-0.6b, processor_config.json: prompt_dictionary.
constexpr std::array<std::pair<std::string_view, std::int64_t>, 121> k_languages{{
    {"af-ZA", 54},  {"am-ET", 49},  {"ar", 7},      {"ar-AR", 7},  {"auto", 101},  {"ay-BO", 81},
    {"az-AZ", 66},  {"bg", 30},     {"bg-BG", 30},  {"bn-IN", 36}, {"cs", 22},     {"cs-CZ", 22},
    {"da", 25},     {"da-DK", 25},  {"de", 9},      {"de-DE", 9},  {"el", 21},     {"el-GR", 21},
    {"en", 0},      {"en-GB", 1},   {"en-US", 0},   {"enGB", 1},   {"es", 3},      {"es-ES", 2},
    {"es-US", 3},   {"esES", 2},    {"et", 60},     {"et-EE", 60}, {"fa-IR", 38},  {"fi", 26},
    {"fi-FI", 26},  {"fr", 8},      {"fr-CA", 100}, {"fr-FR", 8},  {"gn-PY", 82},  {"gu-IN", 42},
    {"ha-NG", 50},  {"haw-US", 97}, {"he-IL", 64},  {"hi", 6},     {"hi-HI", 6},   {"hi-IN", 6},
    {"hr", 29},     {"hr-HR", 29},  {"hu", 23},     {"hu-HU", 23}, {"hy-AM", 68},  {"id-ID", 34},
    {"ig-NG", 53},  {"it", 15},     {"it-IT", 15},  {"ja-JA", 10}, {"ja-JP", 10},  {"ka-GE", 67},
    {"km-KH", 47},  {"kn-IN", 43},  {"ko", 14},     {"ko-KO", 14}, {"ko-KR", 14},  {"ku-TR", 65},
    {"ky-KG", 71},  {"ln-CD", 58},  {"lt", 31},     {"lt-LT", 31}, {"lv", 61},     {"lv-LV", 61},
    {"mi-NZ", 96},  {"ml-IN", 44},  {"mr-IN", 41},  {"ms-MY", 35}, {"mt-MT", 102}, {"nah-MX", 83},
    {"nb", 103},    {"nb-NO", 103}, {"ne-NP", 46},  {"nl", 16},    {"nl-NL", 16},  {"nn", 104},
    {"nn-NO", 104}, {"no", 27},     {"no-NO", 27},  {"ny-MW", 57}, {"or-KE", 59},  {"pl", 17},
    {"pl-PL", 17},  {"pt", 13},     {"pt-BR", 12},  {"pt-PT", 13}, {"qu-PE", 80},  {"ro", 20},
    {"ro-RO", 20},  {"ru", 11},     {"ru-RU", 11},  {"rw-RW", 55}, {"si-LK", 45},  {"sk", 28},
    {"sk-SK", 28},  {"sl", 62},     {"sl-SI", 62},  {"sm-WS", 98}, {"so-SO", 56},  {"sv", 24},
    {"sv-SE", 24},  {"sw-KE", 48},  {"ta-IN", 39},  {"te-IN", 40}, {"tg-TJ", 70},  {"th-TH", 32},
    {"to-TO", 99},  {"tr", 18},     {"tr-TR", 18},  {"uk", 19},    {"uk-UA", 19},  {"ur-PK", 37},
    {"uz-UZ", 69},  {"vi-VN", 33},  {"yo-NG", 52},  {"zh-CN", 4},  {"zh-TW", 5},   {"zh-ZH", 4},
    {"zu-ZA", 51},
}};

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

void wait(core::Ticket& job, const char* what) {
    if (job.wait() == core::TicketStatus::Failed) {
        throw Error(std::string("NemotronAsr: the ") + what + " failed (" +
                    std::to_string(job.error()) + ")");
    }
}

}  // namespace

std::optional<std::int64_t> language_id(std::string_view language) noexcept {
    const auto* found =
        std::ranges::find(k_languages, language, &std::pair<std::string_view, std::int64_t>::first);
    if (found == k_languages.end()) { return std::nullopt; }
    return found->second;
}

NemotronAsr::NemotronAsr(anira::Context& context,
                         const core::ModelSource& weights,
                         AsrConfig config)
    : m_config(std::move(config))
    , m_vocabulary(core::Vocabulary::parse_lines(weights.read_text("vocab.txt")))
    , m_encoder(std::make_unique<core::AsyncHandler>(context,
                                                     k_nemotron_asr_streaming_encoder_model_json,
                                                     weights,
                                                     m_config.m_threads))
    , m_decoder(std::make_unique<core::AsyncHandler>(context,
                                                     k_nemotron_asr_streaming_decoder_model_json,
                                                     weights,
                                                     1))
    , m_joint(std::make_unique<core::AsyncHandler>(context,
                                                   k_nemotron_asr_streaming_joint_model_json,
                                                   weights,
                                                   1))
    , m_mels((k_step_mels + 64) * k_mels)
    , m_features(k_step_mels * k_mels)
    , m_encoded(k_step_frames * k_hidden)
    , m_h(k_lstm)
    , m_c(k_lstm)
    , m_h_next(k_lstm)
    , m_c_next(k_lstm)
    , m_decoder_out(k_decoder)
    , m_logits(k_vocab) {
    const auto id = language_id(m_config.m_language);
    if (!id) { throw Error("NemotronAsr: unknown language '" + m_config.m_language + "'"); }
    m_language = *id;
    if (m_vocabulary.size() != k_vocab) {
        throw Error("NemotronAsr: vocab.txt has the wrong size");
    }
    m_audio.reserve(k_audio_queue);
    m_segments.reserve(k_segment_queue);
}

NemotronAsr::~NemotronAsr() {
    stop();
}

void NemotronAsr::stop() {
    m_running.store(false);
    if (m_thread.joinable()) { m_thread.join(); }
}

void NemotronAsr::prepare(double sample_rate, std::size_t max_block) {
    stop();
    m_encoder->prepare(anira::Async{.max_in_flight = 1});
    m_encoder->bind_output(0,
                           host(m_encoded.data(), ANIRA_DTYPE_F32, {1, k_step_frames, k_hidden}));
    m_encoder->bind_output(1, host(&m_encoded_length, ANIRA_DTYPE_I64, {1}));
    m_decoder->prepare(anira::Async{.max_in_flight = 1});
    m_decoder->bind_output(0, host(m_decoder_out.data(), ANIRA_DTYPE_F32, {1, k_decoder, 1}));
    m_decoder->bind_output(1, host(m_h_next.data(), ANIRA_DTYPE_F32, {2, 1, k_decoder}));
    m_decoder->bind_output(2, host(m_c_next.data(), ANIRA_DTYPE_F32, {2, 1, k_decoder}));
    m_joint->prepare(anira::Async{.max_in_flight = 1});
    m_joint->bind_output(0, host(m_logits.data(), ANIRA_DTYPE_F32, {1, 1, 1, k_vocab}));
    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_resampler.prepare(sample_rate, k_sample_rate, max_block);
    m_resampled.assign(m_resampler.max_output(max_block), 0.0F);
    m_mel.prepare(k_block);
    m_new_mels.assign(core::LogMel::max_frames(k_block) * k_mels, 0.0F);
    reset();
}

void NemotronAsr::reset() {
    stop();
    m_resampler.reset();
    m_pending = {};
    m_audio.clear();
    m_segments.clear();
    m_latest.reset();
    m_dropped.store(0);
    m_dropped_audio.store(0);
    m_failed.store(0);
    m_queued.store(0);
    m_taken.store(0);
    m_model_samples = 0;
    m_segment_id = 0;
    m_start_over = false;
    start_over();
    m_running.store(true);
    m_thread = std::thread([this] { loop(); });
}

// The model from the start: the encoder's caches, the LSTM state, the front end, an empty segment.
void NemotronAsr::start_over() {
    m_encoder->reset();
    m_mel.reset();
    std::ranges::fill(m_mels, 0.0F);  // the first step's cached frames are zeros
    m_mel_count = 0;
    m_base = m_model_samples;
    m_frames = 0;
    m_segment_frame = 0;
    m_tokens.clear();
    std::ranges::fill(m_h, 0.0F);
    std::ranges::fill(m_c, 0.0F);
    run_decoder(k_blank);
}

anira_status NemotronAsr::process(std::span<const float> mono) noexcept {
    if (mono.size() > m_max_block) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const std::size_t produced = m_resampler.process(mono, m_resampled);
    anira_status status = ANIRA_OK;
    for (std::size_t i = 0; i < produced; ++i) {
        m_pending.m_samples[m_pending.m_count++] = m_resampled[i];
        if (m_pending.m_count == k_block) {
            if (m_audio.push(m_pending)) {
                m_queued.fetch_add(1, std::memory_order_release);
            } else {
                m_dropped_audio.fetch_add(k_block, std::memory_order_relaxed);
                status = ANIRA_ERROR_CAPACITY;
            }
            m_pending.m_count = 0;
        }
    }
    return status;
}

void NemotronAsr::end_segment() noexcept {
    m_pending.m_end = true;
    if (m_audio.push(m_pending)) { m_queued.fetch_add(1, std::memory_order_release); }
    m_pending = {};
}

bool NemotronAsr::idle() const noexcept {
    return m_taken.load(std::memory_order_acquire) == m_queued.load(std::memory_order_acquire);
}

std::int64_t NemotronAsr::to_host(std::int64_t model_sample) const noexcept {
    return std::llround(static_cast<double>(model_sample) * m_host_rate / k_sample_rate);
}

void NemotronAsr::loop() {
    Block block;
    while (m_running.load()) {
        if (!m_audio.pop(block)) {
            std::this_thread::sleep_for(k_poll);
            continue;
        }
        try {
            m_model_samples += block.m_count;
            if (m_start_over) {  // the last start over failed: the model state is not clean
                start_over();
                m_start_over = false;
            } else {
                take(std::span<const float>(block.m_samples.data(), block.m_count));
            }
            if (block.m_end) {
                // The last frames: the right half window of zeros (the frames centred near the
                // audio's end), then a short step over what is left.
                const std::array<float, core::LogMel::k_fft / 2> silence{};
                take(silence);
                if (m_mel_count > 0) { step(m_mel_count); }
                finish_segment(true);
            }
        } catch (const std::exception&) {
            // A failed job: the open segment ends with the text it has (its revisions were given)
            // and the model starts over after the lost audio; failing that, at the next block.
            m_failed.fetch_add(1, std::memory_order_relaxed);
            try {
                emit(true);
                start_over();
                m_start_over = false;
            } catch (const std::exception&) { m_start_over = true; }
        }
        m_taken.fetch_add(1, std::memory_order_release);
    }
}

// Model samples into mel frames; a step every 56 new frames.
void NemotronAsr::take(std::span<const float> samples) {
    const std::size_t made = m_mel.process(samples, m_new_mels);
    std::copy_n(
        m_new_mels.begin(),
        made * k_mels,
        m_mels.begin() + static_cast<std::ptrdiff_t>((k_cache_mels + m_mel_count) * k_mels));
    m_mel_count += made;
    while (m_mel_count >= k_new_mels) {
        step(k_new_mels);
        emit(false);
        const auto samples_in =
            static_cast<double>((m_frames - m_segment_frame) * core::k_samples_per_frame);
        if (samples_in / k_sample_rate >= m_config.m_max_segment) { finish_segment(false); }
    }
}

// One encoder step: the 9 cached frames and `new_mels` new ones (zeros after them), then the RNNT
// loop over its frames.
void NemotronAsr::step(std::size_t new_mels) {
    const std::size_t used = k_cache_mels + new_mels;
    std::copy_n(m_mels.begin(), used * k_mels, m_features.begin());
    std::fill(m_features.begin() + static_cast<std::ptrdiff_t>(used * k_mels),
              m_features.end(),
              0.0F);
    m_length = static_cast<std::int64_t>(used);
    const std::array<anira::Tensor, 6> inputs{
        host(m_features.data(), ANIRA_DTYPE_F32, {1, k_step_mels, k_mels}),
        host(&m_length, ANIRA_DTYPE_I64, {1}),
        none(ANIRA_DTYPE_F32),
        none(ANIRA_DTYPE_F32),
        none(ANIRA_DTYPE_I64),
        host(&m_language, ANIRA_DTYPE_I64, {1})};
    core::Ticket job = m_encoder->submit(inputs);
    wait(job, "encoder");
    const auto frames =
        static_cast<std::size_t>(std::clamp<std::int64_t>(m_encoded_length, 0, k_step_frames));
    decode(frames);
    m_frames += static_cast<std::int64_t>(frames);

    // The step's last 9 frames become the next step's cached ones.
    const std::size_t left = k_cache_mels + m_mel_count;
    std::copy(m_mels.begin() + static_cast<std::ptrdiff_t>(new_mels * k_mels),
              m_mels.begin() + static_cast<std::ptrdiff_t>(left * k_mels),
              m_mels.begin());
    m_mel_count -= new_mels;
}

// Greedy RNNT: per encoder frame, tokens until the joint network says blank (at most 10).
void NemotronAsr::decode(std::size_t frames) {
    for (std::size_t t = 0; t < frames; ++t) {
        for (int symbol = 0; symbol < k_max_symbols; ++symbol) {
            const std::array<anira::Tensor, 2> inputs{
                host(m_encoded.data() + (t * k_hidden), ANIRA_DTYPE_F32, {1, 1, k_hidden}),
                host(m_decoder_out.data(), ANIRA_DTYPE_F32, {1, 1, k_decoder})};
            core::Ticket job = m_joint->submit(inputs);
            wait(job, "joint network");
            const auto best =
                static_cast<std::int64_t>(std::ranges::max_element(m_logits) - m_logits.begin());
            if (best == k_blank) { break; }
            m_tokens.push_back({.m_id = static_cast<std::int32_t>(best),
                                .m_frame = static_cast<std::size_t>(m_frames - m_segment_frame +
                                                                    static_cast<std::int64_t>(t))});
            std::ranges::copy(m_h_next, m_h.begin());  // a copy: the decoder's outputs stay bound
            std::ranges::copy(m_c_next, m_c.begin());
            run_decoder(best);
        }
    }
}

// The prediction network after `token` from the committed state: its output for the joint
// network, and the state it moves to once a token is emitted.
void NemotronAsr::run_decoder(std::int64_t token) {
    m_target = token;
    const std::array<anira::Tensor, 3> inputs{host(&m_target, ANIRA_DTYPE_I64, {1, 1}),
                                              host(m_h.data(), ANIRA_DTYPE_F32, {2, 1, k_decoder}),
                                              host(m_c.data(), ANIRA_DTYPE_F32, {2, 1, k_decoder})};
    core::Ticket job = m_decoder->submit(inputs);
    wait(job, "decoder");
}

// The open segment ends: at the audio's end (end_segment; the model starts over) or at the step
// boundary (the length limit; the stream goes on).
void NemotronAsr::finish_segment(bool end) {
    emit(true, end);
    if (end) {
        start_over();
    } else {
        m_segment_frame = m_frames;
        m_tokens.clear();
    }
}

void NemotronAsr::emit(bool final, bool at_end) {
    std::vector<core::Token> text_tokens;
    text_tokens.reserve(m_tokens.size());
    for (const core::Token& token : m_tokens) {
        if (!m_vocabulary.is_control(token.m_id)) { text_tokens.push_back(token); }
    }
    if (text_tokens.empty()) { return; }
    const std::int64_t start = m_base + (m_segment_frame * core::k_samples_per_frame);
    const std::int64_t stepped = m_base + (m_frames * core::k_samples_per_frame);
    // A final segment at the audio's end runs to the last sample; every other to the last step.
    const std::int64_t end = at_end ? std::max(m_model_samples, start) : stepped;
    std::vector<std::int32_t> ids;
    ids.reserve(text_tokens.size());
    for (const core::Token& token : text_tokens) { ids.push_back(token.m_id); }
    core::Segment segment;
    segment.m_id = m_segment_id;
    segment.m_start_sample = to_host(start);
    segment.m_end_sample = to_host(end);
    segment.m_text = m_vocabulary.decode(ids);
    if (!segment.m_text.empty() && segment.m_text.front() == ' ') { segment.m_text.erase(0, 1); }
    segment.m_words = core::words_from_tokens(text_tokens, m_vocabulary, start, end);
    for (core::Word& word : segment.m_words) {
        word.m_start_sample = to_host(word.m_start_sample);
        word.m_end_sample = to_host(word.m_end_sample);
    }
    segment.m_is_final = final;
    m_latest.write(segment);
    if (!m_segments.push(segment)) { m_dropped.fetch_add(1, std::memory_order_relaxed); }
    if (final) { ++m_segment_id; }
}

std::vector<core::Segment> NemotronAsr::run(anira::Context& context,
                                            const core::ModelSource& weights,
                                            std::span<const float> mono,
                                            double sample_rate,
                                            AsrConfig config) {
    constexpr std::size_t k_host_block = 4096;
    NemotronAsr asr(context, weights, std::move(config));
    asr.prepare(sample_rate, k_host_block);
    std::vector<core::Segment> out;
    core::Segment segment;
    const auto take = [&] {
        while (asr.pop(segment)) {
            if (segment.m_is_final) { out.push_back(segment); }
        }
    };
    for (std::size_t start = 0; start < mono.size(); start += k_host_block) {
        asr.process(mono.subspan(start, std::min(k_host_block, mono.size() - start)));
        while (!asr.idle()) {
            take();
            std::this_thread::sleep_for(k_poll);
        }
    }
    asr.end_segment();
    while (!asr.idle()) {
        take();
        std::this_thread::sleep_for(k_poll);
    }
    take();
    return out;
}

}  // namespace anira::models::nemotron_asr_streaming
