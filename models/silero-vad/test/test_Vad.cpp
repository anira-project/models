// SPDX-License-Identifier: Apache-2.0

// Silero VAD's wrapper: the clock and the aligned output exact to the sample, the probabilities
// those of ONNX Runtime on the same audio (test/vad_golden.json on Hugging Face), the same values at 48 kHz as at
// 16 kHz, the offline run the same segments as streaming. The tests with the model need its
// weights: ANIRA_MODELS_TEST_WEIGHTS/silero-vad (a checkout of the HF repository at the catalog's
// revision, which also holds the golden); without them they skip. They run non-real-time (set_non_realtime), so every value is the model's and none a
// miss.

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/Delay.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/silero_vad/Vad.h>
#include <anira/models/silero_vad/VadSegmenter.h>
#include <gtest/gtest.h>
#include <tanh/dsp-basics/Resampler.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <span>
#include <vector>

#include "../../../core/test/AudioModelLike.h"
#include "VadSignal.h"

namespace core = anira::models::core;
namespace silero_vad = anira::models::silero_vad;

static_assert(models_test::AudioModelLike<silero_vad::Vad>);

namespace {

constexpr std::array<std::size_t, 7> k_irregular{1, 37, 512, 160, 1000, 7, 255};

std::optional<core::ModelSource> silero_weights() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    const std::filesystem::path dir = std::filesystem::path(root) / "silero-vad";
    if (!std::filesystem::exists(dir)) { return std::nullopt; }
    return core::ModelSource::directory(dir);
}

std::vector<float> golden() {
    std::ifstream file(std::filesystem::path(std::getenv("ANIRA_MODELS_TEST_WEIGHTS")) /
                       "silero-vad/test/vad_golden.json");
    const auto json = nlohmann::json::parse(file);
    return json.at("probabilities").get<std::vector<float>>();
}

std::vector<float> resample(std::span<const float> in, double from, double to) {
    thl::dsp_basics::Resampler resampler;
    resampler.prepare(from, to, in.size());
    std::vector<float> out(resampler.max_output(in.size()));
    out.resize(resampler.process(in, out));
    return out;
}

struct Streamed {
    std::vector<core::Frame> m_frames;
    std::vector<silero_vad::SpeechSpan> m_spans;
    std::vector<float> m_aligned;  // one per host sample
};

// Streams `audio` in blocks of the given sizes (in turn), then silence until `frames` frames are
// out.
Streamed stream(silero_vad::Vad& vad,
                std::span<const float> audio,
                std::span<const std::size_t> blocks,
                std::size_t frames) {
    Streamed out;
    std::vector<float> probability(vad.get_latency() + 4096);
    const auto take = [&] {
        core::Frame frame;
        while (vad.pop(frame)) { out.m_frames.push_back(frame); }
        silero_vad::SpeechSpan span;
        while (vad.pop(span)) { out.m_spans.push_back(span); }
    };
    std::size_t start = 0;
    for (std::size_t k = 0; start < audio.size(); ++k) {
        const auto count = std::min(blocks[k % blocks.size()], audio.size() - start);
        EXPECT_EQ(vad.process(audio.subspan(start, count), std::span(probability).first(count)),
                  ANIRA_OK);
        out.m_aligned.insert(out.m_aligned.end(),
                             probability.begin(),
                             probability.begin() + static_cast<std::ptrdiff_t>(count));
        start += count;
        take();
    }
    const std::vector<float> silence(512, 0.0f);
    for (int guard = 0; out.m_frames.size() < frames && guard < 10000; ++guard) {
        EXPECT_EQ(vad.process(silence, std::span(probability).first(512)), ANIRA_OK);
        out.m_aligned.insert(out.m_aligned.end(), probability.begin(), probability.begin() + 512);
        take();
    }
    return out;
}

// A Vad on ONNX Runtime, prepared, non-real-time.
void prepare(silero_vad::Vad& vad, double rate, std::size_t max_block) {
    const anira::PlanReport report = vad.prepare(rate, max_block);
    const auto plan = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}});
    if (!plan) {
        FAIL() << "no ONNX Runtime plan";
        return;
    }
    vad.set_plan(*plan);
    vad.set_non_realtime(true);
}

}  // namespace

TEST(Clock, ResamplerIsZeroPhaseOnTheModelGrid) {
    // Model sample k is centred on host sample 3k at 48 kHz: an impulse at host 4500 peaks at 1500.
    std::vector<float> host(9000, 0.0f);
    host[4500] = 1.0f;
    const std::vector<float> model = resample(host, 48000.0, 16000.0);
    ASSERT_GT(model.size(), 1500U);
    const auto peak = std::ranges::max_element(model) - model.begin();
    EXPECT_EQ(peak, 1500);
}

TEST(Delay, DelaysEveryChannelByItsLength) {
    core::Delay delay;
    delay.prepare(2, 37);
    std::array<std::vector<float>, 2> audio{std::vector<float>(200, 0.0f),
                                            std::vector<float>(200, 0.0f)};
    audio[0][5] = 1.0f;
    audio[1][9] = -1.0f;
    for (std::size_t start = 0; start < 200;) {
        const std::size_t count = std::min<std::size_t>(start % 2 == 0 ? 13 : 50, 200 - start);
        const std::array<float*, 2> channels{audio[0].data() + start, audio[1].data() + start};
        delay.process(channels, count);
        start += count;
    }
    EXPECT_EQ(audio[0][42], 1.0f);
    EXPECT_EQ(audio[1][46], -1.0f);
    EXPECT_EQ(std::count(audio[0].begin(), audio[0].end(), 0.0f), 199);
}

TEST(Vad, MatchesOnnxRuntimeAt16kInIrregularBlocks) {
    const auto weights = silero_weights();
    if (!weights) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is not set"; }
    anira::Context context{anira::ContextConfig{}};
    silero_vad::Vad vad(context, *weights);
    prepare(vad, 16000.0, 1024);

    const std::vector<float> reference = golden();
    const std::vector<float> audio = models_test::vad_signal();
    const Streamed out = stream(vad, audio, k_irregular, reference.size());
    ASSERT_GE(out.m_frames.size(), reference.size());
    float worst = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        EXPECT_EQ(out.m_frames[i].m_position, static_cast<std::int64_t>(512 * i));
        worst = std::max(worst, std::abs(out.m_frames[i].m_value - reference[i]));
    }
    EXPECT_LT(worst, 1e-4f);
    EXPECT_EQ(vad.dropped(), 0U);
}

TEST(Vad, SameValuesAndPositionsAt48kAnd16k) {
    const auto weights = silero_weights();
    if (!weights) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is not set"; }
    anira::Context context{anira::ContextConfig{}};
    // The 48 kHz audio, and the 16 kHz audio the wrapper makes of it (block size does not matter).
    const std::vector<float> host = resample(models_test::vad_signal(), 16000.0, 48000.0);
    const std::vector<float> model = resample(host, 48000.0, 16000.0);

    silero_vad::Vad at_48k(context, *weights);
    prepare(at_48k, 48000.0, 3000);
    silero_vad::Vad at_16k(context, *weights);
    prepare(at_16k, 16000.0, 1024);
    const std::size_t frames = model.size() / 512;
    const Streamed high = stream(at_48k, host, k_irregular, frames);
    const Streamed low = stream(at_16k, model, k_irregular, frames);

    ASSERT_GE(high.m_frames.size(), frames);
    ASSERT_GE(low.m_frames.size(), frames);
    for (std::size_t i = 0; i < frames; ++i) {
        EXPECT_EQ(high.m_frames[i].m_position, 3 * low.m_frames[i].m_position);
        EXPECT_NEAR(high.m_frames[i].m_value, low.m_frames[i].m_value, 1e-6f);
    }
}

TEST(Vad, AlignedProbabilityTrailsTheInputByTheLatency) {
    const auto weights = silero_weights();
    if (!weights) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is not set"; }
    anira::Context context{anira::ContextConfig{}};
    for (const double rate : {48000.0, 16000.0, 44100.0}) {
        silero_vad::Vad vad(context, *weights);
        prepare(vad, rate, 1024);
        const std::vector<float> host = resample(models_test::vad_signal(), 16000.0, rate);
        const Streamed out = stream(vad, host, k_irregular, 120);

        // aligned[h] is the probability of the frame host sample h - latency belongs to.
        const std::size_t latency = vad.get_latency();
        const double frame_samples = 512.0 * rate / 16000.0;
        std::size_t compared = 0;
        std::size_t wrong = 0;
        for (std::size_t h = latency; h < out.m_aligned.size(); ++h) {
            const auto frame = static_cast<std::size_t>(
                std::floor(static_cast<double>(h - latency) / frame_samples));
            if (frame >= out.m_frames.size()) { break; }
            ++compared;
            if (out.m_aligned[h] != out.m_frames[frame].m_value) { ++wrong; }
        }
        EXPECT_GT(compared, host.size() / 2) << rate;
        EXPECT_EQ(wrong, 0U) << rate << " Hz, latency " << latency;
    }
}

TEST(Vad, RunGivesTheSegmentsOfStreaming) {
    const auto weights = silero_weights();
    if (!weights) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is not set"; }
    anira::Context context{anira::ContextConfig{}};
    const std::vector<float> audio = models_test::vad_signal();
    const std::vector<silero_vad::SpeechSpan> offline = silero_vad::Vad::run(context, *weights, audio, 16000.0);

    silero_vad::Vad vad(context, *weights);
    prepare(vad, 16000.0, 1024);
    const Streamed online = stream(vad, audio, k_irregular, golden().size());
    ASSERT_FALSE(offline.empty());
    EXPECT_EQ(offline, online.m_spans);
}

TEST(Vad, SilenceIsNoSpeech) {
    const auto weights = silero_weights();
    if (!weights) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is not set"; }
    anira::Context context{anira::ContextConfig{}};
    silero_vad::Vad vad(context, *weights);
    prepare(vad, 48000.0, 1024);
    const std::vector<float> silence(std::size_t{3} * 48000, 0.0f);
    const Streamed out = stream(vad, silence, k_irregular, 80);
    EXPECT_TRUE(out.m_spans.empty());
    for (const core::Frame& frame : out.m_frames) { EXPECT_LT(frame.m_value, 0.1f); }
}
