// SPDX-License-Identifier: Apache-2.0

// Nemotron 3.5 ASR Streaming (int4) on Moonshine's test recording (moonshine-ai/moonshine
// test-assets/two_cities_16k.wav, a LibriVox reading of A Tale of Two Cities, public domain), in
// test/ of the weights' HF repository. The tests need
// ANIRA_MODELS_TEST_WEIGHTS/nemotron-3.5-asr-streaming-int4 with onnx/ and test/; without it they skip.

#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/nemotron_asr_streaming/NemotronAsr.h>
#include <tanh/dsp-basics/Resampler.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace core = anira::models::core;
namespace na = anira::models::nemotron_asr_streaming;
namespace fs = std::filesystem;

namespace {

std::optional<fs::path> weights_dir() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / "nemotron-3.5-asr-streaming-int4";
    if (!fs::exists(dir / "test") || !fs::exists(dir / "onnx" / "encoder.onnx")) { return std::nullopt; }
    return dir;
}

std::vector<float> read_wav(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    std::size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        std::uint32_t size = 0;
        std::memcpy(&size, bytes.data() + offset + 4, 4);
        if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
            std::vector<float> samples(size / 2);
            for (std::size_t i = 0; i < samples.size(); ++i) {
                std::int16_t value = 0;
                std::memcpy(&value, bytes.data() + offset + 8 + (i * 2), 2);
                samples[i] = static_cast<float>(value) / 32768.0F;
            }
            return samples;
        }
        offset += 8 + size;
    }
    return {};
}

// The recording's first 10 s ("It was the best of times, it was the worst of times, ...").
std::vector<float> opening(const fs::path& dir) {
    std::vector<float> audio = read_wav(dir / "test" / "two_cities_16k.wav");
    audio.resize(std::min<std::size_t>(audio.size(), 160000));
    return audio;
}

std::string lower(std::string text) {
    std::ranges::transform(text, text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

}  // namespace

TEST(NemotronAsr, LanguagesAreTheModelsPromptIndices) {
    EXPECT_EQ(na::language_id("en-US"), 0);
    EXPECT_EQ(na::language_id("de"), 9);
    EXPECT_EQ(na::language_id("auto"), 101);
    EXPECT_FALSE(na::language_id("xx-XX").has_value());
}

TEST(NemotronAsr, RunTranscribesTheOpeningWithWords) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/nemotron-3.5-asr-streaming-int4 is missing"; }
    const std::vector<float> audio = opening(*dir);
    anira::Context context{anira::ContextConfig{}};
    const std::vector<core::Segment> segments = na::NemotronAsr::run(
        context, core::ModelSource::directory(*dir), audio, 16000.0, na::AsrConfig{.m_language = "en-US"});
    ASSERT_EQ(segments.size(), 1U);
    const core::Segment& segment = segments.front();
    EXPECT_NE(lower(segment.m_text).find("best of times"), std::string::npos) << segment.m_text;
    EXPECT_NE(lower(segment.m_text).find("worst of times"), std::string::npos) << segment.m_text;
    EXPECT_EQ(segment.m_text.find('<'), std::string::npos) << segment.m_text;
    EXPECT_EQ(segment.m_start_sample, 0);
    EXPECT_EQ(segment.m_end_sample, static_cast<std::int64_t>(audio.size()));
    ASSERT_GE(segment.m_words.size(), 10U);
    for (std::size_t i = 1; i < segment.m_words.size(); ++i) {
        EXPECT_LE(segment.m_words[i - 1].m_end_sample, segment.m_words[i].m_start_sample);
    }
    EXPECT_LT(segment.m_words.back().m_end_sample, segment.m_end_sample + 1);
}

TEST(NemotronAsr, StreamsRevisionsAt48kInAutoMode) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/nemotron-3.5-asr-streaming-int4 is missing"; }
    const std::vector<float> at_16k = opening(*dir);
    thl::dsp_basics::Resampler up;
    up.prepare(16000.0, 48000.0, at_16k.size());
    std::vector<float> audio(up.max_output(at_16k.size()));
    audio.resize(up.process(at_16k, audio));

    anira::Context context{anira::ContextConfig{}};
    na::NemotronAsr asr(context, core::ModelSource::directory(*dir));
    constexpr std::size_t k_host_block = 480;
    asr.prepare(48000.0, k_host_block);
    for (std::size_t start = 0; start < audio.size(); start += k_host_block) {
        const auto count = std::min(k_host_block, audio.size() - start);
        ASSERT_EQ(asr.process(std::span<const float>(audio).subspan(start, count)), ANIRA_OK);
    }
    asr.end_segment();
    while (!asr.idle()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }

    std::vector<core::Segment> revisions;
    core::Segment segment;
    while (asr.pop(segment)) { revisions.push_back(segment); }
    ASSERT_GE(revisions.size(), 2U);
    const core::Segment& last = revisions.back();
    EXPECT_TRUE(last.m_is_final);
    EXPECT_EQ(last.m_id, 0U);
    EXPECT_NEAR(static_cast<double>(last.m_end_sample), static_cast<double>(audio.size()), 48000.0 * 0.1);
    EXPECT_NE(lower(last.m_text).find("worst of times"), std::string::npos) << last.m_text;
    EXPECT_EQ(last.m_text.find('<'), std::string::npos) << last.m_text;
    for (std::size_t i = 0; i + 1 < revisions.size(); ++i) {
        EXPECT_FALSE(revisions[i].m_is_final);
        EXPECT_EQ(revisions[i].m_id, 0U);
    }
    EXPECT_EQ(asr.latest().m_text, last.m_text);
    EXPECT_EQ(asr.dropped_audio(), 0U);
    EXPECT_EQ(asr.failed(), 0U);
}

TEST(NemotronAsr, LongAudioIsCutAtTheLimit) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/nemotron-3.5-asr-streaming-int4 is missing"; }
    const std::vector<float> audio = read_wav(*dir / "test" / "two_cities_16k.wav");  // 44 s
    anira::Context context{anira::ContextConfig{}};
    const auto start = std::chrono::steady_clock::now();
    const std::vector<core::Segment> segments = na::NemotronAsr::run(
        context, core::ModelSource::directory(*dir), audio, 16000.0, na::AsrConfig{.m_max_segment = 10.0});
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ASSERT_GE(segments.size(), 4U);
    std::string text;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        EXPECT_EQ(segments[i].m_id, i);
        if (i > 0) { EXPECT_EQ(segments[i].m_start_sample, segments[i - 1].m_end_sample); }
        text += segments[i].m_text + " ";
    }
    EXPECT_NE(lower(text).find("best of times"), std::string::npos) << text;
    EXPECT_NE(lower(text).find("worst of times"), std::string::npos) << text;
    std::printf("[ info ] 44 s of audio in %.1f s: %s\n", seconds, text.c_str());
}
