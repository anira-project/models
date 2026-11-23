// SPDX-License-Identifier: Apache-2.0

// Moonshine streaming tiny-en on Moonshine's own test recording (moonshine-ai/moonshine
// test-assets/two_cities_16k.wav, a LibriVox reading of A Tale of Two Cities, public domain), in
// test/ of the weights' HF repository. The tests
// need ANIRA_MODELS_TEST_WEIGHTS/moonshine-streaming with tiny-en and test/; without it they skip.

#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/moonshine_streaming/Moonshine.h>
#include <anira/models/moonshine_streaming/SegmentDecoder.h>
#include <tanh/dsp-basics/Resampler.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace core = anira::models::core;
namespace ms = anira::models::moonshine_streaming;
namespace fs = std::filesystem;

namespace {

std::optional<fs::path> weights_dir() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    const fs::path dir = fs::path(root) / "moonshine-streaming";
    if (!fs::exists(dir / "test") || !fs::exists(dir / "tiny-en" / "encoder.onnx")) { return std::nullopt; }
    return dir;
}

// A 16-bit PCM mono WAV as floats.
std::vector<float> read_wav(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
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

// The recording's first 10 s: one segment ("It was the best of times, ...").
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

TEST(Moonshine, TokenizerSpellsTheSpecialTokens) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/moonshine-streaming is missing"; }
    const std::string data = core::ModelSource::directory(*dir).read_text("tiny-en/tokenizer.bin");
    const ms::Tokenizer tokenizer(std::as_bytes(std::span<const char>(data.data(), data.size())));
    EXPECT_GT(tokenizer.size(), 30000U);
    const std::vector<std::int32_t> specials{ms::k_bos, ms::k_eos};
    EXPECT_EQ(tokenizer.text(specials), "");
}

TEST(Moonshine, OneSegmentTranscribesTheOpening) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/moonshine-streaming is missing"; }
    anira::Context context{anira::ContextConfig{}};
    ms::SegmentDecoder decoder(context, core::ModelSource::directory(*dir), ms::Variant::TinyEn);
    decoder.prepare();
    std::vector<float> audio = opening(*dir);
    ASSERT_FALSE(audio.empty());
    audio.resize(((audio.size() + ms::k_step - 1) / ms::k_step) * ms::k_step, 0.0F);
    for (std::size_t start = 0; start < audio.size(); start += ms::k_step) {
        ASSERT_TRUE(decoder.push(std::span<const float>(audio).subspan(start, ms::k_step)));
    }
    decoder.flush();
    const std::vector<std::int32_t> tokens =
        decoder.decode(static_cast<double>(audio.size()) / ms::k_sample_rate);
    const std::string text = decoder.tokenizer().text(tokens);
    EXPECT_NE(lower(text).find("best of times"), std::string::npos) << text;
    EXPECT_NE(lower(text).find("worst of times"), std::string::npos) << text;
}

TEST(Moonshine, StreamsRevisionsAndTheFinalTextAt48k) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/moonshine-streaming is missing"; }
    const std::vector<float> at_16k = opening(*dir);
    thl::dsp_basics::Resampler up;
    up.prepare(16000.0, 48000.0, at_16k.size());
    std::vector<float> audio(up.max_output(at_16k.size()));
    audio.resize(up.process(at_16k, audio));

    anira::Context context{anira::ContextConfig{}};
    ms::Moonshine moonshine(context, core::ModelSource::directory(*dir));
    constexpr std::size_t k_host_block = 480;
    moonshine.prepare(48000.0, k_host_block);
    for (std::size_t start = 0; start < audio.size(); start += k_host_block) {
        const auto count = std::min(k_host_block, audio.size() - start);
        ASSERT_EQ(moonshine.process(std::span<const float>(audio).subspan(start, count)), ANIRA_OK);
        std::this_thread::sleep_for(std::chrono::microseconds(1000));  // faster than real time
    }
    moonshine.end_segment();
    while (!moonshine.idle()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }

    std::vector<core::Segment> revisions;
    core::Segment segment;
    while (moonshine.pop(segment)) { revisions.push_back(segment); }
    ASSERT_GE(revisions.size(), 2U);
    const core::Segment& last = revisions.back();
    EXPECT_TRUE(last.m_is_final);
    EXPECT_EQ(last.m_id, 0U);
    EXPECT_EQ(last.m_start_sample, 0);
    EXPECT_NEAR(static_cast<double>(last.m_end_sample), static_cast<double>(audio.size()), 48000.0 * 0.5);
    EXPECT_NE(lower(last.m_text).find("worst of times"), std::string::npos) << last.m_text;
    for (std::size_t i = 0; i + 1 < revisions.size(); ++i) {
        EXPECT_FALSE(revisions[i].m_is_final);
        EXPECT_EQ(revisions[i].m_id, 0U);
    }
    EXPECT_EQ(moonshine.latest().m_text, last.m_text);
    EXPECT_EQ(moonshine.dropped_audio(), 0U);
    EXPECT_EQ(moonshine.failed(), 0U);
}

TEST(Moonshine, RunTranscribesALongRecordingInSegments) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/moonshine-streaming is missing"; }
    const std::vector<float> audio = read_wav(*dir / "test" / "two_cities_16k.wav");  // 44 s
    anira::Context context{anira::ContextConfig{}};
    const std::vector<core::Segment> segments =
        ms::Moonshine::run(context, core::ModelSource::directory(*dir), audio, 16000.0);
    ASSERT_EQ(segments.size(), 3U);  // cut at 15.5 s
    std::string text;
    for (std::size_t i = 0; i < segments.size(); ++i) {
        EXPECT_EQ(segments[i].m_id, i);
        if (i > 0) { EXPECT_EQ(segments[i].m_start_sample, segments[i - 1].m_end_sample); }
        text += segments[i].m_text + " ";
    }
    EXPECT_NE(lower(text).find("best of times"), std::string::npos) << text;
    EXPECT_NE(lower(text).find("worst of times"), std::string::npos) << text;
}
