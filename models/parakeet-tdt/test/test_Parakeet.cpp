// SPDX-License-Identifier: Apache-2.0

// Parakeet TDT 0.6B v3 (ONNX: the 4-bit encoder, the int8 decoder) on Moonshine's test recording
// (two_cities_16k.wav, a LibriVox reading, public domain), in test/ of the weights' HF repository:
// a short segment in the smallest window, the whole recording in segments, the words' times. The
// tests need ANIRA_MODELS_TEST_WEIGHTS/parakeet-tdt-0.6b-v3 with its ONNX files and test/; without
// it they skip.

#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/parakeet_tdt/Parakeet.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace core = anira::models::core;
namespace pt = anira::models::parakeet_tdt;
namespace fs = std::filesystem;

namespace {

std::optional<fs::path> weights_dir() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / "parakeet-tdt-0.6b-v3";
    if (!fs::exists(dir / "test") || !fs::exists(dir / "encoder-model.q4.onnx")) {
        return std::nullopt;
    }
    return dir;
}

std::vector<float> read_wav(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(file),
                                  std::istreambuf_iterator<char>()};
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

std::string lower(std::string text) {
    std::ranges::transform(text, text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
}

}  // namespace

TEST(Parakeet, ASegmentHasItsTextAndWordsInOrder) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/parakeet-tdt-0.6b-v3 is missing"; }
    std::vector<float> audio = read_wav(*dir / "test" / "two_cities_16k.wav");
    audio.resize(std::size_t{7} * 16000);  // 7 s: the 8 s window
    anira::Context context{anira::ContextConfig{}};
    pt::Parakeet parakeet(context, core::ModelSource::directory(*dir));
    parakeet.prepare();
    const core::Segment segment = parakeet.transcribe(audio, 1000, 7);
    EXPECT_EQ(segment.m_id, 7U);
    EXPECT_TRUE(segment.m_is_final);
    EXPECT_EQ(segment.m_start_sample, 1000);
    EXPECT_EQ(segment.m_end_sample, 1000 + 7 * 16000);
    EXPECT_NE(lower(segment.m_text).find("best of times"), std::string::npos) << segment.m_text;
    ASSERT_GE(segment.m_words.size(), 5U);
    for (std::size_t i = 0; i < segment.m_words.size(); ++i) {
        const core::Word& word = segment.m_words[i];
        EXPECT_GE(word.m_start_sample, segment.m_start_sample);
        EXPECT_LE(word.m_end_sample, segment.m_end_sample);
        EXPECT_LE(word.m_start_sample, word.m_end_sample);
        if (i > 0) { EXPECT_GE(word.m_start_sample, segment.m_words[i - 1].m_start_sample); }
    }
}

TEST(Parakeet, RunTranscribesALongRecording) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/parakeet-tdt-0.6b-v3 is missing"; }
    const std::vector<float> audio = read_wav(*dir / "test" / "two_cities_16k.wav");  // 44 s
    anira::Context context{anira::ContextConfig{}};
    const std::vector<core::Segment> segments =
        pt::Parakeet::run(context, core::ModelSource::directory(*dir), audio, 16000.0);
    ASSERT_EQ(segments.size(), 2U);  // 31 s + 13 s
    EXPECT_EQ(segments[1].m_start_sample, segments[0].m_end_sample);
    const std::string text = lower(segments[0].m_text + " " + segments[1].m_text);
    EXPECT_NE(text.find("best of times"), std::string::npos) << text;
    EXPECT_NE(text.find("worst of times"), std::string::npos) << text;
}
