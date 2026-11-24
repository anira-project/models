// SPDX-License-Identifier: Apache-2.0

// Nemotron 3 Diarization (ONNX, q4) on a recording of two speakers (three LibriSpeech dev-clean
// utterances, CC BY 4.0: a woman, a man, the woman again, 0.5 s apart; made by
// export/make_test_audio.py, in test/ of the weights' HF repository, revision 23e7f4f): two
// speakers, in the right order, each turn where that voice speaks. The tests need
// ANIRA_MODELS_TEST_WEIGHTS/nemotron-3-diarization with onnx/ and test/; without it they skip.

#include <anira/anira.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/Segment.h>
#include <anira/models/nemotron_diarization/Diarizer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
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
#include <thread>
#include <vector>

namespace core = anira::models::core;
namespace nd = anira::models::nemotron_diarization;
namespace fs = std::filesystem;

namespace {

std::optional<fs::path> weights_dir() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / "nemotron-3-diarization";
    if (!fs::exists(dir / "test") || !fs::exists(dir / "onnx" / "model_q4.onnx")) {
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

// The voices as the recording was made: speaker, first and end sample (16 kHz).
struct Voice {
    int m_speaker;
    std::int64_t m_start;
    std::int64_t m_end;
};
constexpr std::array<Voice, 3> k_voices{{{.m_speaker = 0, .m_start = 0, .m_end = 108800},
                                         {.m_speaker = 1, .m_start = 116800, .m_end = 275200},
                                         {.m_speaker = 0, .m_start = 283200, .m_end = 395440}}};
constexpr std::int64_t k_tolerance = 16000;  // 1 s at a turn's edges

}  // namespace

TEST(Diarizer, FindsTwoSpeakersAndTheirTurns) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/nemotron-3-diarization is missing"; }
    const std::vector<float> audio = read_wav(*dir / "test" / "two_speakers_16k.wav");
    anira::Context context{anira::ContextConfig{}};
    // A pause within an utterance may split its turn: one speaker's turns less than 1 s apart are
    // one.
    std::vector<core::Segment> turns;
    for (const core::Segment& turn :
         nd::Diarizer::run(context, core::ModelSource::directory(*dir), audio, 16000.0)) {
        if (!turns.empty() && turns.back().m_speaker == turn.m_speaker &&
            turn.m_start_sample - turns.back().m_end_sample < k_tolerance) {
            turns.back().m_end_sample = turn.m_end_sample;
        } else {
            turns.push_back(turn);
        }
    }
    ASSERT_EQ(turns.size(), k_voices.size());
    for (std::size_t i = 0; i < k_voices.size(); ++i) {
        EXPECT_EQ(turns[i].m_speaker, k_voices[i].m_speaker) << "turn " << i;
        EXPECT_NEAR(static_cast<double>(turns[i].m_start_sample),
                    static_cast<double>(k_voices[i].m_start),
                    k_tolerance)
            << "turn " << i;
        EXPECT_NEAR(static_cast<double>(turns[i].m_end_sample),
                    static_cast<double>(k_voices[i].m_end),
                    k_tolerance)
            << "turn " << i;
    }
}

TEST(Diarizer, StreamsAFrameEvery10ms) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/nemotron-3-diarization is missing"; }
    const std::vector<float> audio = read_wav(*dir / "test" / "two_speakers_16k.wav");
    anira::Context context{anira::ContextConfig{}};
    nd::Diarizer diarizer(context, core::ModelSource::directory(*dir));
    constexpr std::size_t k_block = 160;
    diarizer.prepare(16000.0, k_block);
    for (std::size_t start = 0; start < audio.size(); start += k_block) {
        ASSERT_EQ(diarizer.process(std::span<const float>(audio).subspan(
                      start,
                      std::min(k_block, audio.size() - start))),
                  ANIRA_OK);
    }
    diarizer.flush();
    while (!diarizer.idle()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    nd::SpeakerFrame frame;
    std::int64_t count = 0;
    while (diarizer.pop(frame)) {
        EXPECT_EQ(frame.m_position, count * 160);
        ++count;
    }
    EXPECT_NEAR(static_cast<double>(count), static_cast<double>(audio.size()) / 160.0, 2.0);
    EXPECT_EQ(diarizer.dropped(), 0U);
    EXPECT_EQ(diarizer.dropped_audio(), 0U);
}
