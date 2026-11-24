// SPDX-License-Identifier: Apache-2.0

// The effect wrappers (core::AudioEffect, header-only per model) on their weights: each prepares
// at 48 kHz (resampled to the 44.1 kHz it was trained at), runs a plucked-string test signal and
// gives a finite, audible result after its latency; a reset() and the same signal give the same
// result again (a stateful model starts from zero state). Each needs
// ANIRA_MODELS_TEST_WEIGHTS/<its HF repository> with its ONNX files; without them it skips.

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#ifdef ANIRA_MODELS_HAS_GUITAR_LSTM
#include <anira/models/guitar_lstm/GuitarLstm.h>
#endif
#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
#include <anira/models/stateful_lstm/StatefulLstm.h>
#endif
#ifdef ANIRA_MODELS_HAS_STEERABLE_NAFX
#include <anira/models/steerable_nafx/SteerableNafx.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_DJEMBE
#include <anira/models/rave_djembe/RaveDjembe.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_FUNK_DRUM
#include <anira/models/rave_funk_drum/RaveFunkDrum.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_MAGNETIC_RESONATOR_PIANO
#include <anira/models/rave_magnetic_resonator_piano/RaveMagneticResonatorPiano.h>
#endif

namespace core = anira::models::core;
namespace fs = std::filesystem;

namespace {

constexpr double k_rate = 48000.0;

std::optional<fs::path> weights_dir(const std::string& repo, const std::string& onnx) {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / repo;
    if (!fs::exists(dir / onnx)) { return std::nullopt; }
    return dir;
}

// Decaying plucks at 110, 165 and 220 Hz, one every 0.25 s, 1.5 s.
std::vector<float> plucks() {
    std::vector<float> out(static_cast<std::size_t>(1.5 * k_rate), 0.0F);
    const std::array<double, 3> notes{110.0, 165.0, 220.0};
    for (std::size_t k = 0; k < 6; ++k) {
        const auto start = static_cast<std::size_t>(0.25 * k_rate * static_cast<double>(k));
        for (std::size_t i = start; i < out.size(); ++i) {
            const double t = static_cast<double>(i - start) / k_rate;
            out[i] += static_cast<float>(0.3 * std::exp(-4.0 * t) *
                                         std::sin(2.0 * std::numbers::pi * notes[k % 3] * t));
        }
    }
    return out;
}

void check(core::AudioEffect& effect) {
    const std::vector<float> input = plucks();
    const std::vector<float> first = effect.run(input);
    ASSERT_EQ(first.size(), input.size());
    double energy = 0.0;
    for (const float x : first) {
        ASSERT_TRUE(std::isfinite(x));
        energy += static_cast<double>(x) * x;
    }
    EXPECT_GT(energy / static_cast<double>(first.size()), 1e-8) << "silent";
    EXPECT_GT(effect.get_latency(), 0U);
    const std::vector<float> again = effect.run(input);  // run() resets: the same stream again
    float worst = 0.0F;
    for (std::size_t i = 0; i < first.size(); ++i) {
        worst = std::max(worst, std::abs(first[i] - again[i]));
    }
    EXPECT_LT(worst, 1e-4F);
    std::printf("[ info ] latency %zu samples at 48 kHz, rms %.4f\n",
                effect.get_latency(),
                std::sqrt(energy / static_cast<double>(first.size())));
}

}  // namespace

#ifdef ANIRA_MODELS_HAS_GUITAR_LSTM
TEST(Effects, GuitarLstm) {
    const auto dir = weights_dir("guitar-lstm", "onnx/GuitarLSTM-libtorch-dynamic.onnx");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/guitar-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::guitar_lstm::GuitarLstm effect(context, core::ModelSource::directory(*dir));
    effect.prepare(k_rate, 512);
    check(effect);
}
#endif

#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
TEST(Effects, StatefulLstm) {
    const auto dir = weights_dir("stateful-lstm", "onnx/stateful_lstm.onnx");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/stateful-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::stateful_lstm::StatefulLstm effect(context, core::ModelSource::directory(*dir));
    effect.prepare(k_rate, 512);
    check(effect);
}
#endif

#ifdef ANIRA_MODELS_HAS_STEERABLE_NAFX
TEST(Effects, SteerableNafxInEverySize) {
    const auto dir = weights_dir("steerable-nafx", "onnx/steerable-nafx-libtorch-dynamic.onnx");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/steerable-nafx is missing"; }
    using anira::models::steerable_nafx::SteerableNafx;
    for (const auto size : {SteerableNafx::Size::Full,
                            SteerableNafx::Size::ThreeBlocks,
                            SteerableNafx::Size::TwoBlocks}) {
        anira::Context context{anira::ContextConfig{}};
        SteerableNafx effect(context, core::ModelSource::directory(*dir), size);
        effect.prepare(k_rate, 512);
        check(effect);
    }
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_DJEMBE
TEST(Effects, RaveDjembe) {
    const auto dir = weights_dir("rave-djembe", "onnx/rave_forward.onnx");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-djembe is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_djembe::RaveDjembe effect(context, core::ModelSource::directory(*dir));
    effect.prepare(k_rate, 512);
    check(effect);
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_FUNK_DRUM
TEST(Effects, RaveFunkDrum) {
    const auto dir = weights_dir("rave-funk-drum", "onnx/rave_funk_drum_forward.onnx");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-funk-drum is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_funk_drum::RaveFunkDrum effect(context, core::ModelSource::directory(*dir));
    effect.prepare(k_rate, 512);
    check(effect);
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_MAGNETIC_RESONATOR_PIANO
TEST(Effects, RaveMagneticResonatorPiano) {
    const auto dir =
        weights_dir("rave-magnetic-resonator-piano", "onnx/mrp_strengjavera_forward.onnx");
    if (!dir) {
        GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-magnetic-resonator-piano is missing";
    }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_magnetic_resonator_piano::RaveMagneticResonatorPiano effect(
        context,
        core::ModelSource::directory(*dir));
    effect.prepare(k_rate, 512);
    check(effect);
}
#endif
