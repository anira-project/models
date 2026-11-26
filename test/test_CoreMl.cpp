// SPDX-License-Identifier: Apache-2.0

// The Core ML exports through core::coreml_engine() (temporary, until anira has a Core ML engine):
// every effect wrapper runs the same plucked-string signal on ONNX Runtime and on Core ML (every
// compute unit Core ML picks: the descriptions pin the provider "coreml"), and the outputs agree.
// The stream runs through the wrapper as a plugin runs it (resampled, block by block, its state
// carried), so a Core ML export that loses state or reads its inputs wrongly diverges. Built with
// ANIRA_MODELS_COREML_ENGINE on macOS; each needs ANIRA_MODELS_TEST_WEIGHTS/<its HF repository>
// with its ONNX, LiteRT and Core ML files.

#include <anira/anira.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/CoreMlEngine.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
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
// Relative to the reference's peak, as tools/check_streaming.py's TOLERANCE_COREML: Core ML may
// run in float16 on the GPU and the Neural Engine.
constexpr double k_tolerance = 2e-2;

std::optional<fs::path> weights_dir(const std::string& repo, const std::string& package) {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / repo;
    if (!fs::exists(dir / package)) { return std::nullopt; }
    return dir;
}

// Decaying plucks at 110, 165 and 220 Hz, one every 0.25 s, 1.5 s (test_Effects.cpp's).
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

// The plan's output for `input`, and the time of a second run (the first one warms up), in ms.
std::vector<float> run_timed(core::AudioEffect& effect,
                             const std::vector<float>& input,
                             double& ms) {
    effect.run(input);
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> out = effect.run(input);
    ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void compare(core::AudioEffect& effect, const char* name) {
    const anira::PlanReport report = effect.prepare(k_rate, 512);
    const auto reference = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}});
    ASSERT_TRUE(reference) << name << ": no ONNX Runtime plan";
    const std::vector<float> input = plucks();
    effect.set_plan(*reference);
    double onnx_ms = 0.0;
    const std::vector<float> expected = run_timed(effect, input, onnx_ms);
    float peak = 0.0F;
    for (const float x : expected) { peak = std::max(peak, std::abs(x)); }

    {
        const auto plan = report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
                                                   .provider = ANIRA_PROVIDER_COREML,
                                                   .engine_id = core::k_coreml_engine_id}});
        ASSERT_TRUE(plan) << name << ": no Core ML plan";
        effect.set_plan(*plan);
        double coreml_ms = 0.0;
        const std::vector<float> got = run_timed(effect, input, coreml_ms);
        ASSERT_EQ(got.size(), expected.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            ASSERT_TRUE(std::isfinite(got[i])) << name << " at sample " << i;
            worst = std::max(worst, static_cast<double>(std::abs(got[i] - expected[i])));
        }
        worst /= 1.0 + static_cast<double>(peak);
        std::printf("[ info ] %s on Core ML: max |Core ML - ONNX Runtime| %.2e of peak %.3f\n",
                    name,
                    worst,
                    static_cast<double>(peak));
        std::printf("[ info ] %s: %.1f s of audio in %.0f ms on ONNX Runtime, %.0f ms on Core ML\n",
                    name,
                    static_cast<double>(input.size()) / k_rate,
                    onnx_ms,
                    coreml_ms);
        EXPECT_LT(worst, k_tolerance) << name;
    }
}

}  // namespace

#ifdef ANIRA_MODELS_HAS_GUITAR_LSTM
TEST(CoreMl, GuitarLstm) {
    const auto dir = weights_dir("guitar-lstm", "coreml/GuitarLSTM.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/guitar-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::guitar_lstm::GuitarLstm effect(context, core::ModelSource::directory(*dir));
    compare(effect, "GuitarLSTM");
}
#endif

#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
TEST(CoreMl, StatefulLstm) {
    const auto dir = weights_dir("stateful-lstm", "coreml/stateful_lstm.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/stateful-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::stateful_lstm::StatefulLstm effect(context, core::ModelSource::directory(*dir));
    compare(effect, "Stateful LSTM");
}
#endif

#ifdef ANIRA_MODELS_HAS_STEERABLE_NAFX
TEST(CoreMl, SteerableNafxInEverySize) {
    const auto dir = weights_dir("steerable-nafx", "coreml/steerable-nafx.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/steerable-nafx is missing"; }
    using anira::models::steerable_nafx::SteerableNafx;
    for (const auto size : {SteerableNafx::Size::Full,
                            SteerableNafx::Size::ThreeBlocks,
                            SteerableNafx::Size::TwoBlocks}) {
        anira::Context context{anira::ContextConfig{}};
        SteerableNafx effect(context, core::ModelSource::directory(*dir), size);
        compare(effect, "Steerable NAFX");
    }
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_DJEMBE
TEST(CoreMl, RaveDjembe) {
    const auto dir = weights_dir("rave-djembe", "coreml/rave_forward.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-djembe is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_djembe::RaveDjembe effect(context, core::ModelSource::directory(*dir));
    compare(effect, "RAVE Djembe");
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_FUNK_DRUM
TEST(CoreMl, RaveFunkDrum) {
    const auto dir = weights_dir("rave-funk-drum", "coreml/rave_funk_drum_forward.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-funk-drum is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_funk_drum::RaveFunkDrum effect(context, core::ModelSource::directory(*dir));
    compare(effect, "RAVE Funk Drum");
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_MAGNETIC_RESONATOR_PIANO
TEST(CoreMl, RaveMagneticResonatorPiano) {
    const auto dir =
        weights_dir("rave-magnetic-resonator-piano", "coreml/mrp_strengjavera_forward.mlpackage");
    if (!dir) {
        GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-magnetic-resonator-piano is missing";
    }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_magnetic_resonator_piano::RaveMagneticResonatorPiano effect(
        context,
        core::ModelSource::directory(*dir));
    compare(effect, "RAVE Magnetic Resonator Piano");
}
#endif
