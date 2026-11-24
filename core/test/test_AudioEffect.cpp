// SPDX-License-Identifier: Apache-2.0

// core::AudioEffect on the gain model (test-models/simple-gain, mono, ONNX Runtime), as anira's
// handler with the model built in: the output is the input times the gain exactly get_latency()
// (anira's) later, in any blocks; every channel carries the downmix's result; run() gives the
// output without the latency.

#include <GainDescriptions.h>
#include <anira/anira.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

namespace core = anira::models::core;

namespace {

constexpr float k_gain = 0.5F;

class GainEffect : public core::AudioEffect {
public:
    explicit GainEffect(anira::Context& context)
        : core::AudioEffect(context,
                            core::load_model(models_test::k_gain_model_json,
                                             core::ModelSource::directory(MODELS_TEST_GAIN_DIR)),
                            anira::ContractHandle::from_json(models_test::k_gain_contract_json)) {
        keep_static_input(1, {1}, {k_gain});
    }
};

std::vector<float> sine(std::size_t n, double frequency, double rate) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(
            0.5 * std::sin(2.0 * std::numbers::pi * frequency * static_cast<double>(i) / rate));
    }
    return out;
}

anira::Tensor mono(std::span<float> samples) {
    return anira::Tensor::from_host(
        samples.data(),
        ANIRA_DTYPE_F32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(samples.size())});
}

}  // namespace

TEST(AudioEffect, TheOutputIsTheInputThroughTheModelAniraLatencyLater) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    effect.prepare(48000.0, 512);
    std::vector<float> audio(8192, 0.0F);
    audio[100] = 1.0F;
    audio[101] = 0.25F;
    const std::array<std::size_t, 3> sizes{512, 37, 300};
    std::size_t k = 0;
    for (std::size_t start = 0; start < audio.size(); ++k) {
        const std::size_t count = std::min(sizes[k % sizes.size()], audio.size() - start);
        const anira::Tensor block = mono(std::span<float>(audio).subspan(start, count));
        ASSERT_EQ(effect.process_wait(block, block, ANIRA_WAIT_FOREVER), ANIRA_OK);
        start += count;
    }
    const std::size_t latency = effect.get_latency();
    ASSERT_LT(latency + 101, audio.size());
    for (std::size_t i = 0; i < audio.size(); ++i) {
        const float expected =
            i == latency + 100 ? k_gain : (i == latency + 101 ? k_gain * 0.25F : 0.0F);
        ASSERT_FLOAT_EQ(audio[i], expected) << "sample " << i << ", latency " << latency;
    }
}

TEST(AudioEffect, EveryChannelCarriesTheDownmixsResult) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    effect.prepare(44100.0, 256);
    GainEffect reference(context);
    reference.prepare(44100.0, 256);
    std::vector<float> left = sine(4096, 300.0, 44100.0);
    std::vector<float> right(left.size(), 0.0F);
    std::vector<float> expected = left;
    for (float& x : expected) { x *= 0.5F; }
    for (std::size_t start = 0; start < left.size(); start += 256) {
        std::array<float*, 2> planes{left.data() + start, right.data() + start};
        const anira::Tensor stereo =
            anira::Tensor::from_host_planar<float>(std::span<float* const>(planes),
                                                   std::array<std::int64_t, 2>{2, 256});
        ASSERT_EQ(effect.process_wait(stereo, stereo, ANIRA_WAIT_FOREVER), ANIRA_OK);
        const anira::Tensor block = mono(std::span<float>(expected).subspan(start, 256));
        ASSERT_EQ(reference.process_wait(block, block, ANIRA_WAIT_FOREVER), ANIRA_OK);
    }
    EXPECT_EQ(left, expected);
    EXPECT_EQ(right, expected);
}

TEST(AudioEffect, RunGivesTheOutputWithoutTheLatency) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    effect.prepare(44100.0, 512);
    const std::vector<float> input = sine(20000, 500.0, 44100.0);
    const std::vector<float> output = effect.run(input);
    ASSERT_EQ(output.size(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        ASSERT_FLOAT_EQ(output[i], k_gain * input[i]) << "sample " << i;
    }
}
