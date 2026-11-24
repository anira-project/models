// SPDX-License-Identifier: Apache-2.0

// core::AudioEffect on the gain model (test-models/simple-gain, mono, ONNX Runtime) run at a model
// rate of 44.1 kHz: at that rate the output is the input times the gain exactly get_latency() later
// (anira's latency); at 48 kHz, through both resamplers, a band-limited signal comes out the same
// way within the filters' ripple, in any blocks; every channel carries the downmix's result; run()
// gives the output without the latency.

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
                            anira::ContractHandle::from_json(models_test::k_gain_contract_json),
                            44100.0) {
        keep_static_input(1, {1}, {k_gain});
    }
};

void prepare_offline(GainEffect& effect, double rate, std::size_t max_block) {
    effect.prepare(rate, max_block);
    effect.set_non_realtime(true);
}

std::vector<float> sine(std::size_t n, double frequency, double rate) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(
            0.5 * std::sin(2.0 * std::numbers::pi * frequency * static_cast<double>(i) / rate));
    }
    return out;
}

// `audio` in blocks of the sizes in turn.
void process(GainEffect& effect, std::vector<float>& audio, std::span<const std::size_t> sizes) {
    std::size_t k = 0;
    for (std::size_t start = 0; start < audio.size(); ++k) {
        const std::size_t count = std::min(sizes[k % sizes.size()], audio.size() - start);
        ASSERT_EQ(effect.process(std::span<float>(audio).subspan(start, count)), ANIRA_OK);
        start += count;
    }
}

}  // namespace

TEST(AudioEffect, AtTheModelsRateTheLatencyIsAniras) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    prepare_offline(effect, 44100.0, 512);
    std::vector<float> audio(8192, 0.0F);
    audio[100] = 1.0F;
    audio[101] = 0.25F;
    const std::array<std::size_t, 3> sizes{512, 37, 300};
    process(effect, audio, sizes);
    const std::size_t latency = effect.get_latency();
    ASSERT_LT(latency + 101, audio.size());
    for (std::size_t i = 0; i < audio.size(); ++i) {
        const float expected =
            i == latency + 100 ? k_gain : (i == latency + 101 ? k_gain * 0.25F : 0.0F);
        ASSERT_FLOAT_EQ(audio[i], expected) << "sample " << i << ", latency " << latency;
    }
}

TEST(AudioEffect, At48kTheOutputIsTheInputThroughTheModelLatencyLater) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    prepare_offline(effect, 48000.0, 480);
    const std::vector<float> input = sine(48000, 440.0, 48000.0);
    std::vector<float> audio = input;
    const std::array<std::size_t, 4> sizes{480, 1, 333, 64};
    process(effect, audio, sizes);
    const std::size_t latency = effect.get_latency();
    float worst = 0.0F;
    for (std::size_t i = latency + 4800; i < audio.size(); ++i) {
        worst = std::max(worst, std::abs(audio[i] - (k_gain * input[i - latency])));
    }
    EXPECT_LT(worst, 2e-3F) << "latency " << latency;
}

TEST(AudioEffect, EveryChannelCarriesTheDownmixsResult) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    prepare_offline(effect, 44100.0, 256);
    std::vector<float> left = sine(4096, 300.0, 44100.0);
    std::vector<float> right(left.size(), 0.0F);
    std::vector<float> mono = left;
    for (float& x : mono) { x *= 0.5F; }
    GainEffect reference(context);
    prepare_offline(reference, 44100.0, 256);
    for (std::size_t start = 0; start < left.size(); start += 256) {
        std::array<float*, 2> planes{left.data() + start, right.data() + start};
        const anira::Tensor stereo =
            anira::Tensor::from_host_planar<float>(std::span<float* const>(planes),
                                                   std::array<std::int64_t, 2>{2, 256});
        ASSERT_EQ(effect.process(stereo), ANIRA_OK);
        ASSERT_EQ(reference.process(std::span<float>(mono).subspan(start, 256)), ANIRA_OK);
    }
    EXPECT_EQ(left, mono);
    EXPECT_EQ(right, mono);
}

TEST(AudioEffect, RunGivesTheOutputWithoutTheLatency) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    prepare_offline(effect, 44100.0, 512);
    const std::vector<float> input = sine(20000, 500.0, 44100.0);
    const std::vector<float> output = effect.run(input);
    ASSERT_EQ(output.size(), input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        ASSERT_FLOAT_EQ(output[i], k_gain * input[i]) << "sample " << i;
    }
}
