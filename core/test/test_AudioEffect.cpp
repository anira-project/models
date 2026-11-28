// SPDX-License-Identifier: Apache-2.0

// core::AudioEffect on the gain model (test-models/simple-gain, mono, ONNX Runtime) run at a model
// rate of 44.1 kHz, as anira's handler with the model built in: at that rate the output is the
// input times the gain exactly get_latency() (anira's) later, in any blocks; at 48 kHz, through
// both resamplers, a band-limited signal comes out the same way within the filters' ripple; every
// channel carries the downmix's result; run() gives the output without the latency. At every
// common host rate, and in real time on a clock, the output has no artefacts.

#include <GainDescriptions.h>
#include <anira/anira.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <span>
#include <thread>
#include <vector>

namespace core = anira::models::core;

namespace {

constexpr float k_gain = 0.5F;
constexpr double k_model_rate = 44100.0;  // the gain model runs at 44.1 kHz here

class GainEffect : public core::AudioEffect {
public:
    explicit GainEffect(anira::Context& context)
        : core::AudioEffect(context,
                            core::load_model(models_test::k_gain_model_json,
                                             core::ModelSource::directory(MODELS_TEST_GAIN_DIR)),
                            anira::ContractHandle::from_json(models_test::k_gain_contract_json),
                            k_model_rate) {
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

TEST(AudioEffect, AtTheModelsRateTheOutputIsTheInputThroughTheModelAniraLatencyLater) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    effect.prepare(44100.0, 512);  // the model's rate: the resamplers are pass-through
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

TEST(AudioEffect, At48kTheOutputIsTheInputThroughTheModelLatencyLater) {
    anira::Context context{anira::ContextConfig{}};
    GainEffect effect(context);
    effect.prepare(48000.0, 480);
    const std::vector<float> input = sine(48000, 440.0, 48000.0);
    std::vector<float> audio = input;
    const std::array<std::size_t, 4> sizes{480, 1, 333, 64};
    std::size_t k = 0;
    for (std::size_t start = 0; start < audio.size(); ++k) {
        const std::size_t count = std::min(sizes[k % sizes.size()], audio.size() - start);
        const anira::Tensor block = mono(std::span<float>(audio).subspan(start, count));
        ASSERT_EQ(effect.process_wait(block, block, ANIRA_WAIT_FOREVER), ANIRA_OK);
        start += count;
    }
    const std::size_t latency = effect.get_latency();
    float worst = 0.0F;
    for (std::size_t i = latency + 4800; i < audio.size(); ++i) {
        worst = std::max(worst, std::abs(audio[i] - (k_gain * input[i - latency])));
    }
    EXPECT_LT(worst, 2e-3F) << "latency " << latency;
}

namespace {

/// The input, two sines in the passband of every rate below (at 16 kHz it ends near 7 kHz), at
/// any (fractional) sample position.
double two_sines_at(double position, double rate) {
    const double t = 2.0 * std::numbers::pi * position / rate;
    return 0.25 * (std::sin(440.0 * t) + std::sin(3000.0 * t));
}

struct Fit {
    double m_fraction = 0.0;  ///< the delay beyond get_latency(), in samples
    float m_worst = 0.0F;     ///< the largest difference left
};

/// The output against the input through the model (the gain) get_latency() + a fraction later,
/// after `settle` samples (the resamplers' filters filling from silence): the fraction that fits
/// best (a resampled path delays by a fraction of a host sample that an integer latency cannot
/// say) and the largest difference left, which a click, a dropped or repeated sample or a step at a
/// block edge would raise far above the filters' ripple.
Fit fit(const std::vector<float>& output, std::size_t latency, std::size_t settle, double rate) {
    Fit best{.m_fraction = 0.0, .m_worst = 1e9F};
    for (int step = -100; step <= 100; ++step) {
        const double fraction = step / 100.0;
        float worst = 0.0F;
        for (std::size_t i = latency + settle; i < output.size(); ++i) {
            const double expected =
                k_gain * two_sines_at(static_cast<double>(i - latency) - fraction, rate);
            worst = std::max(worst, std::abs(output[i] - static_cast<float>(expected)));
        }
        if (worst < best.m_worst) { best = {.m_fraction = fraction, .m_worst = worst}; }
    }
    return best;
}

std::vector<float> two_sines(std::size_t n, double rate) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(two_sines_at(static_cast<double>(i), rate));
    }
    return out;
}

}  // namespace

// The resamplers and the model in between add nothing: at every common host rate, in irregular
// blocks, the output is the input through the model, get_latency() later, within the filters'
// ripple (a click, a dropped or repeated sample, a step at a block edge would be far above it).
TEST(AudioEffect, AtEveryHostRateTheOutputHasNoArtefacts) {
    anira::Context context{anira::ContextConfig{}};
    const std::array<double, 8>
        rates{16000.0, 22050.0, 24000.0, 32000.0, 44100.0, 48000.0, 88200.0, 96000.0};
    const std::array<std::size_t, 5> sizes{512, 1, 333, 64, 129};
    for (const double rate : rates) {
        GainEffect effect(context);
        effect.prepare(rate, 512);
        const auto n = static_cast<std::size_t>(rate);  // a second
        const std::vector<float> input = two_sines(n, rate);
        std::vector<float> audio = input;
        std::size_t k = 0;
        for (std::size_t start = 0; start < audio.size(); ++k) {
            const std::size_t count = std::min(sizes[k % sizes.size()], audio.size() - start);
            const anira::Tensor block = mono(std::span<float>(audio).subspan(start, count));
            ASSERT_EQ(effect.process_wait(block, block, ANIRA_WAIT_FOREVER), ANIRA_OK) << rate;
            start += count;
        }
        const Fit f = fit(audio, effect.get_latency(), static_cast<std::size_t>(rate / 10.0), rate);
        EXPECT_LT(f.m_worst, 2e-3F)
            << rate << " Hz, latency " << effect.get_latency() << " + " << f.m_fraction;
        EXPECT_LT(std::abs(f.m_fraction), 1.0) << rate << " Hz";
        std::printf("[ info ] %.0f Hz: latency %zu + %.2f samples, worst %.2g\n",
                    rate,
                    effect.get_latency(),
                    f.m_fraction,
                    static_cast<double>(f.m_worst));
    }
}

namespace {

// Waits until `deadline` by spinning (yielding), not sleeping: a sleep wakes up to a timer tick
// late (15.6 ms on Windows by default, longer than a block at 24 kHz; milliseconds in a virtual
// machine), while an audio device calls back on time.
void wait_until(std::chrono::steady_clock::time_point deadline) {
    while (std::chrono::steady_clock::now() < deadline) { std::this_thread::yield(); }
}

// Whether this is macOS in a virtual machine (GitHub's macOS runners), whose vCPUs stall for
// milliseconds under the host's load, which no wait on the guest can make up for.
bool in_a_virtual_mac() {
#ifdef __APPLE__
    int present = 0;
    std::size_t size = sizeof(present);
    return sysctlbyname("kern.hv_vmm_present", &present, &size, nullptr, 0) == 0 && present != 0;
#else
    return false;
#endif
}

}  // namespace

// In real time (process(), which never waits) on a clock, a block per period as an audio device
// calls: no block is missed, and the output has no artefacts (a miss would fill a block with
// what the miss policy gives, not the input through the model).
TEST(AudioEffect, InRealTimeOnAClockNoBlockIsMissedAndTheOutputHasNoArtefacts) {
    if (in_a_virtual_mac()) { GTEST_SKIP() << "a virtual Mac (CI) stalls: no clock to keep"; }
    anira::Context context{anira::ContextConfig{}};
    for (const double rate : {24000.0, 48000.0, 96000.0}) {
        constexpr std::size_t k_block = 256;
        GainEffect effect(context);
        effect.prepare(rate, k_block);
        const std::vector<float> input = two_sines(static_cast<std::size_t>(rate), rate);
        std::vector<float> audio = input;
        const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(static_cast<double>(k_block) / rate));
        auto next = std::chrono::steady_clock::now();
        std::size_t missed = 0;
        for (std::size_t start = 0; start + k_block <= audio.size(); start += k_block) {
            const anira::Tensor block = mono(std::span<float>(audio).subspan(start, k_block));
            const anira_status status = effect.process(block, block);
            ASSERT_FALSE(ANIRA_FAILED(status)) << rate;
            if (status == ANIRA_MISSED) { ++missed; }
            next += period;
            wait_until(next);
        }
        EXPECT_EQ(missed, 0U) << rate << " Hz";
        audio.resize(audio.size() / k_block * k_block);
        const Fit f = fit(audio, effect.get_latency(), static_cast<std::size_t>(rate / 10.0), rate);
        EXPECT_LT(f.m_worst, 2e-3F)
            << rate << " Hz, latency " << effect.get_latency() << " + " << f.m_fraction;
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
