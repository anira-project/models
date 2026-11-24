// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Error.h>
#include <tanh/dsp-basics/Resampler.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "HostDelay.h"

namespace anira::models::core {

namespace {

// Channel c of a [channels, n] or [n] float host tensor, nullptr when the tensor is none of those.
float* channel(const anira::Tensor& audio, std::uint32_t c) noexcept {
    if (audio.dtype != ANIRA_DTYPE_F32) { return nullptr; }
    if ((audio.flags & ANIRA_TENSOR_PLANAR) != 0U) {
        return static_cast<float*>(anira_tensor_plane(&audio, c, ANIRA_DTYPE_F32));
    }
    float* data = anira_tensor_data_f32(&audio);
    if (data == nullptr) { return nullptr; }
    return audio.ndim == 1 ? data : data + (static_cast<std::size_t>(c) * audio.shape[1]);
}

// The host -> model -> host path on resamplers of its own, one host sample at a time (a
// resampler's output count depends only on how much it was given, never on the blocks; anira gives
// as many samples as it gets): `m_prime`, the most samples the path ever owes the host, and
// `m_latency`, where an impulse comes out behind those zeros, anira's `model_latency` included.
struct PathTiming {
    std::size_t m_prime = 0;
    std::size_t m_latency = 0;
};

PathTiming path_timing(double host_rate, double model_rate, std::uint32_t model_latency) {
    thl::dsp_basics::Resampler down;
    thl::dsp_basics::Resampler up;
    down.prepare(host_rate, model_rate, 1);
    const std::size_t model_max = std::max<std::size_t>(down.max_output(1), model_latency);
    up.prepare(model_rate, host_rate, model_max);
    std::vector<float> model(model_max, 0.0F);
    std::vector<float> host(up.max_output(model_max));
    // Long enough for anira's delay and both filters to settle.
    const auto length = static_cast<std::size_t>(
        (static_cast<double>(model_latency) * host_rate / model_rate) + (host_rate * 0.25));

    PathTiming timing;
    std::vector<float> out;
    out.reserve(length + host.size());
    const std::size_t primed =
        up.process(std::span<const float>(model.data(), model_latency), host);
    out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(primed));
    for (std::size_t i = 0; i < length; ++i) {
        const float sample = i == 0 ? 1.0F : 0.0F;  // the impulse at host sample 0
        const std::size_t to_model = down.process(std::span<const float>(&sample, 1), model);
        const std::size_t to_host =
            up.process(std::span<const float>(model.data(), to_model), host);
        out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(to_host));
    }
    // What the path owes, on fresh filters (anira's priming zeros are inside its own count).
    down.reset();
    up.reset();
    std::int64_t owed = 0;
    for (std::size_t i = 0; i < length; ++i) {
        const float zero = 0.0F;
        const std::size_t to_model = down.process(std::span<const float>(&zero, 1), model);
        owed += 1 - static_cast<std::int64_t>(
                        up.process(std::span<const float>(model.data(), to_model), host));
        timing.m_prime =
            std::max(timing.m_prime, static_cast<std::size_t>(std::max<std::int64_t>(owed, 0)));
    }
    const auto peak = std::ranges::max_element(out, {}, [](float x) { return std::abs(x); });
    timing.m_latency = timing.m_prime + static_cast<std::size_t>(peak - out.begin());
    return timing;
}

}  // namespace

AudioEffect::AudioEffect(anira::Context& context,
                         anira::ModelConfig model,
                         anira::ContractHandle contract,
                         double model_rate)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_model_rate(model_rate)
    , m_handler(context, anira::Pipeline().add_inference(m_model))
    , m_delay(std::make_unique<detail::HostDelay>()) {}

AudioEffect::~AudioEffect() = default;

anira::PlanReport AudioEffect::prepare(double sample_rate,
                                       std::size_t max_block,
                                       std::size_t min_block) {
    const double step = sample_rate / m_model_rate;
    m_down.prepare(sample_rate, m_model_rate, max_block);
    const std::size_t model_max = m_down.max_output(max_block);
    m_up.prepare(m_model_rate, sample_rate, model_max);
    // Each call carries at least one model sample (an empty one is skipped).
    const double model_min = std::max(1.0, std::floor(static_cast<double>(min_block) / step));
    m_contract.hard_host_blocks(model_min, static_cast<double>(model_max), m_model_rate);
    anira::PlanReport report = m_handler.prepare(m_contract);
    for (StaticInput& input : m_statics) {
        const anira_status status = m_handler.set_static_input(
            input.m_slot,
            anira::Tensor::from_host(input.m_values.data(),
                                     ANIRA_DTYPE_F32,
                                     std::span<const std::int64_t>(input.m_shape)));
        if (ANIRA_FAILED(status)) {
            throw Error("AudioEffect: anira refused static input " + std::to_string(input.m_slot) +
                        " (" + std::to_string(status) + ")");
        }
    }

    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_mono.assign(max_block, 0.0F);
    m_model_audio.assign(model_max, 0.0F);
    m_up_audio.assign(m_up.max_output(model_max), 0.0F);
    m_plane = {m_model_audio.data()};
    m_model_tensor = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(m_plane),
        std::array<int64_t, 2>{1, static_cast<int64_t>(model_max)});

    const PathTiming timing = path_timing(sample_rate, m_model_rate, m_handler.get_latency(0));
    m_latency = timing.m_latency;
    m_delay->assign(timing.m_prime, max_block + m_up_audio.size());
    reset();
    return report;
}

void AudioEffect::keep_static_input(std::uint32_t slot,
                                    std::vector<std::int64_t> shape,
                                    std::vector<float> values) {
    m_statics.push_back(
        {.m_slot = slot, .m_shape = std::move(shape), .m_values = std::move(values)});
}

void AudioEffect::reset() {
    m_handler.reset();
    m_down.reset();
    m_up.reset();
    m_delay->reset();
}

anira_status AudioEffect::process(std::span<float> mono) noexcept {
    std::array<float*, 1> plane{mono.data()};
    const anira::Tensor audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(plane),
        std::array<int64_t, 2>{1, static_cast<int64_t>(mono.size())});
    return process(audio);
}

anira_status AudioEffect::process(const anira::Tensor& audio) noexcept {
    if (audio.ndim != 1 && audio.ndim != 2) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto channels =
        audio.ndim == 1 ? std::uint32_t{1} : static_cast<std::uint32_t>(audio.shape[0]);
    const auto samples = static_cast<std::size_t>(audio.shape[audio.ndim - 1]);
    if (channels == 0 || samples > m_max_block || m_delay->empty()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }

    std::fill_n(m_mono.begin(), samples, 0.0F);
    for (std::uint32_t c = 0; c < channels; ++c) {
        const float* source = channel(audio, c);
        if (source == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        for (std::size_t i = 0; i < samples; ++i) { m_mono[i] += source[i]; }
    }
    if (channels > 1) {
        const float scale = 1.0F / static_cast<float>(channels);
        for (std::size_t i = 0; i < samples; ++i) { m_mono[i] *= scale; }
    }

    const std::size_t model_samples =
        m_down.process(std::span<const float>(m_mono.data(), samples), m_model_audio);
    anira_status status = ANIRA_OK;
    std::size_t made = 0;
    if (model_samples > 0) {
        m_model_tensor.shape[1] = static_cast<int64_t>(model_samples);
        status =
            non_realtime()
                ? m_handler.process_wait(m_model_tensor, 0, m_model_tensor, 0, ANIRA_WAIT_FOREVER)
                : m_handler.process(m_model_tensor, 0, m_model_tensor, 0);
        made =
            m_up.process(std::span<const float>(m_model_audio.data(), model_samples), m_up_audio);
    }

    m_delay->write(std::span<const float>(m_up_audio.data(), made));
    float* first = channel(audio, 0);
    m_delay->read(std::span<float>(first, samples));
    for (std::uint32_t c = 1; c < channels; ++c) { std::copy_n(first, samples, channel(audio, c)); }
    return status;
}

std::vector<float> AudioEffect::run(std::span<const float> mono) {
    reset();
    const bool was_non_realtime = non_realtime();
    set_non_realtime(true);
    std::vector<float> out(mono.size() + get_latency(), 0.0F);
    std::ranges::copy(mono, out.begin());
    for (std::size_t start = 0; start < out.size(); start += m_max_block) {
        const std::size_t count = std::min(m_max_block, out.size() - start);
        process(std::span<float>(out).subspan(start, count));
    }
    set_non_realtime(was_non_realtime);
    out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(get_latency()));
    out.resize(mono.size());
    return out;
}

}  // namespace anira::models::core
