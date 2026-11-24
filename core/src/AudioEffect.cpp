// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/HostAudio.h>
#include <tanh/dsp-basics/Resampler.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "HostDelay.h"

namespace anira::models::core {

namespace {

anira::Pipeline pipeline_of(const anira::ModelConfig& model,
                            const std::shared_ptr<anira::Processor>& processor) {
    anira::Pipeline pipeline;
    pipeline.add_inference(model);
    if (processor) { pipeline.add_processor(processor); }
    return pipeline;
}

}  // namespace

AudioEffect::AudioEffect(anira::Context& context,
                         anira::ModelConfig model,
                         anira::ContractHandle contract,
                         double model_rate,
                         const std::shared_ptr<anira::Processor>& processor)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_model_rate(model_rate)
    , m_handler(context, pipeline_of(m_model, processor))
    , m_delay(std::make_unique<detail::HostDelay>()) {}

AudioEffect::~AudioEffect() = default;

void AudioEffect::keep_static_input(std::uint32_t slot,
                                    std::vector<std::int64_t> shape,
                                    std::vector<float> values) {
    m_statics.push_back(
        {.m_slot = slot, .m_shape = std::move(shape), .m_values = std::move(values)});
}

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
    const detail::PathTiming timing =
        detail::effect_timing(sample_rate, m_model_rate, m_handler.get_latency(0));
    m_latency = timing.m_latency;
    m_delay->assign(timing.m_prime, max_block + m_up_audio.size());
    reset();
    return report;
}

void AudioEffect::reset() {
    m_handler.reset();
    m_down.reset();
    m_up.reset();
    m_delay->reset();
}

template <bool Wait>
anira_status AudioEffect::run_block(const anira::Tensor& in,
                                    const anira::Tensor& out,
                                    double timeout_ms,
                                    std::size_t* delivered) noexcept {
    if (delivered != nullptr) { *delivered = 0; }
    const std::size_t n = samples(in);
    if (n > m_max_block || samples(out) != n || channels(out) == 0) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (!downmix(in, m_mono)) { return ANIRA_ERROR_INVALID_ARGUMENT; }

    const std::size_t model_samples =
        m_down.process(std::span<const float>(m_mono.data(), n), m_model_audio);
    anira_status status = ANIRA_OK;
    if (model_samples > 0) {
        const anira::Tensor model = anira::Tensor::from_host_planar<float>(
            std::span<float* const>(m_plane),
            std::array<int64_t, 2>{1, static_cast<int64_t>(model_samples)});
        if constexpr (Wait) {
            status = m_handler.process_wait(model, 0, model, 0, timeout_ms);
        } else {
            status = m_handler.process(model, 0, model, 0);
        }
        const std::size_t made =
            m_up.process(std::span<const float>(m_model_audio.data(), model_samples), m_up_audio);
        m_delay->write(std::span<const float>(m_up_audio.data(), made));
    }
    m_delay->read(std::span<float>(m_mono.data(), n));
    if (!spread(std::span<const float>(m_mono.data(), n), out)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (delivered != nullptr && !ANIRA_FAILED(status)) { *delivered = n; }
    return status;
}

anira_status AudioEffect::process(const anira::Tensor& in,
                                  const anira::Tensor& out,
                                  std::size_t* delivered) noexcept {
    return run_block<false>(in, out, 0.0, delivered);
}

anira_status AudioEffect::process_wait(const anira::Tensor& in,
                                       const anira::Tensor& out,
                                       double timeout_ms,
                                       std::size_t* delivered) noexcept {
    return run_block<true>(in, out, timeout_ms, delivered);
}

anira_status AudioEffect::process(std::span<float> mono) noexcept {
    const anira::Tensor audio = anira::Tensor::from_host(
        mono.data(),
        ANIRA_DTYPE_F32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(mono.size())});
    return process(audio, audio);
}

std::vector<float> AudioEffect::run(std::span<const float> mono) {
    reset();
    std::vector<float> out(mono.size() + m_latency, 0.0F);
    std::ranges::copy(mono, out.begin());
    for (std::size_t start = 0; start < out.size(); start += m_max_block) {
        const std::size_t count = std::min(m_max_block, out.size() - start);
        const anira::Tensor block =
            anira::Tensor::from_host(out.data() + start,
                                     ANIRA_DTYPE_F32,
                                     std::array<std::int64_t, 1>{static_cast<std::int64_t>(count)});
        process_wait(block, block, ANIRA_WAIT_FOREVER);
    }
    out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(m_latency));
    out.resize(mono.size());
    return out;
}

}  // namespace anira::models::core
