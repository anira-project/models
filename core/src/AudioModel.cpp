// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/Model.h>
#include <tanh/dsp/utils/Resampler.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>

namespace anira::models::core {

namespace {

// Input samples a resampler needs beyond an output sample's centre before it emits that sample:
// the first output comes after lookahead + 1 input samples.
std::size_t resampler_lookahead(double input_rate, double output_rate) {
    thl::dsp::utils::Resampler probe;
    probe.prepare(input_rate, output_rate, 1);
    if (probe.is_passthrough()) { return 0; }
    const float zero = 0.0f;
    std::array<float, 8> out{};
    for (std::size_t fed = 1; fed < 1U << 20U; ++fed) {
        if (probe.process(std::span<const float>(&zero, 1), out) > 0) { return fed - 1; }
    }
    throw Error("the resampler never produced output");
}

// Channel c of a [channels, n] or [n] float host tensor, nullptr when the tensor is none of those.
const float* channel(const anira::Tensor& in, std::uint32_t c) noexcept {
    if (in.dtype != ANIRA_DTYPE_F32) { return nullptr; }
    if ((in.flags & ANIRA_TENSOR_PLANAR) != 0U) {
        return static_cast<const float*>(anira_tensor_plane(&in, c, ANIRA_DTYPE_F32));
    }
    const float* data = anira_tensor_data_f32(&in);
    if (data == nullptr) { return nullptr; }
    return in.ndim == 1 ? data : data + (static_cast<std::size_t>(c) * in.shape[1]);
}

}  // namespace

AudioModel::AudioModel(anira::Context& context,
                       anira::ModelConfig model,
                       anira::ContractHandle contract,
                       Stream stream)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_stream(stream)
    , m_handler(context, model_pipeline(m_model)) {}

anira::PlanReport AudioModel::prepare_stream(double sample_rate,
                                             std::size_t max_block,
                                             std::size_t min_block) {
    const double step = sample_rate / m_stream.m_model_rate;
    m_resampler.prepare(sample_rate, m_stream.m_model_rate, max_block);
    // Each push carries at least one model sample (an empty one is skipped).
    const double model_min = std::max(1.0, std::floor(static_cast<double>(min_block) / step));
    m_contract.hard_host_blocks(model_min,
                                static_cast<double>(m_resampler.max_output(max_block)),
                                m_stream.m_model_rate);
    return prepare_contract(m_contract, sample_rate, max_block);
}

anira::PlanReport AudioModel::prepare_stream(const anira::Hard& host) {
    const double step = host.host_rate / m_stream.m_model_rate;
    const auto max_block = static_cast<std::size_t>(std::ceil(host.host_block_max));
    m_resampler.prepare(host.host_rate, m_stream.m_model_rate, max_block);
    anira::Hard model = host;
    model.host_block_min = std::max(1.0, std::floor(host.host_block_min / step));
    model.host_block_max = static_cast<double>(m_resampler.max_output(max_block));
    model.host_rate = m_stream.m_model_rate;
    return prepare_contract(anira::ContractHandle(model), host.host_rate, max_block);
}

anira::PlanReport AudioModel::prepare_contract(const anira::ContractHandle& contract,
                                               double sample_rate,
                                               std::size_t max_block) {
    m_host_rate = sample_rate;
    m_step = sample_rate / m_stream.m_model_rate;
    m_max_block = max_block;
    anira::PlanReport report = m_handler.prepare(contract);

    m_mono.assign(max_block, 0.0f);
    m_resampled.assign(m_resampler.max_output(max_block), 0.0f);
    m_plane = {m_resampled.data()};
    m_push = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(m_plane),
        std::array<int64_t, 2>{1, static_cast<int64_t>(m_resampled.size())});

    // Frame c's value is popped when model sample hop * (c + L + 1) has been pushed; that sample
    // is out of the resampler once the host is lookahead samples past its centre. A frame's first
    // host sample is the one that waits longest.
    m_frame_latency = m_handler.get_latency(m_stream.m_output_slot);
    const auto hop = static_cast<double>(m_stream.m_hop);
    const double model_wait = (hop * (m_frame_latency + 1.0)) - 1.0;
    m_latency = static_cast<std::size_t>(std::ceil(
        (model_wait * m_step) +
        static_cast<double>(resampler_lookahead(sample_rate, m_stream.m_model_rate)) - 1e-9));
    m_dropped.store(0, std::memory_order_relaxed);
    reset_stream();
    return report;
}

void AudioModel::reset_stream() {
    m_handler.reset();
    m_resampler.reset();
    m_model_samples = 0;
    m_frames = 0;
    m_position.store(0, std::memory_order_relaxed);
}

anira_status AudioModel::feed(const anira::Tensor& in, std::size_t& frames_due) noexcept {
    frames_due = 0;
    if (in.ndim != 1 && in.ndim != 2) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto channels = in.ndim == 1 ? std::uint32_t{1} : static_cast<std::uint32_t>(in.shape[0]);
    const auto samples = static_cast<std::size_t>(in.shape[in.ndim - 1]);
    if (channels == 0 || samples > m_max_block) { return ANIRA_ERROR_INVALID_ARGUMENT; }

    std::fill_n(m_mono.begin(), samples, 0.0f);
    for (std::uint32_t c = 0; c < channels; ++c) {
        const float* source = channel(in, c);
        if (source == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
        for (std::size_t i = 0; i < samples; ++i) { m_mono[i] += source[i]; }
    }
    if (channels > 1) {
        const float scale = 1.0f / static_cast<float>(channels);
        for (std::size_t i = 0; i < samples; ++i) { m_mono[i] *= scale; }
    }

    const std::size_t produced =
        m_resampler.process(std::span<const float>(m_mono.data(), samples), m_resampled);
    anira_status status = ANIRA_OK;
    if (produced > 0) {
        m_push.shape[1] = static_cast<int64_t>(produced);
        status = m_handler.push_data(m_push, 0);
        m_model_samples += static_cast<std::int64_t>(produced);
        const std::int64_t frames = m_model_samples / static_cast<std::int64_t>(m_stream.m_hop);
        frames_due = static_cast<std::size_t>(frames - m_frames);
        m_frames = frames;
    }
    m_position.fetch_add(static_cast<std::int64_t>(samples), std::memory_order_relaxed);
    return status;
}

std::int64_t AudioModel::host_position(std::int64_t model_sample) const noexcept {
    return std::llround(static_cast<double>(model_sample) * m_step);
}

std::int64_t AudioModel::frame_of(std::int64_t host_sample) const noexcept {
    return static_cast<std::int64_t>(std::floor(static_cast<double>(host_sample) /
                                                (m_step * static_cast<double>(m_stream.m_hop))));
}

}  // namespace anira::models::core
