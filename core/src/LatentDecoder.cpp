// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/HostAudio.h>
#include <anira/models/core/LatentDecoder.h>
#include <anira/models/core/Latents.h>
#include <tanh/dsp/utils/Resampler.h>

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

LatentDecoder::LatentDecoder(anira::Context& context,
                             anira::ModelConfig model,
                             anira::ContractHandle contract,
                             Stream stream)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_stream(stream)
    , m_handler(context, anira::Pipeline().add_inference(m_model))
    , m_delay(std::make_unique<detail::HostDelay>()) {
    if (stream.m_dims == 0 || stream.m_dims > k_max_latents || stream.m_hop == 0) {
        throw Error("LatentDecoder: unsupported latent size or hop");
    }
    m_static_shape = {1, static_cast<std::int64_t>(stream.m_dims), 1};
}

LatentDecoder::~LatentDecoder() = default;

void LatentDecoder::keep_static_input(std::uint32_t slot,
                                      std::vector<std::int64_t> shape,
                                      std::vector<float> values) {
    m_statics.push_back(
        {.m_slot = slot, .m_shape = std::move(shape), .m_values = std::move(values)});
}

anira::PlanReport LatentDecoder::prepare(double sample_rate,
                                         std::size_t max_block,
                                         std::size_t /*min_block*/) {
    const double ratio = m_stream.m_model_rate / sample_rate;
    const auto model_max =
        static_cast<std::size_t>(std::ceil(static_cast<double>(max_block) * ratio)) + 1;
    m_up.prepare(m_stream.m_model_rate, sample_rate, model_max);
    // The model is anchored on its audio: the host blocks are model audio samples (an empty one
    // is skipped).
    m_contract.hard_host_blocks(1.0, static_cast<double>(model_max), m_stream.m_model_rate);
    anira::PlanReport report = m_handler.prepare(m_contract);
    for (StaticInput& input : m_statics) {
        const anira_status status = m_handler.set_static_input(
            input.m_slot,
            anira::Tensor::from_host(input.m_values.data(),
                                     ANIRA_DTYPE_F32,
                                     std::span<const std::int64_t>(input.m_shape)));
        if (ANIRA_FAILED(status)) {
            throw Error("LatentDecoder: anira refused static input " +
                        std::to_string(input.m_slot) + " (" + std::to_string(status) + ")");
        }
    }
    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_mono.assign(max_block, 0.0F);
    m_model_audio.assign(model_max, 0.0F);
    m_up_audio.assign(m_up.max_output(model_max), 0.0F);
    m_plane = {m_model_audio.data()};
    const detail::PathTiming timing =
        detail::generator_timing(sample_rate,
                                 m_stream.m_model_rate,
                                 m_handler.get_latency(m_stream.m_audio_slot));
    m_latency = timing.m_latency;
    m_delay->assign(timing.m_prime, max_block + m_up_audio.size());
    reset();
    return report;
}

void LatentDecoder::reset() {
    m_handler.reset();
    m_up.reset();
    m_delay->reset();
    m_generated = 0;
    m_pushed = 0;
    m_popped = 0;
}

anira_status LatentDecoder::push_data(const anira::Tensor& latents) noexcept {
    if (latents.ndim != 2 || std::cmp_not_equal(latents.shape[0], m_stream.m_dims)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    const auto frames = static_cast<std::int64_t>(latents.shape[1]);
    anira_status status = ANIRA_OK;
    if (m_stream.m_static_latents) {
        if (frames == 0) { return ANIRA_OK; }
        // The last frame: [dims, n] read by the tensor's channels.
        for (std::uint32_t c = 0; c < m_stream.m_dims; ++c) {
            const float* row = channel(latents, c);
            if (row == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
            m_static_latents[c] = row[frames - 1];
        }
        status = m_handler.set_static_input(
            m_stream.m_latent_slot,
            anira::Tensor::from_host(m_static_latents.data(),
                                     ANIRA_DTYPE_F32,
                                     std::span<const std::int64_t>(m_static_shape)));
    } else {
        status = m_handler.push_data(latents, m_stream.m_latent_slot);
    }
    if (!ANIRA_FAILED(status)) { m_pushed += frames; }
    return status;
}

anira_status LatentDecoder::push_data(std::span<const float> frame) noexcept {
    std::array<float, k_max_latents> values{};
    std::copy_n(frame.begin(),
                std::min<std::size_t>(frame.size(), m_stream.m_dims),
                values.begin());
    const anira::Tensor latents =
        anira::Tensor::from_host(values.data(),
                                 ANIRA_DTYPE_F32,
                                 std::array<std::int64_t, 2>{m_stream.m_dims, 1});
    return push_data(latents);
}

std::size_t LatentDecoder::frames_due(std::size_t samples) const noexcept {
    const double ratio = m_stream.m_model_rate / m_host_rate;
    const std::int64_t model =
        detail::model_samples_by(m_popped + static_cast<std::int64_t>(samples), ratio);
    const std::int64_t due = model / static_cast<std::int64_t>(m_stream.m_hop);
    return due > m_pushed ? static_cast<std::size_t>(due - m_pushed) : 0;
}

template <bool Wait>
anira_status LatentDecoder::pop_block(const anira::Tensor& audio,
                                      double timeout_ms,
                                      std::size_t* delivered) noexcept {
    if (delivered != nullptr) { *delivered = 0; }
    const std::size_t n = samples(audio);
    if (n > m_max_block || channels(audio) == 0) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const double ratio = m_stream.m_model_rate / m_host_rate;
    m_popped += static_cast<std::int64_t>(n);
    const auto model_samples =
        static_cast<std::size_t>(detail::model_samples_by(m_popped, ratio) - m_generated);
    anira_status status = ANIRA_OK;
    if (model_samples > 0) {
        const anira::Tensor model = anira::Tensor::from_host_planar<float>(
            std::span<float* const>(m_plane),
            std::array<int64_t, 2>{1, static_cast<int64_t>(model_samples)});
        if constexpr (Wait) {
            status = m_handler.pop_data_wait(model, m_stream.m_audio_slot, timeout_ms);
        } else {
            status = m_handler.pop_data(model, m_stream.m_audio_slot);
        }
        if (ANIRA_FAILED(status)) { std::fill_n(m_model_audio.begin(), model_samples, 0.0F); }
        m_generated += static_cast<std::int64_t>(model_samples);
        const std::size_t made =
            m_up.process(std::span<const float>(m_model_audio.data(), model_samples), m_up_audio);
        m_delay->write(std::span<const float>(m_up_audio.data(), made));
    }
    m_delay->read(std::span<float>(m_mono.data(), n));
    if (!spread(std::span<const float>(m_mono.data(), n), audio)) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    if (delivered != nullptr && !ANIRA_FAILED(status)) { *delivered = n; }
    return status;
}

anira_status LatentDecoder::pop_data(const anira::Tensor& audio, std::size_t* delivered) noexcept {
    return pop_block<false>(audio, 0.0, delivered);
}

anira_status LatentDecoder::pop_data_wait(const anira::Tensor& audio,
                                          double timeout_ms,
                                          std::size_t* delivered) noexcept {
    return pop_block<true>(audio, timeout_ms, delivered);
}

anira_status LatentDecoder::pop_data(std::span<float> mono) noexcept {
    const anira::Tensor audio = anira::Tensor::from_host(
        mono.data(),
        ANIRA_DTYPE_F32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(mono.size())});
    return pop_data(audio);
}

std::vector<float> LatentDecoder::run(std::span<const Latents> frames, std::size_t samples) {
    reset();
    const std::size_t latency = get_latency();
    std::vector<float> out(samples + latency, 0.0F);
    std::size_t next = 0;
    for (std::size_t start = 0; start < out.size(); start += m_max_block) {
        const std::size_t count = std::min(m_max_block, out.size() - start);
        for (std::size_t k = frames_due(count); k > 0 && !frames.empty(); --k) {
            const Latents& frame = frames[std::min(next++, frames.size() - 1)];
            push_data(std::span<const float>(frame.m_values.data(), frame.m_count));
        }
        const anira::Tensor block =
            anira::Tensor::from_host(out.data() + start,
                                     ANIRA_DTYPE_F32,
                                     std::array<std::int64_t, 1>{static_cast<std::int64_t>(count)});
        pop_data_wait(block, ANIRA_WAIT_FOREVER);
    }
    out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(latency));
    out.resize(samples);
    return out;
}

}  // namespace anira::models::core
