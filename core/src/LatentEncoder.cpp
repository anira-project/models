// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/HostAudio.h>
#include <anira/models/core/LatentEncoder.h>
#include <anira/models/core/Latents.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace anira::models::core {

LatentEncoder::LatentEncoder(anira::Context& context,
                             anira::ModelConfig model,
                             anira::ContractHandle contract,
                             std::size_t hop,
                             std::uint32_t dims)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_handler(context, anira::Pipeline().add_inference(m_model))
    , m_hop(hop)
    , m_dims(dims) {
    if (dims == 0 || dims > k_max_latents || hop == 0) {
        throw Error("LatentEncoder: unsupported latent size or hop");
    }
}

anira::PlanReport LatentEncoder::prepare(double sample_rate,
                                         std::size_t max_block,
                                         std::size_t min_block) {
    m_contract.hard_host_blocks(static_cast<double>(std::max<std::size_t>(min_block, 1)),
                                static_cast<double>(max_block),
                                sample_rate);
    anira::PlanReport report = m_handler.prepare(m_contract);
    m_host_rate = sample_rate;
    m_max_block = max_block;
    m_mono.assign(max_block, 0.0F);
    m_plane = {m_mono.data()};
    reset();
    return report;
}

void LatentEncoder::reset() {
    m_handler.reset();
    m_pushed = 0;
    m_popped = 0;
}

anira_status LatentEncoder::push_data(const anira::Tensor& audio) noexcept {
    const std::size_t n = samples(audio);
    if (n > m_max_block || !downmix(audio, m_mono)) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const anira::Tensor mono =
        anira::Tensor::from_host_planar<float>(std::span<float* const>(m_plane),
                                               std::array<int64_t, 2>{1, static_cast<int64_t>(n)});
    const anira_status status = m_handler.push_data(mono, 0);
    if (!ANIRA_FAILED(status)) { m_pushed += static_cast<std::int64_t>(n); }
    return status;
}

anira_status LatentEncoder::push_data(std::span<const float> mono) noexcept {
    // The tensor only reads: anira_tensor_init_host takes a non-const pointer.
    const anira::Tensor audio = anira::Tensor::from_host(
        const_cast<float*>(mono.data()),
        ANIRA_DTYPE_F32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(mono.size())});
    return push_data(audio);
}

std::size_t LatentEncoder::frames_due() const noexcept {
    return static_cast<std::size_t>((m_pushed / static_cast<std::int64_t>(m_hop)) - m_popped);
}

anira_status LatentEncoder::pop_data(const anira::Tensor& latents,
                                     std::size_t* delivered) noexcept {
    const anira_status status = m_handler.pop_data(latents, 0, delivered);
    if (!ANIRA_FAILED(status)) { ++m_popped; }
    return status;
}

anira_status LatentEncoder::pop_data_wait(const anira::Tensor& latents,
                                          double timeout_ms,
                                          std::size_t* delivered) noexcept {
    const anira_status status = m_handler.pop_data_wait(latents, 0, timeout_ms, delivered);
    if (!ANIRA_FAILED(status)) { ++m_popped; }
    return status;
}

template <bool Wait>
anira_status LatentEncoder::pop_frame(Latents& latents, double timeout_ms) noexcept {
    latents = {};
    latents.m_count = m_dims;
    latents.m_position =
        (m_popped - static_cast<std::int64_t>(get_latency())) * static_cast<std::int64_t>(m_hop);
    const anira::Tensor tensor = anira::Tensor::from_host(latents.m_values.data(),
                                                          ANIRA_DTYPE_F32,
                                                          std::array<std::int64_t, 2>{m_dims, 1});
    if constexpr (Wait) { return pop_data_wait(tensor, timeout_ms); }
    return pop_data(tensor);
}

anira_status LatentEncoder::pop_data(Latents& latents) noexcept {
    return pop_frame<false>(latents, 0.0);
}

anira_status LatentEncoder::pop_data_wait(Latents& latents, double timeout_ms) noexcept {
    return pop_frame<true>(latents, timeout_ms);
}

std::vector<Latents> LatentEncoder::run(std::span<const float> mono) {
    reset();
    std::vector<Latents> frames;
    // The recording, then anira's latency in silence: every frame of the recording comes out.
    std::vector<float> audio(mono.begin(), mono.end());
    audio.resize(mono.size() + ((get_latency() + 1) * m_hop), 0.0F);
    for (std::size_t start = 0; start < audio.size(); start += m_max_block) {
        const std::size_t count = std::min(m_max_block, audio.size() - start);
        push_data(std::span<const float>(audio).subspan(start, count));
        while (frames_due() > 0) {
            Latents latents;
            if (ANIRA_FAILED(pop_data_wait(latents, ANIRA_WAIT_FOREVER))) { break; }
            if (latents.m_position >= 0 && std::cmp_less(latents.m_position, mono.size())) {
                frames.push_back(latents);
            }
        }
    }
    return frames;
}

}  // namespace anira::models::core
