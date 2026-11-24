// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/Error.h>
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
                             double model_rate,
                             std::size_t hop,
                             std::uint32_t dims)
    : AudioModel(context,
                 std::move(model),
                 std::move(contract),
                 {.m_model_rate = model_rate, .m_hop = hop, .m_output_slot = 0})
    , m_dims(dims) {
    if (dims == 0 || dims > k_max_latents) {
        throw Error("LatentEncoder: unsupported latent size");
    }
    // One frame: [dims, 1], packed.
    m_values_tensor = anira::Tensor::from_host(m_values.data(),
                                               ANIRA_DTYPE_F32,
                                               std::array<std::int64_t, 2>{dims, 1});
}

anira::PlanReport LatentEncoder::prepare(double sample_rate,
                                         std::size_t max_block,
                                         std::size_t min_block) {
    anira::PlanReport report = prepare_stream(sample_rate, max_block, min_block);
    m_frames.reserve(k_queue);
    reset();
    return report;
}

void LatentEncoder::reset() {
    reset_stream();
    m_frames.clear();
    m_latest.reset();
    m_popped = 0;
}

anira_status LatentEncoder::process(const anira::Tensor& audio) noexcept {
    std::size_t frames_due = 0;
    const anira_status status = feed(audio, frames_due);
    if (ANIRA_FAILED(status)) { return status; }
    const anira_status popped = pop_frames(frames_due);
    return ANIRA_FAILED(popped) || popped == ANIRA_MISSED ? popped : status;
}

anira_status LatentEncoder::process(std::span<const float> mono) noexcept {
    // The tensor only reads: anira_tensor_init_host takes a non-const pointer.
    const anira::Tensor audio = anira::Tensor::from_host(
        const_cast<float*>(mono.data()),
        ANIRA_DTYPE_F32,
        std::array<std::int64_t, 1>{static_cast<std::int64_t>(mono.size())});
    return process(audio);
}

anira_status LatentEncoder::pop_frames(std::size_t frames_due) noexcept {
    anira_status result = ANIRA_OK;
    for (std::size_t i = 0; i < frames_due; ++i) {
        const anira_status popped =
            non_realtime() ? handler().pop_data_wait(m_values_tensor, 0, ANIRA_WAIT_FOREVER)
                           : handler().pop_data(m_values_tensor, 0);
        if (ANIRA_FAILED(popped)) { std::ranges::fill(m_values, 0.0F); }
        if (ANIRA_FAILED(popped) || popped == ANIRA_MISSED) { result = popped; }

        // The first frame_latency() frames are anira's priming.
        const std::int64_t frame = m_popped++ - static_cast<std::int64_t>(frame_latency());
        if (frame < 0) { continue; }
        Latents latents;
        latents.m_position = host_position(frame * static_cast<std::int64_t>(hop()));
        latents.m_count = m_dims;
        std::copy_n(m_values.begin(), m_dims, latents.m_values.begin());
        m_latest.write(latents);
        if (!m_frames.push(latents)) { count_dropped(); }
    }
    return result;
}

std::vector<Latents> LatentEncoder::run(std::span<const float> mono) {
    reset();
    const bool was_non_realtime = non_realtime();
    set_non_realtime(true);
    std::vector<Latents> frames;
    // The recording, then the latency in silence: every frame of the recording comes out.
    std::vector<float> audio(mono.begin(), mono.end());
    audio.resize(mono.size() + get_latency() + max_block(), 0.0F);
    for (std::size_t start = 0; start < audio.size(); start += max_block()) {
        const std::size_t count = std::min(max_block(), audio.size() - start);
        process(std::span<const float>(audio).subspan(start, count));
        Latents latents;
        while (pop(latents)) {
            if (std::cmp_less(latents.m_position, mono.size())) { frames.push_back(latents); }
        }
    }
    set_non_realtime(was_non_realtime);
    return frames;
}

}  // namespace anira::models::core
