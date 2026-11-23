// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/abi/tensor.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/stt/Descriptions.h>
#include <anira/models/stt/Vad.h>
#include <anira/models/stt/VadSegmenter.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::stt {

namespace {

constexpr double k_model_rate = 16000.0;
constexpr std::size_t k_queue = 4096;  // probabilities a reader may fall behind: about 2 minutes

bool given(const anira::Tensor& tensor) noexcept {
    return tensor.dtype != 0;
}

float* first_plane(const anira::Tensor& tensor) noexcept {
    if ((tensor.flags & ANIRA_TENSOR_PLANAR) != 0U) {
        return static_cast<float*>(anira_tensor_plane(&tensor, 0, ANIRA_DTYPE_F32));
    }
    return anira_tensor_data_f32(&tensor);
}

anira_status worse(anira_status a, anira_status b) noexcept {
    if (ANIRA_FAILED(a)) { return a; }
    if (ANIRA_FAILED(b)) { return b; }
    return a == ANIRA_MISSED ? a : b;
}

}  // namespace

Vad::Vad(anira::Context& context, const core::ModelSource& weights, VadConfig config)
    : core::AudioModel(context,
                       core::load_model(k_silero_vad_model_json, weights),
                       anira::ContractHandle::from_json(k_silero_vad_contract_json),
                       {.m_model_rate = k_model_rate,
                        .m_hop = static_cast<std::size_t>(config.m_chunk),
                        .m_output_slot = 0})
    , m_config(config)
    , m_segmenter(config) {
    m_value_tensor = anira::Tensor::from_host_planar<float>(std::span<float* const>(m_value_plane),
                                                            std::array<int64_t, 2>{1, 1});
}

anira::PlanReport Vad::prepare(double sample_rate, std::size_t max_block, std::size_t min_block) {
    anira::PlanReport report = prepare_stream(sample_rate, max_block, min_block);
    size_buffers();
    return report;
}

anira::PlanReport Vad::prepare(const anira::Hard& host) {
    anira::PlanReport report = prepare_stream(host);
    size_buffers();
    return report;
}

void Vad::size_buffers() {
    // Every frame an aligned block can reach: the block itself and the latency behind it.
    const double frame_samples = step() * static_cast<double>(hop());
    const auto reach = static_cast<std::size_t>(
        std::ceil(static_cast<double>(max_block() + get_latency()) / frame_samples));
    m_values.assign(reach + frame_latency() + 8, 0.0f);
    m_spans.reserve(k_queue);
    m_frames.reserve(k_queue);
    reset();
}

void Vad::reset() {
    reset_stream();
    m_segmenter.reset();
    m_spans.clear();
    m_frames.clear();
    std::ranges::fill(m_values, 0.0f);
    m_popped = 0;
    m_known = 0;
    m_latest_position.store(0, std::memory_order_relaxed);
    m_latest_value.store(0.0f, std::memory_order_relaxed);
}

anira_status Vad::process(const anira::Tensor& in,
                          const anira::Tensor& probability,
                          std::size_t* delivered) noexcept {
    if (delivered != nullptr) { *delivered = 0; }
    const std::int64_t block_start = position();
    std::size_t frames_due = 0;
    anira_status status = feed(in, frames_due);
    if (ANIRA_FAILED(status)) { return status; }
    status = worse(status, pop_frames(frames_due));

    const auto samples = static_cast<std::size_t>(in.shape[in.ndim - 1]);
    if (given(probability)) {
        float* out = first_plane(probability);
        if (out == nullptr ||
            static_cast<std::size_t>(anira_tensor_num_elements(&probability)) < samples) {
            return ANIRA_ERROR_INVALID_ARGUMENT;
        }
        write_aligned(block_start, out, samples);
    }
    if (delivered != nullptr) { *delivered = samples; }
    return status;
}

anira_status Vad::process(std::span<const float> mono, std::span<float> probability) noexcept {
    const std::array<int64_t, 1> in_shape{static_cast<int64_t>(mono.size())};
    // The tensor only reads: anira_tensor_init_host takes a non-const pointer.
    const anira::Tensor in =
        anira::Tensor::from_host(const_cast<float*>(mono.data()), ANIRA_DTYPE_F32, in_shape);
    if (probability.empty()) { return process(in); }
    if (probability.size() < mono.size()) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const anira::Tensor out =
        anira::Tensor::from_host(probability.data(), ANIRA_DTYPE_F32, in_shape);
    return process(in, out);
}

anira_status Vad::pop_frames(std::size_t frames_due) noexcept {
    anira_status result = ANIRA_OK;
    for (std::size_t i = 0; i < frames_due; ++i) {
        const anira_status popped =
            non_realtime() ? handler().pop_data_wait(m_value_tensor, 0, ANIRA_WAIT_FOREVER)
                           : handler().pop_data(m_value_tensor, 0);
        if (ANIRA_FAILED(popped)) { m_value = 0.0f; }
        result = worse(result, popped);

        // The first frame_latency() frames are anira's priming.
        const std::int64_t frame = m_popped++ - static_cast<std::int64_t>(frame_latency());
        if (frame < 0) { continue; }
        m_values[static_cast<std::size_t>(frame) % m_values.size()] = m_value;
        m_known = frame + 1;

        const core::Frame value{
            .m_position = host_position(frame * static_cast<std::int64_t>(hop())),
            .m_value = m_value};
        if (!m_frames.push(value)) { count_dropped(); }
        m_latest_position.store(value.m_position, std::memory_order_relaxed);
        m_latest_value.store(value.m_value, std::memory_order_relaxed);

        if (m_frame_limit >= 0 && frame >= m_frame_limit) { continue; }
        if (const auto span = m_segmenter.push(m_value)) {
            if (!m_spans.push(to_host(*span))) { count_dropped(); }
        }
    }
    return result;
}

void Vad::write_aligned(std::int64_t block_start, float* out, std::size_t samples) noexcept {
    const auto latency = static_cast<std::int64_t>(get_latency());
    const auto reach = static_cast<std::int64_t>(m_values.size());
    for (std::size_t i = 0; i < samples; ++i) {
        const std::int64_t source = block_start + static_cast<std::int64_t>(i) - latency;
        if (source < 0) {
            out[i] = 0.0f;
            continue;
        }
        const std::int64_t frame = frame_of(source);
        // get_latency() makes every frame an aligned sample needs known; guard anyway.
        out[i] = frame < m_known && frame >= m_known - reach
                     ? m_values[static_cast<std::size_t>(frame % reach)]
                     : 0.0f;
    }
}

SpeechSpan Vad::to_host(const SpeechSpan& span) const noexcept {
    return {.m_start = host_position(span.m_start),
            .m_end = host_position(span.m_end),
            .m_speech_start = host_position(span.m_speech_start),
            .m_speech_end = host_position(span.m_speech_end)};
}

std::vector<SpeechSpan> Vad::run(anira::Context& context,
                                 const core::ModelSource& weights,
                                 std::span<const float> mono,
                                 double sample_rate,
                                 VadConfig config) {
    constexpr std::size_t k_block = 4096;
    Vad vad(context, weights, config);
    vad.prepare(sample_rate, k_block);
    vad.set_non_realtime(true);
    // Only frames that begin inside the recording reach the segmenter.
    const double model_samples = static_cast<double>(mono.size()) / vad.step();
    vad.m_frame_limit =
        static_cast<std::int64_t>(std::ceil(model_samples / static_cast<double>(vad.hop())));

    std::vector<SpeechSpan> spans;
    SpeechSpan span;
    const auto take = [&] {
        while (vad.pop(span)) { spans.push_back(span); }
    };
    for (std::size_t start = 0; start < mono.size(); start += k_block) {
        const auto count = std::min(k_block, mono.size() - start);
        const anira_status status = vad.process(mono.subspan(start, count));
        if (ANIRA_FAILED(status)) { throw Error("Vad::run: anira refused a block"); }
        take();
    }
    // Silence after the recording until its last frame is out.
    const std::vector<float> silence(k_block, 0.0f);
    while (vad.m_known < vad.m_frame_limit) {
        if (ANIRA_FAILED(vad.process(silence))) { throw Error("Vad::run: anira refused a block"); }
        take();
    }
    if (const auto last = vad.m_segmenter.flush()) { spans.push_back(vad.to_host(*last)); }
    return spans;
}

}  // namespace anira::models::stt
