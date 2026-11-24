// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/LatentDecoder.h>
#include <anira/models/core/Latents.h>
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

float* channel(const anira::Tensor& audio, std::uint32_t c) noexcept {
    if (audio.dtype != ANIRA_DTYPE_F32) { return nullptr; }
    if ((audio.flags & ANIRA_TENSOR_PLANAR) != 0U) {
        return static_cast<float*>(anira_tensor_plane(&audio, c, ANIRA_DTYPE_F32));
    }
    float* data = anira_tensor_data_f32(&audio);
    if (data == nullptr) { return nullptr; }
    return audio.ndim == 1 ? data : data + (static_cast<std::size_t>(c) * audio.shape[1]);
}

// Model samples due after `host` host samples: floor(host * model / host rate).
std::int64_t model_samples_by(std::int64_t host, double ratio) noexcept {
    return static_cast<std::int64_t>(std::floor(static_cast<double>(host) * ratio));
}

struct Timing {
    std::size_t m_prime = 0;    // the most host samples the path ever owes the host
    std::size_t m_latency = 0;  // a model impulse anira gives `model_latency` samples late
};

// The model -> host path on a resampler of its own, one host sample at a time (the counts depend
// only on how much was given).
Timing path_timing(double host_rate, double model_rate, std::uint32_t model_latency) {
    const double ratio = model_rate / host_rate;
    thl::dsp_basics::Resampler up;
    const std::size_t model_max = static_cast<std::size_t>(std::ceil(ratio)) + 1;
    up.prepare(model_rate, host_rate, model_max);
    std::vector<float> model(model_max, 0.0F);
    std::vector<float> host(up.max_output(model_max));
    const auto length = static_cast<std::int64_t>((static_cast<double>(model_latency) / ratio) +
                                                  (host_rate * 0.25));

    Timing timing;
    std::int64_t owed = 0;
    for (std::int64_t n = 1; n <= length; ++n) {
        const auto m =
            static_cast<std::size_t>(model_samples_by(n, ratio) - model_samples_by(n - 1, ratio));
        owed += 1 - static_cast<std::int64_t>(
                        up.process(std::span<const float>(model.data(), m), host));
        timing.m_prime =
            std::max(timing.m_prime, static_cast<std::size_t>(std::max<std::int64_t>(owed, 0)));
    }

    up.reset();
    std::vector<float> out;
    std::int64_t given = 0;
    for (std::int64_t n = 1; n <= length; ++n) {
        const std::int64_t due = model_samples_by(n, ratio);
        const auto m = static_cast<std::size_t>(due - given);
        for (std::size_t i = 0; i < m; ++i) {
            model[i] = given + static_cast<std::int64_t>(i) == model_latency ? 1.0F : 0.0F;
        }
        given = due;
        const std::size_t made = up.process(std::span<const float>(model.data(), m), host);
        out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(made));
    }
    const auto peak = std::ranges::max_element(out, {}, [](float x) { return std::abs(x); });
    timing.m_latency = timing.m_prime + static_cast<std::size_t>(peak - out.begin());
    return timing;
}

}  // namespace

LatentDecoder::LatentDecoder(anira::Context& context,
                             anira::ModelConfig model,
                             anira::ContractHandle contract,
                             Stream stream)
    : m_model(std::move(model))
    , m_contract(std::move(contract))
    , m_stream(stream)
    , m_dims(stream.m_dims)
    , m_model_rate(stream.m_model_rate)
    , m_handler(context, anira::Pipeline().add_inference(m_model))
    , m_delay(std::make_unique<detail::HostDelay>()) {
    if (m_dims == 0 || m_dims > k_max_latents) {
        throw Error("LatentDecoder: unsupported latent size");
    }
    m_static_shape = {1, static_cast<std::int64_t>(m_dims), 1};
}

LatentDecoder::~LatentDecoder() = default;

void LatentDecoder::keep_static_input(std::uint32_t slot,
                                      std::vector<std::int64_t> shape,
                                      std::vector<float> values) {
    m_statics.push_back(
        {.m_slot = slot, .m_shape = std::move(shape), .m_values = std::move(values)});
}

anira::PlanReport LatentDecoder::prepare(double sample_rate, std::size_t max_block) {
    const double ratio = m_model_rate / sample_rate;
    const auto model_max =
        static_cast<std::size_t>(std::ceil(static_cast<double>(max_block) * ratio)) + 1;
    m_up.prepare(m_model_rate, sample_rate, model_max);
    // The audio is the anchor: blocks of model audio samples.
    m_contract.hard_host_blocks(1.0, static_cast<double>(model_max), m_model_rate);
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
    m_model_audio.assign(model_max, 0.0F);
    m_up_audio.assign(m_up.max_output(model_max), 0.0F);
    m_plane = {m_model_audio.data()};
    m_audio_tensor = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(m_plane),
        std::array<int64_t, 2>{1, static_cast<int64_t>(model_max)});
    const auto max_latents = static_cast<std::size_t>(
        std::ceil(static_cast<double>(model_max) * m_stream.m_latents_per_sample) + 1);
    m_latent_block.assign(m_dims * max_latents, 0.0F);

    const Timing timing =
        path_timing(sample_rate, m_model_rate, m_handler.get_latency(m_stream.m_audio_slot));
    m_latency = timing.m_latency;
    m_delay->assign(timing.m_prime, max_block + m_up_audio.size());
    reset();
    return report;
}

void LatentDecoder::reset() {
    m_handler.reset();
    m_up.reset();
    m_delay->reset();
    m_host_samples = 0;
    m_model_samples = 0;
    m_latent_frames = 0;
}

void LatentDecoder::set_latents(std::span<const float> values) noexcept {
    std::array<float, k_max_latents> latents{};
    std::copy_n(values.begin(), std::min<std::size_t>(values.size(), m_dims), latents.begin());
    m_latents.write(latents);
}

anira_status LatentDecoder::generate(std::size_t model_samples) noexcept {
    const std::array<float, k_max_latents>& latents = m_latents.read();
    m_audio_tensor.shape[1] = static_cast<int64_t>(model_samples);
    const bool wait = non_realtime();
    const std::uint32_t audio_slot = m_stream.m_audio_slot;
    anira_status status = ANIRA_OK;
    if (m_stream.m_latents_per_sample > 0.0) {
        // The latent frames due by the end of this block, each the held latents.
        const auto due = static_cast<std::int64_t>(std::floor(
            static_cast<double>(m_model_samples + static_cast<std::int64_t>(model_samples)) *
            m_stream.m_latents_per_sample));
        const auto count = static_cast<std::size_t>(due - m_latent_frames);
        for (std::size_t c = 0; c < m_dims; ++c) {
            std::fill_n(m_latent_block.begin() + static_cast<std::ptrdiff_t>(c * count),
                        count,
                        latents[c]);
        }
        const anira::Tensor in = anira::Tensor::from_host(
            m_latent_block.data(),
            ANIRA_DTYPE_F32,
            std::array<std::int64_t, 2>{m_dims, static_cast<std::int64_t>(count)});
        status = wait ? m_handler.process_wait(in,
                                               m_stream.m_latent_slot,
                                               m_audio_tensor,
                                               audio_slot,
                                               ANIRA_WAIT_FOREVER)
                      : m_handler.process(in, m_stream.m_latent_slot, m_audio_tensor, audio_slot);
        m_latent_frames = due;
    } else {
        std::copy_n(latents.begin(), m_dims, m_static_latents.begin());
        status = m_handler.set_static_input(
            m_stream.m_latent_slot,
            anira::Tensor::from_host(m_static_latents.data(),
                                     ANIRA_DTYPE_F32,
                                     std::span<const std::int64_t>(m_static_shape)));
        if (!ANIRA_FAILED(status)) {
            status = wait ? m_handler.pop_data_wait(m_audio_tensor, audio_slot, ANIRA_WAIT_FOREVER)
                          : m_handler.pop_data(m_audio_tensor, audio_slot);
        }
    }
    if (ANIRA_FAILED(status)) { std::fill_n(m_model_audio.begin(), model_samples, 0.0F); }
    m_model_samples += static_cast<std::int64_t>(model_samples);
    return status;
}

anira_status LatentDecoder::process(std::span<float> mono) noexcept {
    std::array<float*, 1> plane{mono.data()};
    const anira::Tensor audio = anira::Tensor::from_host_planar<float>(
        std::span<float* const>(plane),
        std::array<int64_t, 2>{1, static_cast<int64_t>(mono.size())});
    return process(audio);
}

anira_status LatentDecoder::process(const anira::Tensor& audio) noexcept {
    if (audio.ndim != 1 && audio.ndim != 2) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const auto channels =
        audio.ndim == 1 ? std::uint32_t{1} : static_cast<std::uint32_t>(audio.shape[0]);
    const auto samples = static_cast<std::size_t>(audio.shape[audio.ndim - 1]);
    if (channels == 0 || samples > m_max_block || m_delay->empty()) {
        return ANIRA_ERROR_INVALID_ARGUMENT;
    }
    for (std::uint32_t c = 0; c < channels; ++c) {
        if (channel(audio, c) == nullptr) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    }

    const double ratio = m_model_rate / m_host_rate;
    m_host_samples += static_cast<std::int64_t>(samples);
    const auto model_samples =
        static_cast<std::size_t>(model_samples_by(m_host_samples, ratio) - m_model_samples);
    anira_status status = ANIRA_OK;
    if (model_samples > 0) {
        status = generate(model_samples);
        const std::size_t made =
            m_up.process(std::span<const float>(m_model_audio.data(), model_samples), m_up_audio);
        m_delay->write(std::span<const float>(m_up_audio.data(), made));
    }
    float* first = channel(audio, 0);
    m_delay->read(std::span<float>(first, samples));
    for (std::uint32_t c = 1; c < channels; ++c) { std::copy_n(first, samples, channel(audio, c)); }
    return status;
}

std::vector<float> LatentDecoder::run(std::span<const Latents> frames, std::size_t samples) {
    reset();
    const bool was_non_realtime = non_realtime();
    set_non_realtime(true);
    std::vector<float> out(samples + m_latency, 0.0F);
    std::size_t next = 0;
    for (std::size_t start = 0; start < out.size(); start += m_max_block) {
        while (next < frames.size() && std::cmp_less_equal(frames[next].m_position, start)) {
            set_latents(std::span<const float>(frames[next].m_values.data(), frames[next].m_count));
            ++next;
        }
        const std::size_t count = std::min(m_max_block, out.size() - start);
        process(std::span<float>(out).subspan(start, count));
    }
    set_non_realtime(was_non_realtime);
    out.erase(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(m_latency));
    out.resize(samples);
    return out;
}

}  // namespace anira::models::core
