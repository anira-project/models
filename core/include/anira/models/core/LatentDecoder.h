// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <anira/models/core/Latents.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core {

/// The decoder of an audio autoencoder (RAVE) as anira's InferenceHandler with the model built in,
/// at the host's rate as anira's RAVE example runs it (anchored on its audio): push_data (latent
/// frames, one per hop() samples: frames_due(n) before popping n samples), pop_data (the audio,
/// the same on every channel), pop_data_wait, reset, get_latency, plan_report, set_plan,
/// rt_error. A model whose latents are a Static input takes the last frame pushed for its next
/// inferences.
class LatentDecoder {
public:
    LatentDecoder(const LatentDecoder&) = delete;
    LatentDecoder& operator=(const LatentDecoder&) = delete;
    LatentDecoder(LatentDecoder&&) = delete;
    LatentDecoder& operator=(LatentDecoder&&) = delete;

    /// Main thread, no push or pop running: the model's own Hard contract completed with the
    /// host's rate and block range (of audio samples), then anira's prepare. A new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Main thread, no push or pop running: a new stream.
    void reset();

    /// Audio thread, anira's push_data: latent frames, [dims(), n] float host memory.
    anira_status push_data(const anira::Tensor& latents) noexcept;
    /// Audio thread: one frame, its first dims() values.
    anira_status push_data(std::span<const float> frame) noexcept;
    /// Audio thread: the frames to push before popping the next `samples` samples (one per hop()
    /// samples, counted from reset()).
    [[nodiscard]] std::size_t frames_due(std::size_t samples) const noexcept;
    /// Audio thread, anira's pop_data: `audio` ([channels, n] or [n], float host memory, n at most
    /// the prepared maximum) written, the same on every channel.
    anira_status pop_data(const anira::Tensor& audio, std::size_t* delivered = nullptr) noexcept;
    anira_status pop_data(std::span<float> mono) noexcept;
    anira_status pop_data_wait(const anira::Tensor& audio,
                               double timeout_ms,
                               std::size_t* delivered = nullptr) noexcept;

    /// Main thread after prepare(), no push or pop running: `samples` samples from `frames` (one
    /// per hop(), the last one held), as a new stream, without the latency.
    std::vector<float> run(std::span<const Latents> frames, std::size_t samples);

    /// anira's latency in samples.
    [[nodiscard]] std::size_t get_latency() const noexcept {
        return m_handler.get_latency(m_stream.m_audio_slot);
    }
    [[nodiscard]] std::size_t hop() const noexcept { return m_stream.m_hop; }
    [[nodiscard]] std::uint32_t dims() const noexcept { return m_stream.m_dims; }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }

protected:
    struct Stream {
        std::uint32_t m_dims = 0;
        std::size_t m_hop = 0;            ///< samples per latent frame
        bool m_static_latents = false;    ///< the latents are a Static input ([1, dims, 1])
        std::uint32_t m_latent_slot = 0;  ///< the latents' input slot
        std::uint32_t m_audio_slot = 0;   ///< the audio's output slot
    };

    LatentDecoder(anira::Context& context,
                  anira::ModelConfig model,
                  anira::ContractHandle contract,
                  Stream stream);
    ~LatentDecoder() = default;

    anira::InferenceHandler& handler() noexcept { return m_handler; }
    /// Constructor: the value of another Static input (float32, `shape`), given at every prepare().
    void keep_static_input(std::uint32_t slot,
                           std::vector<std::int64_t> shape,
                           std::vector<float> values);

private:
    struct StaticInput {
        std::uint32_t m_slot = 0;
        std::vector<std::int64_t> m_shape;
        std::vector<float> m_values;
    };

    template <bool Wait>
    anira_status pop_block(const anira::Tensor& audio,
                           double timeout_ms,
                           std::size_t* delivered) noexcept;

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    Stream m_stream;
    anira::InferenceHandler m_handler;
    std::vector<StaticInput> m_statics;

    double m_host_rate = 0.0;
    std::size_t m_max_block = 0;
    std::vector<float> m_mono;
    std::array<float*, 1> m_plane{};
    std::array<float, k_max_latents> m_static_latents{};
    std::array<std::int64_t, 3> m_static_shape{};
    std::int64_t m_pushed = 0;  // frames since reset()
    std::int64_t m_popped = 0;  // samples since reset()
};

}  // namespace anira::models::core
