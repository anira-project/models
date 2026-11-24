// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <anira/models/core/Latents.h>
#include <anira/models/core/Latest.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace anira::models::core {

namespace detail {
class HostDelay;
}  // namespace detail

/// The decoder of an audio autoencoder (RAVE) as a generator: latents in (set by one control
/// thread, a UI or a hand tracker, held until the next ones), audio out at the host's rate (the
/// model runs at its own, resampled), the same on every channel. Shaped like anira's
/// InferenceHandler: prepare, process, reset, get_latency.
///
/// A model whose latents are a streamed input gets the held latents at the latents' rate; one
/// whose latents are a Static input gets them before each call.
class LatentDecoder {
public:
    LatentDecoder(const LatentDecoder&) = delete;
    LatentDecoder& operator=(const LatentDecoder&) = delete;
    LatentDecoder(LatentDecoder&&) = delete;
    LatentDecoder& operator=(LatentDecoder&&) = delete;

    /// Main thread, no process() running: the model's own Hard contract with the host's rate and
    /// largest block. Starts a new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block);
    /// Main thread, no process() running: a new stream (the model's state, the resampler and the
    /// delay line start from silence; the latents stay).
    void reset();

    /// The control thread (one): the latents from now on, the first dims() of `values` (missing
    /// ones are 0).
    void set_latents(std::span<const float> values) noexcept;

    /// Audio thread: `audio` ([channels, n] or [n], float, host memory, planar or packed, n at
    /// most the prepared maximum) is written, the same on every channel. ANIRA_MISSED when anira's
    /// miss policy filled a late block.
    anira_status process(const anira::Tensor& audio) noexcept;
    anira_status process(std::span<float> mono) noexcept;

    /// Main thread after prepare(), no process() running: `samples` host samples as a new stream,
    /// each frame's latents from its m_position on (the frames in order), without the latency
    /// (anira waits for each block).
    std::vector<float> run(std::span<const Latents> frames, std::size_t samples);

    /// Host samples from set_latents() to their sound; fixed after prepare().
    [[nodiscard]] std::size_t get_latency() const noexcept { return m_latency; }
    [[nodiscard]] std::uint32_t dims() const noexcept { return m_dims; }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    [[nodiscard]] double model_rate() const noexcept { return m_model_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }
    void set_non_realtime(bool non_realtime) noexcept {
        m_non_realtime.store(non_realtime, std::memory_order_relaxed);
    }
    [[nodiscard]] bool non_realtime() const noexcept {
        return m_non_realtime.load(std::memory_order_relaxed);
    }

protected:
    struct Stream {
        double m_model_rate = 44100.0;
        std::uint32_t m_dims = 0;
        /// Latent frames per model audio sample; 0: the latents are a Static input.
        double m_latents_per_sample = 0.0;
        std::uint32_t m_latent_slot = 0;  ///< the latents' input slot
        std::uint32_t m_audio_slot = 0;   ///< the audio's output slot
    };

    LatentDecoder(anira::Context& context,
                  anira::ModelConfig model,
                  anira::ContractHandle contract,
                  Stream stream);
    ~LatentDecoder();

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

    anira_status generate(std::size_t model_samples) noexcept;

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    Stream m_stream;
    std::uint32_t m_dims;
    double m_model_rate;
    anira::InferenceHandler m_handler;
    std::vector<StaticInput> m_statics;

    Latest<std::array<float, k_max_latents>> m_latents;
    double m_host_rate = 0.0;
    std::size_t m_max_block = 0;
    std::size_t m_latency = 0;
    thl::dsp_basics::Resampler m_up;
    std::vector<float> m_model_audio;
    std::vector<float> m_up_audio;
    std::array<float*, 1> m_plane{};
    anira::Tensor m_audio_tensor{};
    std::vector<float> m_latent_block;  // [dims, n] packed: the held latents n times
    std::array<float, k_max_latents> m_static_latents{};
    std::vector<std::int64_t> m_static_shape;
    std::unique_ptr<detail::HostDelay> m_delay;
    std::int64_t m_host_samples = 0;   // since reset()
    std::int64_t m_model_samples = 0;  // generated since reset()
    std::int64_t m_latent_frames = 0;  // pushed since reset()
    std::atomic<bool> m_non_realtime{false};
};

}  // namespace anira::models::core
