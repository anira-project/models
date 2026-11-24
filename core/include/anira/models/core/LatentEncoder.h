// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <anira/models/core/Latents.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core {

/// The encoder of an audio autoencoder (RAVE) as anira's InferenceHandler with the model built in:
/// push_data (the host's audio: downmixed and resampled to the rate the model was trained at),
/// pop_data (one latent frame per hop() model samples pushed: frames_due() of them, each with the
/// host sample its audio starts at), their _wait twins, reset, get_latency, plan_report, set_plan,
/// rt_error.
class LatentEncoder {
public:
    LatentEncoder(const LatentEncoder&) = delete;
    LatentEncoder& operator=(const LatentEncoder&) = delete;
    LatentEncoder(LatentEncoder&&) = delete;
    LatentEncoder& operator=(LatentEncoder&&) = delete;

    /// Main thread, no push running: the model's own Hard contract completed with the host's rate
    /// and block range, then anira's prepare. A new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Main thread, no push running: a new stream.
    void reset();

    /// Audio thread, anira's push_data: [channels, n] or [n] float host audio (downmixed).
    anira_status push_data(const anira::Tensor& audio) noexcept;
    anira_status push_data(std::span<const float> mono) noexcept;
    /// Audio thread: the frames whose audio has been pushed and that have not been popped.
    [[nodiscard]] std::size_t frames_due() const noexcept;
    /// Audio thread, anira's pop_data: one frame into `latents` ([dims(), 1] float host memory).
    /// The first get_latency() frames after reset() are anira's priming (zeros).
    anira_status pop_data(const anira::Tensor& latents, std::size_t* delivered = nullptr) noexcept;
    /// Audio thread: one frame with the host sample its audio starts at (negative for anira's
    /// priming).
    anira_status pop_data(Latents& latents) noexcept;
    anira_status pop_data_wait(const anira::Tensor& latents,
                               double timeout_ms,
                               std::size_t* delivered = nullptr) noexcept;
    anira_status pop_data_wait(Latents& latents, double timeout_ms) noexcept;

    /// Main thread after prepare(), no push running: the frames of a mono recording whose audio
    /// starts inside it, as a new stream, anira's priming left out.
    std::vector<Latents> run(std::span<const float> mono);

    /// anira's latency in frames.
    [[nodiscard]] std::size_t get_latency() const noexcept { return m_handler.get_latency(0); }
    /// Model samples per frame.
    [[nodiscard]] std::size_t hop() const noexcept { return m_hop; }
    [[nodiscard]] double model_rate() const noexcept { return m_model_rate; }
    [[nodiscard]] std::uint32_t dims() const noexcept { return m_dims; }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }

protected:
    /// `model`'s streamed input 0 carries the audio at `model_rate`, its streamed output 0 the
    /// latents: `dims` channels, one frame per `hop` model samples.
    LatentEncoder(anira::Context& context,
                  anira::ModelConfig model,
                  anira::ContractHandle contract,
                  double model_rate,
                  std::size_t hop,
                  std::uint32_t dims);
    ~LatentEncoder() = default;

    anira::InferenceHandler& handler() noexcept { return m_handler; }

private:
    template <bool Wait>
    anira_status pop_frame(Latents& latents, double timeout_ms) noexcept;

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    anira::InferenceHandler m_handler;
    double m_model_rate;
    std::size_t m_hop;
    std::uint32_t m_dims;

    double m_host_rate = 0.0;
    std::size_t m_max_block = 0;
    thl::dsp_basics::Resampler m_down;
    std::vector<float> m_mono;
    std::vector<float> m_model_audio;
    std::array<float*, 1> m_plane{};
    std::int64_t m_pushed = 0;  // model samples since reset()
    std::int64_t m_popped = 0;  // frames since reset(), anira's priming included
};

}  // namespace anira::models::core
