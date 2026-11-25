// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <tanh/dsp/utils/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace anira::models::core {

namespace detail {
class HostDelay;
}  // namespace detail

/// A model that changes the sound (an amp, a CNN effect, a RAVE timbre transfer) as anira's
/// InferenceHandler with the model built in: prepare, process (anira's Hard entry), process_wait
/// (its _wait twin), reset, get_latency, plan_report, set_plan, rt_error. The model runs at the
/// rate it was trained at: the host's audio (any channel count, block size up to the prepared
/// maximum, any rate) is downmixed, resampled to the model's rate, run through anira, resampled
/// back and written to every output channel, a fixed get_latency() samples later. At the model's
/// own rate both resamplers are pass-through and the latency is anira's.
///
/// A model's wrapper derives from it and compiles in its description (and its anira::Processor,
/// if it has one). No virtual functions.
class AudioEffect {
public:
    AudioEffect(const AudioEffect&) = delete;
    AudioEffect& operator=(const AudioEffect&) = delete;
    AudioEffect(AudioEffect&&) = delete;
    AudioEffect& operator=(AudioEffect&&) = delete;

    /// Main thread, no process() running: the model's own Hard contract (budget, warm-up, miss
    /// policy) completed with the host's block range at the model's rate, then anira's prepare.
    /// A new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Main thread, no process() running: a new stream (anira's state, the resamplers and the
    /// delay line start from silence).
    void reset();

    /// Audio thread, anira's process: `in` through the model into `out` ([channels, n] or [n],
    /// float, host memory, planar or packed, n at most the prepared maximum; the same tensor for
    /// in place). ANIRA_MISSED when anira's miss policy filled a late block.
    anira_status process(const anira::Tensor& in,
                         const anira::Tensor& out,
                         std::size_t* delivered = nullptr) noexcept;
    /// Audio thread: mono in place.
    anira_status process(std::span<float> mono) noexcept;
    /// anira's process_wait: as process, waiting up to `timeout_ms` for a late block (an offline
    /// bounce, a test).
    anira_status process_wait(const anira::Tensor& in,
                              const anira::Tensor& out,
                              double timeout_ms,
                              std::size_t* delivered = nullptr) noexcept;

    /// Main thread after prepare(), no process() running: a whole mono recording through the
    /// model as a new stream, without the latency (output i belongs to input i).
    std::vector<float> run(std::span<const float> mono);

    /// Host samples by which the output trails the input (anira's and the resamplers'); fixed
    /// after prepare().
    [[nodiscard]] std::size_t get_latency() const noexcept { return m_latency; }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    /// The rate the model runs at.
    [[nodiscard]] double model_rate() const noexcept { return m_model_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }

protected:
    /// `model`'s streamed input 0 and output 0 carry the audio at `model_rate` (the rate it was
    /// trained at); `processor`, when given, is its pre- and post-processing
    /// (anira::Pipeline::add_processor).
    AudioEffect(anira::Context& context,
                anira::ModelConfig model,
                anira::ContractHandle contract,
                double model_rate,
                const std::shared_ptr<anira::Processor>& processor = nullptr);
    ~AudioEffect();

    anira::InferenceHandler& handler() noexcept { return m_handler; }
    /// Constructor: the value of a Static input (float32, `shape`), given at every prepare().
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
    anira_status run_block(const anira::Tensor& in,
                           const anira::Tensor& out,
                           double timeout_ms,
                           std::size_t* delivered) noexcept;

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    double m_model_rate;
    anira::InferenceHandler m_handler;
    std::vector<StaticInput> m_statics;

    double m_host_rate = 0.0;
    std::size_t m_max_block = 0;
    std::size_t m_latency = 0;
    thl::dsp::utils::Resampler m_down;
    thl::dsp::utils::Resampler m_up;
    std::vector<float> m_mono;
    std::vector<float> m_model_audio;
    std::vector<float> m_up_audio;
    std::array<float*, 1> m_plane{};
    std::unique_ptr<detail::HostDelay> m_delay;  // the output at the host's rate, not yet given
};

}  // namespace anira::models::core
