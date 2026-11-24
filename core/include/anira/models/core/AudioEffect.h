// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core {

/// A model that changes the sound (an amp, a compressor, a RAVE timbre transfer), shaped like
/// anira's InferenceHandler: prepare, process in place, reset, get_latency. One anira handler over
/// the model's description runs it at the rate it was trained at: the host's audio (any channel
/// count, block size up to the prepared maximum, any rate) is downmixed, resampled to the model's
/// rate, run through anira, resampled back and written to every channel, a fixed get_latency()
/// samples later. At the model's own rate both resamplers are pass-through and the latency is
/// anira's.
///
/// A model's wrapper derives from it, compiles in its description and adds the model's controls.
/// No virtual functions.
class AudioEffect {
public:
    AudioEffect(const AudioEffect&) = delete;
    AudioEffect& operator=(const AudioEffect&) = delete;
    AudioEffect(AudioEffect&&) = delete;
    AudioEffect& operator=(AudioEffect&&) = delete;

    /// Main thread, no process() running: the model's own Hard contract (budget, warm-up, miss
    /// policy) with the host's rate and block range. Starts a new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Main thread, no process() running: a new stream (the model's state, the resamplers and the
    /// delay line start from silence).
    void reset();

    /// Audio thread: `audio` ([channels, n] or [n], float, host memory, planar or packed, n at
    /// most the prepared maximum) is replaced by the model's output, the same on every channel.
    /// ANIRA_MISSED when anira's miss policy filled a late block.
    anira_status process(const anira::Tensor& audio) noexcept;
    /// Audio thread: mono in place.
    anira_status process(std::span<float> mono) noexcept;

    /// Main thread after prepare(), no process() running: a whole mono recording at the prepared
    /// rate through the model as a new stream, faster than real time, without the latency (output i
    /// belongs to input i).
    std::vector<float> run(std::span<const float> mono);

    /// Host samples by which the output trails the input; fixed after prepare().
    [[nodiscard]] std::size_t get_latency() const noexcept { return m_latency; }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    /// The rate the model runs at.
    [[nodiscard]] double model_rate() const noexcept { return m_model_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }
    /// Faster than real time (an offline bounce, a test): process() waits for every result
    /// instead of letting anira's miss policy fill a late one.
    void set_non_realtime(bool non_realtime) noexcept {
        m_non_realtime.store(non_realtime, std::memory_order_relaxed);
    }
    [[nodiscard]] bool non_realtime() const noexcept {
        return m_non_realtime.load(std::memory_order_relaxed);
    }

protected:
    /// `model`'s first streamed input and output (slot 0) carry the audio; `model_rate` is the
    /// rate it was trained at.
    AudioEffect(anira::Context& context,
                anira::ModelConfig model,
                anira::ContractHandle contract,
                double model_rate);
    ~AudioEffect() = default;

    anira::InferenceHandler& handler() noexcept { return m_handler; }
    /// Constructor: the value of a Static input (float32, `shape`), given to anira at every
    /// prepare().
    void keep_static_input(std::uint32_t slot,
                           std::vector<std::int64_t> shape,
                           std::vector<float> values);

private:
    struct StaticInput {
        std::uint32_t m_slot = 0;
        std::vector<std::int64_t> m_shape;
        std::vector<float> m_values;
    };

    void size_delay();

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    double m_model_rate;
    anira::InferenceHandler m_handler;

    double m_host_rate = 0.0;
    std::size_t m_max_block = 0;
    std::size_t m_latency = 0;
    std::size_t m_prime = 0;  // zeros in front of the delay line: it never runs dry
    thl::dsp_basics::Resampler m_down;
    thl::dsp_basics::Resampler m_up;
    std::vector<float> m_mono;
    std::vector<float> m_model_audio;
    std::vector<float> m_up_audio;
    std::array<float*, 1> m_plane{};
    anira::Tensor m_model_tensor{};
    std::vector<float> m_delay;  // a ring: the model's output at the host's rate, not yet given
    std::size_t m_read = 0;
    std::size_t m_fill = 0;
    std::atomic<bool> m_non_realtime{false};
    std::vector<StaticInput> m_statics;
};

}  // namespace anira::models::core
