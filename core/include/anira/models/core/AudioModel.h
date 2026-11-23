// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/abi/status.h>
#include <tanh/dsp-basics/Resampler.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace anira::models::core {

/// One model value and the host sample it starts at (host samples since reset()).
struct Frame {
    std::int64_t m_position = 0;
    float m_value = 0.0f;

    friend bool operator==(const Frame&, const Frame&) = default;
};

/// What every audio model of this library shares: one anira handler over the model's description,
/// fed with the host's audio (any channel count, block size and rate; downmixed and resampled to
/// the model's rate here), one clock (host samples since reset()), and anira's own calls passed
/// through (plan report and plan, latency, real-time error). A model's wrapper derives from it and
/// adds its outputs: values aligned to the host's audio in process(), results for another thread in
/// pop().
///
/// The model's input sample k (at the model's rate) is centred on host sample k * host / model rate
/// (the resampler is zero-phase on that grid). One output frame of the model covers `hop` model
/// samples; anira returns it `frame latency` frames after its input completed.
class AudioModel {
public:
    AudioModel(const AudioModel&) = delete;
    AudioModel& operator=(const AudioModel&) = delete;
    AudioModel(AudioModel&&) = delete;
    AudioModel& operator=(AudioModel&&) = delete;

    /// The host samples by which an aligned output trails the input; fixed after prepare().
    [[nodiscard]] std::size_t get_latency() const noexcept { return m_latency; }
    /// Host samples processed since reset().
    [[nodiscard]] std::int64_t position() const noexcept {
        return m_position.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double sample_rate() const noexcept { return m_host_rate; }
    [[nodiscard]] anira::PlanReport plan_report() const noexcept { return m_handler.plan_report(); }
    anira_status set_plan(std::uint32_t plan) noexcept { return m_handler.set_plan(plan); }
    [[nodiscard]] anira_status rt_error() const noexcept { return m_handler.rt_error(); }
    /// Faster than real time (an offline bounce, a test): process() waits for every result instead
    /// of letting anira's miss policy fill a late one. Until anira's Async contract (M3).
    void set_non_realtime(bool non_realtime) noexcept {
        m_non_realtime.store(non_realtime, std::memory_order_relaxed);
    }
    [[nodiscard]] bool non_realtime() const noexcept {
        return m_non_realtime.load(std::memory_order_relaxed);
    }
    /// Results a slow reader could not take (its queue was full), since prepare().
    [[nodiscard]] std::uint64_t dropped() const noexcept {
        return m_dropped.load(std::memory_order_relaxed);
    }

protected:
    struct Stream {
        double m_model_rate = 16000.0;    ///< the model's input rate
        std::size_t m_hop = 512;          ///< model samples per output frame
        std::uint32_t m_output_slot = 0;  ///< the output whose latency counts
    };

    AudioModel(anira::Context& context,
               anira::ModelConfig model,
               anira::ContractHandle contract,
               Stream stream);
    ~AudioModel() = default;

    // The wrapper's prepare() and reset() call these, then size and clear their own state.
    /// Online, on the message thread with no process() running: the model's own Hard contract
    /// (budget, warm-up, miss policy) with the host's block range and rate.
    anira::PlanReport prepare_stream(double sample_rate,
                                     std::size_t max_block,
                                     std::size_t min_block);
    /// Online with every field the caller's (block range and rate are the host's).
    anira::PlanReport prepare_stream(const anira::Hard& host);
    /// A new stream: the clock starts at 0, the model's state and the rings are cleared.
    void reset_stream();

    /// Audio thread: downmixes and resamples `in` ([channels, n] or [n], float, host memory, planar
    /// or packed; n at most the prepared maximum) and pushes it to the handler. `frames_due` is the
    /// number of output frames whose input completed with this block, for the wrapper to pop.
    anira_status feed(const anira::Tensor& in, std::size_t& frames_due) noexcept;

    anira::InferenceHandler& handler() noexcept { return m_handler; }
    /// anira's latency of the output slot in frames: the first ones popped after reset() are
    /// anira's priming, not results.
    [[nodiscard]] std::uint32_t frame_latency() const noexcept { return m_frame_latency; }
    [[nodiscard]] std::size_t hop() const noexcept { return m_stream.m_hop; }
    [[nodiscard]] std::size_t max_block() const noexcept { return m_max_block; }
    /// Host samples per model sample.
    [[nodiscard]] double step() const noexcept { return m_step; }
    /// The host sample model sample `model_sample` is centred on (rounded).
    [[nodiscard]] std::int64_t host_position(std::int64_t model_sample) const noexcept;
    /// The output frame host sample `host_sample` belongs to.
    [[nodiscard]] std::int64_t frame_of(std::int64_t host_sample) const noexcept;
    void count_dropped() noexcept { m_dropped.fetch_add(1, std::memory_order_relaxed); }

private:
    anira::PlanReport prepare_contract(const anira::ContractHandle& contract,
                                       double sample_rate,
                                       std::size_t max_block);

    anira::ModelConfig m_model;  // outlives the handler: borrowed weights
    anira::ContractHandle m_contract;
    Stream m_stream;
    anira::InferenceHandler m_handler;

    double m_host_rate = 0.0;
    double m_step = 1.0;
    std::size_t m_max_block = 0;
    std::uint32_t m_frame_latency = 0;
    std::size_t m_latency = 0;
    thl::dsp_basics::Resampler m_resampler;
    std::vector<float> m_mono;
    std::vector<float> m_resampled;
    std::array<float*, 1> m_plane{};
    anira::Tensor m_push{};
    std::int64_t m_model_samples = 0;  // pushed since reset()
    std::int64_t m_frames = 0;         // frames due since reset()
    std::atomic<std::int64_t> m_position{0};
    std::atomic<std::uint64_t> m_dropped{0};
    std::atomic<bool> m_non_realtime{false};
};

}  // namespace anira::models::core
