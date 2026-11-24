// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>
#include <anira/models/core/AudioModel.h>
#include <anira/models/core/Latents.h>
#include <anira/models/core/Latest.h>
#include <anira/models/core/SpscQueue.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core {

/// The encoder of an audio autoencoder (RAVE): the host's audio in (any channel count, block size
/// up to the prepared maximum, any rate: downmixed and resampled to the model's rate, see
/// core::AudioModel), latent frames out, one per `hop` model samples, each with the host sample
/// its audio starts at. Shaped like anira's InferenceHandler: prepare, process, reset; the audio
/// is never edited.
///
///   - Audio thread: process(audio) never waits.
///   - Reader thread: pop(latents) gives every frame in order, latest() the newest.
///   - Offline: run().
class LatentEncoder : public AudioModel {
public:
    /// Main thread, no process() running: the model's own Hard contract with the host's rate and
    /// block range. Starts a new stream.
    anira::PlanReport prepare(double sample_rate, std::size_t max_block, std::size_t min_block = 1);
    /// Main thread, neither process() nor the reader running: a new stream.
    void reset();

    /// Audio thread: [channels, n] or [n] float host audio, n at most the prepared maximum.
    /// ANIRA_MISSED when anira filled a late frame (zeros).
    anira_status process(const anira::Tensor& audio) noexcept;
    anira_status process(std::span<const float> mono) noexcept;

    /// Reader thread: the oldest frame not taken yet.
    bool pop(Latents& latents) noexcept { return m_frames.pop(latents); }
    /// The reader thread: the newest frame.
    [[nodiscard]] const Latents& latest() noexcept { return m_latest.read(); }
    /// The model's latent dimensions.
    [[nodiscard]] std::uint32_t dims() const noexcept { return m_dims; }

    /// Main thread after prepare(), nothing else running: every frame of a mono recording at the
    /// prepared rate whose audio starts inside it, as a new stream (anira waits for each one).
    std::vector<Latents> run(std::span<const float> mono);

protected:
    /// `model`'s first streamed input carries the audio at `model_rate`, its first streamed output
    /// the latents: `dims` channels, one frame per `hop` model samples.
    LatentEncoder(anira::Context& context,
                  anira::ModelConfig model,
                  anira::ContractHandle contract,
                  double model_rate,
                  std::size_t hop,
                  std::uint32_t dims);

private:
    static constexpr std::size_t k_queue = 4096;

    anira_status pop_frames(std::size_t frames_due) noexcept;

    std::uint32_t m_dims;
    SpscQueue<Latents> m_frames;
    Latest<Latents> m_latest;
    std::array<float, k_max_latents> m_values{};  // anira writes the popped frame here
    anira::Tensor m_values_tensor{};
    std::int64_t m_popped = 0;  // frames popped since reset(), anira's priming included
};

}  // namespace anira::models::core
