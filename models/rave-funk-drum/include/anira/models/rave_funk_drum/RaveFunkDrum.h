// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/LatentDecoder.h>
#include <anira/models/core/LatentEncoder.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/rave_funk_drum/Descriptions.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

/// RAVE funk drum, a RAVE model (weights CC BY-NC 4.0): 4 latents, one frame per 2048 samples,
/// which the decoder takes as a stream (anira's RAVE example). Three graphs, each anira's
/// InferenceHandler with the model built in, trained at 44.1 kHz (the host's audio is
/// resampled); every graph carries its state through anira (reset() clears it), the noise inputs
/// are zeros (the deterministic decode).
///
///   - RaveFunkDrum: the forward pass (encoder and decoder in one graph), a timbre transfer of the
///   input:
///     process, process_wait (core::AudioEffect).
///   - RaveFunkDrumEncoder: push_data (audio), pop_data (latent frames) (core::LatentEncoder).
///   - RaveFunkDrumDecoder: push_data (latent frames), pop_data (audio) (core::LatentDecoder): the
///     encoder's frames, changed or not, as in anira's example, or a hand tracker's.
namespace anira::models::rave_funk_drum {

/// The rate the model was trained at; the host's audio is resampled to it.
inline constexpr double k_model_rate = 44100.0;

inline constexpr std::uint32_t k_latents = 4;
inline constexpr std::size_t k_hop = 2048;  ///< samples per latent frame

class RaveFunkDrum : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    RaveFunkDrum(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(
              context,
              core::load_model(k_rave_funk_drum_forward_model_json, weights),
              anira::ContractHandle::from_json(k_rave_funk_drum_forward_contract_json),
              k_model_rate) {
        keep_static_input(2, {1, 12, 1}, std::vector<float>(12, 0.0F));        // noise
        keep_static_input(3, {1, 2, 16, 64}, std::vector<float>(2048, 0.0F));  // synth_noise
    }
};

class RaveFunkDrumEncoder : public core::LatentEncoder {
public:
    RaveFunkDrumEncoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentEncoder(
              context,
              core::load_model(k_rave_funk_drum_encoder_model_json, weights),
              anira::ContractHandle::from_json(k_rave_funk_drum_encoder_contract_json),
              k_model_rate,
              k_hop,
              k_latents) {}
};

class RaveFunkDrumDecoder : public core::LatentDecoder {
public:
    RaveFunkDrumDecoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentDecoder(
              context,
              core::load_model(k_rave_funk_drum_decoder_model_json, weights),
              anira::ContractHandle::from_json(k_rave_funk_drum_decoder_contract_json),
              {.m_model_rate = k_model_rate,
               .m_dims = k_latents,
               .m_hop = k_hop,
               .m_static_latents = false,
               .m_latent_slot = 0,
               .m_audio_slot = 0}) {
        keep_static_input(2, {1, 12, 1}, std::vector<float>(12, 0.0F));        // noise
        keep_static_input(3, {1, 2, 16, 64}, std::vector<float>(2048, 0.0F));  // synth_noise
    }
};

}  // namespace anira::models::rave_funk_drum
