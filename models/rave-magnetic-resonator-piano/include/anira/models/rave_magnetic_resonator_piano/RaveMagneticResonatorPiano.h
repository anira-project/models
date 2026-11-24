// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/LatentDecoder.h>
#include <anira/models/core/LatentEncoder.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/rave_magnetic_resonator_piano/Descriptions.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

/// RAVE magnetic resonator piano, a RAVE model (weights CC BY-NC 4.0), trained at 44.1 kHz (other
/// host rates are resampled), mono: 8 latents, one frame per 2048 samples at 44.1 kHz; the decoder
/// takes them as a Static input, before each call. Every graph carries its state through anira
/// (reset() clears it); the noise inputs are zeros (the deterministic decode).
///
///   - RaveMagneticResonatorPiano: the forward pass (encoder and decoder in one graph), a timbre
///   transfer of the input,
///     in place (core::AudioEffect).
///   - RaveMagneticResonatorPianoEncoder: audio in, latent frames out (core::LatentEncoder).
///   - RaveMagneticResonatorPianoDecoder: latents in, audio out (core::LatentDecoder), e.g. from a
///   hand tracker or the
///     encoder of another model.
namespace anira::models::rave_magnetic_resonator_piano {

inline constexpr double k_model_rate = 44100.0;
inline constexpr std::uint32_t k_latents = 8;
inline constexpr std::size_t k_hop = 2048;  ///< model samples per latent frame

class RaveMagneticResonatorPiano : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    RaveMagneticResonatorPiano(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(
              context,
              core::load_model(k_rave_magnetic_resonator_piano_forward_model_json, weights),
              anira::ContractHandle::from_json(
                  k_rave_magnetic_resonator_piano_forward_contract_json),
              k_model_rate) {
        keep_static_input(2, {1, 120, 1}, std::vector<float>(120, 0.0F));  // noise
    }
};

class RaveMagneticResonatorPianoEncoder : public core::LatentEncoder {
public:
    RaveMagneticResonatorPianoEncoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentEncoder(
              context,
              core::load_model(k_rave_magnetic_resonator_piano_encoder_model_json, weights),
              anira::ContractHandle::from_json(
                  k_rave_magnetic_resonator_piano_encoder_contract_json),
              k_model_rate,
              k_hop,
              k_latents) {}
};

class RaveMagneticResonatorPianoDecoder : public core::LatentDecoder {
public:
    RaveMagneticResonatorPianoDecoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentDecoder(
              context,
              core::load_model(k_rave_magnetic_resonator_piano_decoder_model_json, weights),
              anira::ContractHandle::from_json(
                  k_rave_magnetic_resonator_piano_decoder_contract_json),
              {.m_model_rate = k_model_rate,
               .m_dims = k_latents,
               .m_latents_per_sample = 0.0,
               .m_latent_slot = 0,
               .m_audio_slot = 0}) {
        keep_static_input(2, {1, 120, 1}, std::vector<float>(120, 0.0F));  // noise
    }
};

}  // namespace anira::models::rave_magnetic_resonator_piano
