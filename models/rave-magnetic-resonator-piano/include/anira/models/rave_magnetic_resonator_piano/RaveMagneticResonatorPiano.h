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

/// RAVE magnetic resonator piano, a RAVE model (weights CC BY-NC 4.0): 8 latents, one frame per
/// 2048 samples, which the decoder takes as a Static input (the last frame pushed). Three graphs,
/// each anira's InferenceHandler with the model built in, at the host's rate as anira's RAVE
/// example runs them; every graph carries its state through anira (reset() clears it), the noise
/// inputs are zeros (the deterministic decode).
///
///   - RaveMagneticResonatorPiano: the forward pass (encoder and decoder in one graph), a timbre
///   transfer of the input:
///     process, process_wait (core::AudioEffect).
///   - RaveMagneticResonatorPianoEncoder: push_data (audio), pop_data (latent frames)
///   (core::LatentEncoder).
///   - RaveMagneticResonatorPianoDecoder: push_data (latent frames), pop_data (audio)
///   (core::LatentDecoder): the
///     encoder's frames, changed or not, as in anira's example, or a hand tracker's.
namespace anira::models::rave_magnetic_resonator_piano {

inline constexpr std::uint32_t k_latents = 8;
inline constexpr std::size_t k_hop = 2048;  ///< samples per latent frame

class RaveMagneticResonatorPiano : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    RaveMagneticResonatorPiano(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(
              context,
              core::load_model(k_rave_magnetic_resonator_piano_forward_model_json, weights),
              anira::ContractHandle::from_json(
                  k_rave_magnetic_resonator_piano_forward_contract_json)) {
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
              {.m_dims = k_latents,
               .m_hop = k_hop,
               .m_static_latents = true,
               .m_latent_slot = 0,
               .m_audio_slot = 0}) {
        keep_static_input(2, {1, 120, 1}, std::vector<float>(120, 0.0F));  // noise
    }
};

}  // namespace anira::models::rave_magnetic_resonator_piano
