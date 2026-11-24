// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/LatentDecoder.h>
#include <anira/models/core/LatentEncoder.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/rave_djembe/Descriptions.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

/// RAVE djembe, a RAVE model (weights CC BY-NC 4.0), trained at 44.1 kHz (other host rates are
/// resampled), mono: 2 latents (PCA-truncated from 16), one frame per 128 samples at 44.1 kHz; the
/// decoder takes them as a stream. Every graph carries its state through anira (reset() clears it);
/// the noise inputs are zeros (the deterministic decode).
///
///   - RaveDjembe: the forward pass (encoder and decoder in one graph), a timbre transfer of the
///   input,
///     in place (core::AudioEffect).
///   - RaveDjembeEncoder: audio in, latent frames out (core::LatentEncoder).
///   - RaveDjembeDecoder: latents in, audio out (core::LatentDecoder), e.g. from a hand tracker or
///   the
///     encoder of another model.
namespace anira::models::rave_djembe {

inline constexpr double k_model_rate = 44100.0;
inline constexpr std::uint32_t k_latents = 2;
inline constexpr std::size_t k_hop = 128;  ///< model samples per latent frame

class RaveDjembe : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    RaveDjembe(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(context,
                            core::load_model(k_rave_djembe_forward_model_json, weights),
                            anira::ContractHandle::from_json(k_rave_djembe_forward_contract_json),
                            k_model_rate) {
        keep_static_input(2, {1, 14, 8}, std::vector<float>(112, 0.0F));  // fill_in
    }
};

class RaveDjembeEncoder : public core::LatentEncoder {
public:
    RaveDjembeEncoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentEncoder(context,
                              core::load_model(k_rave_djembe_encoder_model_json, weights),
                              anira::ContractHandle::from_json(k_rave_djembe_encoder_contract_json),
                              k_model_rate,
                              k_hop,
                              k_latents) {}
};

class RaveDjembeDecoder : public core::LatentDecoder {
public:
    RaveDjembeDecoder(anira::Context& context, const core::ModelSource& weights)
        : core::LatentDecoder(context,
                              core::load_model(k_rave_djembe_decoder_model_json, weights),
                              anira::ContractHandle::from_json(k_rave_djembe_decoder_contract_json),
                              {.m_model_rate = k_model_rate,
                               .m_dims = k_latents,
                               .m_latents_per_sample = 1.0 / 128.0,
                               .m_latent_slot = 0,
                               .m_audio_slot = 0}) {
        keep_static_input(2, {1, 14, 8}, std::vector<float>(112, 0.0F));  // fill_in
    }
};

}  // namespace anira::models::rave_djembe
