// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/rave_funk_drum/Descriptions.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace anira::models::rave_funk_drum {

/// RAVE funk drum, a RAVE model (weights CC BY-NC 4.0): the streaming forward pass (the encoder and
/// the decoder in one graph, their state carried by anira; reset() clears it) as a timbre transfer
/// of the input. Trained at 44.1 kHz (other host rates are resampled); mono, the result on every
/// channel. The noise inputs are zeros: the deterministic decode. prepare, process in place,
/// reset, get_latency: core::AudioEffect.
class RaveFunkDrum : public core::AudioEffect {
public:
    static constexpr double k_model_rate = 44100.0;

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

}  // namespace anira::models::rave_funk_drum
