// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/stateful_lstm/Descriptions.h>

#include <anira/anira.hpp>

namespace anira::models::stateful_lstm {

/// The stateful LSTM (vackva/stateful-lstm): an LSTM guitar effect, 2048 samples per call, its
/// state carried from call to call by anira (reset() clears it). Trained at 44.1 kHz (other host
/// rates are resampled); mono, the result on every channel. prepare, process in place, reset,
/// get_latency: core::AudioEffect.
class StatefulLstm : public core::AudioEffect {
public:
    static constexpr double k_model_rate = 44100.0;

    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    StatefulLstm(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(context,
                            core::load_model(k_stateful_lstm_model_json, weights),
                            anira::ContractHandle::from_json(k_stateful_lstm_contract_json),
                            k_model_rate) {}
};

}  // namespace anira::models::stateful_lstm
