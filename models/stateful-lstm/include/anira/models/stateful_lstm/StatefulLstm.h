// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/stateful_lstm/Descriptions.h>

#include <anira/anira.hpp>

namespace anira::models::stateful_lstm {

/// The stateful LSTM (vackva/stateful-lstm): an LSTM guitar effect, 2048 samples per inference, its
/// state carried from inference to inference by anira (reset() clears it). anira's
/// InferenceHandler with the model built in (core::AudioEffect): prepare, process, process_wait,
/// reset, get_latency; at the host's rate, mono (the downmix in, every channel out).
class StatefulLstm : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    StatefulLstm(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(context,
                            core::load_model(k_stateful_lstm_model_json, weights),
                            anira::ContractHandle::from_json(k_stateful_lstm_contract_json),
                            nullptr) {}
};

}  // namespace anira::models::stateful_lstm
