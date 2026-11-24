// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/guitar_lstm/Descriptions.h>
#include <anira/models/guitar_lstm/GuitarLstmProcessor.h>

#include <anira/anira.hpp>
#include <memory>

namespace anira::models::guitar_lstm {

/// GuitarLSTM (faressc/GuitarLSTM): a guitar amp, a convolution over 150 samples into an LSTM,
/// 256 samples per inference, its windows built by GuitarLstmProcessor as in anira's example.
/// anira's InferenceHandler with the model built in (core::AudioEffect): prepare, process,
/// process_wait, reset, get_latency; at the host's rate, mono (the downmix in, every channel out).
class GuitarLstm : public core::AudioEffect {
public:
    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    GuitarLstm(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(context,
                            core::load_model(k_guitar_lstm_model_json, weights),
                            anira::ContractHandle::from_json(k_guitar_lstm_contract_json),
                            std::make_shared<GuitarLstmProcessor>()) {}
};

}  // namespace anira::models::guitar_lstm
