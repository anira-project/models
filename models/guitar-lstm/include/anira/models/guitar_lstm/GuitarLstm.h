// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/guitar_lstm/Descriptions.h>

#include <anira/anira.hpp>

namespace anira::models::guitar_lstm {

/// GuitarLSTM (faressc/GuitarLSTM): a guitar amp, a convolution over 150 samples into an LSTM,
/// 256 samples per call. Trained at 44.1 kHz (other host rates are resampled); mono, the result on
/// every channel. prepare, process in place, reset, get_latency: core::AudioEffect.
class GuitarLstm : public core::AudioEffect {
public:
    static constexpr double k_model_rate = 44100.0;

    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    GuitarLstm(anira::Context& context, const core::ModelSource& weights)
        : core::AudioEffect(context,
                            core::load_model(k_guitar_lstm_model_json, weights),
                            anira::ContractHandle::from_json(k_guitar_lstm_contract_json),
                            k_model_rate) {}
};

}  // namespace anira::models::guitar_lstm
