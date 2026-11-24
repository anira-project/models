// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/steerable_nafx/Descriptions.h>

#include <anira/anira.hpp>
#include <cstdint>
#include <string_view>

namespace anira::models::steerable_nafx {

/// Steerable NAFX (faressc/steerable-nafx, "Steerable discovery of neural audio effects"): a
/// randomly initialised dilated convolutional network as an effect, 2048 samples per inference, in
/// three sizes. anira's InferenceHandler with the model built in (core::AudioEffect): prepare,
/// process, process_wait, reset, get_latency; at the host's rate, mono (the downmix in, every
/// channel out).
class SteerableNafx : public core::AudioEffect {
public:
    /// The model's size: its receptive field (the full model's 13332 samples, or 2 or 3 of its
    /// dilated blocks).
    enum class Size : std::uint8_t { Full, ThreeBlocks, TwoBlocks };

    /// The weights: the model's Hugging Face repository (a directory or embedded bytes).
    SteerableNafx(anira::Context& context, const core::ModelSource& weights, Size size = Size::Full)
        : core::AudioEffect(context,
                            core::load_model(model_json(size), weights),
                            anira::ContractHandle::from_json(contract_json(size)),
                            nullptr) {}

private:
    static std::string_view model_json(Size size) noexcept {
        switch (size) {
            case Size::ThreeBlocks: return k_steerable_nafx_3_blocks_model_json;
            case Size::TwoBlocks: return k_steerable_nafx_2_blocks_model_json;
            default: return k_steerable_nafx_model_json;
        }
    }
    static std::string_view contract_json(Size size) noexcept {
        switch (size) {
            case Size::ThreeBlocks: return k_steerable_nafx_3_blocks_contract_json;
            case Size::TwoBlocks: return k_steerable_nafx_2_blocks_contract_json;
            default: return k_steerable_nafx_contract_json;
        }
    }
};

}  // namespace anira::models::steerable_nafx
