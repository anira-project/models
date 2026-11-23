// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/config.h>
#include <anira/abi/enums.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/handtracking/Descriptions.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <cstdint>
#include <string_view>

namespace core = anira::models::core;

namespace {

namespace ht = anira::models::handtracking;

// No weights are opened before prepare: any directory serves as their root here.
void expect_model(std::string_view json, uint32_t inputs, uint32_t outputs) {
    const anira::ModelConfig model = core::load_model(json, core::ModelSource::directory("."));
    EXPECT_EQ(model.model_count(), 6U);  // litert, tflite, onnxruntime, libtorch, executorch twice
    EXPECT_EQ(anira_model_config_num_inputs(model.native()), inputs);
    EXPECT_EQ(anira_model_config_num_outputs(model.native()), outputs);
}

}  // namespace

// The descriptions load through anira's configuration layer. That the weights fit them is
// checked by anira at prepare, which for the Async contract waits for anira M3.
TEST(HandtrackingModels, DescriptionsLoad) {
    expect_model(ht::k_hand_detector_model_json, 1, 2);
    expect_model(ht::k_hand_landmarks_detector_model_json, 1, 4);
    expect_model(ht::k_gesture_embedder_model_json, 3, 1);
    expect_model(ht::k_canned_gesture_classifier_model_json, 1, 1);
}

TEST(HandtrackingModels, TheContractIsAsync) {
    const anira::ContractHandle contract =
        anira::ContractHandle::from_json(ht::k_handtracking_contract_json);
    EXPECT_EQ(anira_contract_get_kind(contract.native()), ANIRA_CONTRACT_ASYNC);
}
