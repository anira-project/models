// SPDX-License-Identifier: Apache-2.0

// The gain model (core/test/data/gain) on anira's Hard path: a run that shows it loaded from what a
// source held, and a prepare that reports what anira refused.

#pragma once

#include <anira/abi/context.h>
#include <anira/abi/handler.h>
#include <anira/abi/tensor.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace models_test {

inline constexpr size_t k_block = 512;
inline constexpr double k_rate = 48000.0;
inline constexpr float k_gain = 0.5F;

inline std::filesystem::path gain_dir() {
    return std::filesystem::path(MODELS_TEST_DATA_DIR) / "gain";
}

inline std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<char> chars{std::istreambuf_iterator<char>(stream),
                            std::istreambuf_iterator<char>()};
    std::vector<std::byte> bytes(chars.size());
    for (size_t i = 0; i < chars.size(); ++i) { bytes[i] = static_cast<std::byte>(chars[i]); }
    return bytes;
}

inline std::vector<float> ramp(size_t block_index) {
    std::vector<float> out(k_block);
    for (size_t i = 0; i < k_block; ++i) {
        out[i] = static_cast<float>(block_index * k_block + i) / 65536.0F;
    }
    return out;
}

// A handler on ONNX Runtime (CPU) for the model, prepared under the contract at the test geometry.
// Returns nullptr and the status and message when anira refuses.
inline anira_handler* prepare_handler(anira::Context& context,
                                      const anira::ModelConfig& model,
                                      anira::ContractHandle& contract,
                                      anira_status& status,
                                      std::string& message) {
    contract.hard_host_blocks(k_block, k_block, k_rate);
    anira_error err{};
    anira_pipeline* pipeline = nullptr;
    anira_pipeline_create(&pipeline, &err);
    const std::array<const anira_model_config*, 1> variants{model.native()};
    const std::array<anira_backend_id, 1> candidates{
        anira_backend_id{sizeof(anira_backend_id), ANIRA_ENGINE_ONNXRUNTIME, ANIRA_PROVIDER_CPU}};
    anira_pipeline_add_inference(pipeline, variants.data(), 1, candidates.data(), 1, &err);
    anira_handler* handler = nullptr;
    status = anira_handler_create(context.native(), pipeline, &handler, &err);
    anira_pipeline_destroy(pipeline);
    if (status == ANIRA_OK) { status = anira_handler_prepare(handler, contract.native(), &err); }
    if (status != ANIRA_OK) {
        message = err.message;
        anira_handler_destroy(handler);
        return nullptr;
    }
    return handler;
}

// Runs the model and checks that some later block comes out as an earlier input block times the
// gain: the model loaded and ran from what the source held.
inline void expect_runs_on_hard_path(anira::Context& context,
                                     const anira::ModelConfig& model,
                                     anira::ContractHandle& contract) {
    anira_status status = ANIRA_OK;
    std::string message;
    anira_handler* handler = prepare_handler(context, model, contract, status, message);
    ASSERT_NE(handler, nullptr) << message;

    float gain = k_gain;
    const std::array<int64_t, 1> gain_shape{1};
    anira_tensor gain_tensor{};
    anira_tensor_init_host(&gain_tensor, &gain, ANIRA_DTYPE_F32, 1, gain_shape.data());
    ASSERT_EQ(anira_handler_set_static_input(handler, 1, &gain_tensor), ANIRA_OK);

    constexpr size_t k_blocks = 8;
    std::vector<std::vector<float>> inputs;
    std::vector<float> output(k_block);
    const std::array<int64_t, 2> shape{1, static_cast<int64_t>(k_block)};
    for (size_t k = 0; k < k_blocks; ++k) {
        inputs.push_back(ramp(k + 1));
        const std::array<const float*, 1> in_planes{inputs.back().data()};
        const std::array<float*, 1> out_planes{output.data()};
        anira_tensor in{};
        anira_tensor out{};
        anira_tensor_init_host_planar(&in, in_planes.data(), 1, ANIRA_DTYPE_F32, 2, shape.data());
        anira_tensor_init_host_planar(&out, out_planes.data(), 1, ANIRA_DTYPE_F32, 2, shape.data());
        size_t delivered = 0;
        ASSERT_EQ(anira_handler_process_wait(handler, &in, 0, &out, 0, &delivered, 2000.0),
                  ANIRA_OK);
        EXPECT_EQ(delivered, k_block);
    }

    bool matched = false;
    for (size_t lag = 0; lag < k_blocks && !matched; ++lag) {
        const std::vector<float>& source = inputs[k_blocks - 1 - lag];
        matched = true;
        for (size_t i = 0; i < k_block; ++i) {
            if (std::abs(output[i] - (source[i] * k_gain)) > 1e-6F) {
                matched = false;
                break;
            }
        }
    }
    EXPECT_TRUE(matched) << "the last block is no earlier input block times the gain";
    anira_handler_destroy(handler);
}

}  // namespace models_test
