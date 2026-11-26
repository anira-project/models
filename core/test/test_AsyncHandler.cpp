// SPDX-License-Identifier: Apache-2.0

// core::AsyncHandler on the test networks written for the Async contract (Buffer roles): the same
// jobs give the same results as the model, in order; State carries from job to job; a full queue
// refuses a submit; on_late drop drops a job past its deadline and the next one is still in step.

#include <anira/anira.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core = anira::models::core;

namespace {

// test-models/simple-gain, mono, as an Async model: one 512-sample buffer and the gain per job.
constexpr std::string_view k_gain_json = R"({
  "models": [{"engine": "onnxruntime", "path": "simple-gain/onnx/simple_gain_network_mono.onnx"}],
  "state": "stateless",
  "inputs": [
    {"name": "audio_in", "dtype": "float32", "role": "buffer", "axes": [["batch", 1], ["channel", 1], ["any", 512]]},
    {"name": "gain", "dtype": "float32", "role": "static", "axes": [["any", 1]]}
  ],
  "outputs": [
    {"name": "audio_out", "dtype": "float32", "role": "buffer", "axes": [["batch", 1], ["channel", 1], ["any", 512]]},
    {"name": "gain_out", "dtype": "float32", "role": "static", "axes": [["any", 1]]}
  ]
})";

// test-models/stateful-accumulator, mono: a running sum as a State pair, 64 samples per job.
constexpr std::string_view k_accumulator_json = R"({
  "models": [{"engine": "onnxruntime", "path": "stateful-accumulator/onnx/stateful_accumulator_network_mono.onnx"}],
  "state": "stateful",
  "inputs": [
    {"name": "state_in", "dtype": "float32", "role": "state", "axes": [["batch", 1], ["channel", 1], ["any", 2]], "state_source": "state_out"},
    {"name": "data", "dtype": "float32", "role": "buffer", "axes": [["batch", 1], ["channel", 1], ["any", 64]]}
  ],
  "outputs": [
    {"name": "processed_data", "dtype": "float32", "role": "buffer", "axes": [["batch", 1], ["channel", 1], ["any", 64]]},
    {"name": "state_out", "dtype": "float32", "role": "state", "axes": [["batch", 1], ["channel", 1], ["any", 2]]}
  ]
})";

core::ModelSource weights() {
    return core::ModelSource::directory(MODELS_TEST_GAIN_DIR);
}

anira::Tensor host(float* data, std::initializer_list<std::int64_t> shape) {
    return anira::Tensor::from_host(data,
                                    ANIRA_DTYPE_F32,
                                    std::span<const std::int64_t>(shape.begin(), shape.size()));
}

anira::Tensor none() {
    const std::array<std::int64_t, 1> shape{0};
    anira::Tensor tensor{};
    anira_tensor_init_host(&tensor, nullptr, ANIRA_DTYPE_F32, 1, shape.data());
    return tensor;
}

std::vector<float> ramp(std::size_t n, float start) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = start + static_cast<float>(i); }
    return out;
}

}  // namespace

TEST(AsyncHandler, RewritesBufferTensorsAsWholeWindows) {
    anira::Context context{anira::ContextConfig{}};
    const core::AsyncHandler handler(context, k_gain_json, weights());
    const std::string& hard = handler.hard_description();
    EXPECT_NE(hard.find(R"("role":"streamed")"), std::string::npos) << hard;
    EXPECT_NE(hard.find(R"("max":512)"), std::string::npos) << hard;
    EXPECT_NE(hard.find(R"("anchor":"audio_in")"), std::string::npos) << hard;
}

TEST(AsyncHandler, JobsGiveTheModelsResultsInOrder) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_gain_json, weights());
    handler.prepare(anira::Async{});
    std::vector<float> out(512);
    float peak = 0.0f;
    handler.bind_output(0, host(out.data(), {1, 1, 512}));
    handler.bind_output(1, host(&peak, {1}));

    for (int k = 0; k < 5; ++k) {
        std::vector<float> in = ramp(512, static_cast<float>(k));
        float gain = 0.5f * static_cast<float>(k + 1);
        const std::array<anira::Tensor, 2> inputs{host(in.data(), {1, 1, 512}), host(&gain, {1})};
        core::Ticket ticket = handler.submit(inputs);
        ASSERT_EQ(ticket.wait(), core::TicketStatus::Met) << "job " << k << " " << ticket.error();
        for (std::size_t i = 0; i < 512; ++i) {
            ASSERT_FLOAT_EQ(out[i], in[i] * gain) << "job " << k << " sample " << i;
        }
        EXPECT_FLOAT_EQ(peak, in.back() * gain) << "job " << k;
    }
}

TEST(AsyncHandler, StateCarriesFromJobToJob) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_accumulator_json, weights());
    handler.prepare(anira::Async{});
    std::vector<float> out(64);
    handler.bind_output(0, host(out.data(), {1, 1, 64}));

    float sum = 0.0f;
    for (int k = 0; k < 4; ++k) {
        std::vector<float> in = ramp(64, static_cast<float>(k));
        const std::array<anira::Tensor, 2> inputs{none(), host(in.data(), {1, 1, 64})};
        core::Ticket ticket = handler.submit(inputs);
        ASSERT_EQ(ticket.wait(), core::TicketStatus::Met) << ticket.error();
        for (std::size_t i = 0; i < 64; ++i) { ASSERT_EQ(out[i], in[i] + sum) << "job " << k; }
        for (const float x : in) { sum += x; }
    }
}

TEST(AsyncHandler, AFullQueueRefusesASubmit) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_gain_json, weights());
    handler.prepare(anira::Async{.max_in_flight = 1});
    std::vector<float> out(512);
    float peak = 0.0f;
    handler.bind_output(0, host(out.data(), {1, 1, 512}));
    handler.bind_output(1, host(&peak, {1}));
    std::vector<float> in = ramp(512, 1.0f);
    float gain = 2.0f;
    const std::array<anira::Tensor, 2> inputs{host(in.data(), {1, 1, 512}), host(&gain, {1})};

    core::Ticket first = handler.submit(inputs);
    const core::Ticket refused = handler.submit(inputs);  // the first holds the only slot
    EXPECT_EQ(refused.status(), core::TicketStatus::Failed);
    EXPECT_EQ(refused.error(), ANIRA_ERROR_CAPACITY);
    EXPECT_EQ(first.wait(), core::TicketStatus::Met);
    first = core::Ticket{};  // releases the slot
    core::Ticket again = handler.submit(inputs);
    EXPECT_EQ(again.wait(), core::TicketStatus::Met);
}

// A ticket that saw its job's status frees the only slot: the next submit, right after, is taken
// (the worker publishes the status a moment before it lets go of the job).
TEST(AsyncHandler, ASubmitRightAfterAWaitFindsTheSlotFree) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_gain_json, weights());
    handler.prepare(anira::Async{.max_in_flight = 1});
    std::vector<float> out(512);
    float peak = 0.0f;
    handler.bind_output(0, host(out.data(), {1, 1, 512}));
    handler.bind_output(1, host(&peak, {1}));
    std::vector<float> in = ramp(512, 1.0f);
    float gain = 2.0f;
    const std::array<anira::Tensor, 2> inputs{host(in.data(), {1, 1, 512}), host(&gain, {1})};
    for (int k = 0; k < 2000; ++k) {
        core::Ticket ticket = handler.submit(inputs);
        ASSERT_EQ(ticket.wait(), core::TicketStatus::Met) << "job " << k << " " << ticket.error();
    }
}

TEST(AsyncHandler, OnLateDropDropsAJobPastItsDeadlineAndStaysInStep) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_accumulator_json, weights());
    handler.prepare(anira::Async{.on_late = ANIRA_LATE_DROP});
    std::vector<float> out(64);
    handler.bind_output(0, host(out.data(), {1, 1, 64}));

    std::vector<float> in = ramp(64, 1.0f);
    const std::array<anira::Tensor, 2> inputs{none(), host(in.data(), {1, 1, 64})};
    const auto past = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    core::Ticket late = handler.submit(inputs, {}, past);
    EXPECT_EQ(late.wait(), core::TicketStatus::Dropped);
    for (const float x : out) { EXPECT_EQ(x, 0.0f); }

    // The next job pairs with its own result, never the dropped one's.
    std::vector<float> next = ramp(64, 100.0f);
    const std::array<anira::Tensor, 2> next_inputs{none(), host(next.data(), {1, 1, 64})};
    const auto later = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    core::Ticket ticket = handler.submit(next_inputs, {}, later);
    ASSERT_EQ(ticket.wait(), core::TicketStatus::Met);
    // The dropped job may or may not have reached the state; the output is the next input plus it.
    float before = 0.0f;
    for (const float x : in) { before += x; }
    EXPECT_TRUE(out[0] == next[0] || out[0] == next[0] + before) << out[0];
    for (std::size_t i = 1; i < 64; ++i) { EXPECT_EQ(out[i] - next[i], out[0] - next[0]); }
}

TEST(AsyncHandler, PollRunsTheCallbacksInOrder) {
    anira::Context context{anira::ContextConfig{}};
    core::AsyncHandler handler(context, k_accumulator_json, weights());
    std::array<int, 4> cookies{1, 2, 3, 4};  // each job's user pointer
    std::vector<int> seen;
    handler.on_complete([&](void* user, core::TicketStatus status) {
        EXPECT_EQ(status, core::TicketStatus::Met);
        seen.push_back(*static_cast<int*>(user));
    });
    handler.prepare(anira::Async{.max_in_flight = 8});
    std::vector<float> out(64);
    handler.bind_output(0, host(out.data(), {1, 1, 64}));
    std::vector<float> in = ramp(64, 0.0f);
    const std::array<anira::Tensor, 2> inputs{none(), host(in.data(), {1, 1, 64})};
    std::vector<core::Ticket> tickets;
    tickets.reserve(cookies.size());
    for (int& cookie : cookies) { tickets.push_back(handler.submit(inputs, {}, {}, &cookie)); }
    for (core::Ticket& ticket : tickets) { ticket.wait(); }
    EXPECT_EQ(handler.poll(), 4U);
    EXPECT_EQ(seen, (std::vector<int>{1, 2, 3, 4}));
    EXPECT_EQ(handler.poll(), 0U);
}
