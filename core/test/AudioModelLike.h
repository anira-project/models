// SPDX-License-Identifier: Apache-2.0

// The shape every audio model's wrapper has (plans/09-wrappers.md): anira's lifecycle and calls,
// a real-time process() over a tensor of the host's audio, the clock.

#pragma once

#include <anira/abi/status.h>

#include <anira/anira.hpp>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace models_test {

template <class T>
concept AudioModelLike = requires(T& model, const T& view, const anira::Tensor& audio) {
    { model.prepare(48000.0, std::size_t{512}) } -> std::same_as<anira::PlanReport>;
    { model.process(audio) } -> std::same_as<anira_status>;
    { model.reset() } -> std::same_as<void>;
    { view.get_latency() } -> std::same_as<std::size_t>;
    { view.position() } -> std::same_as<std::int64_t>;
    { view.plan_report() } -> std::same_as<anira::PlanReport>;
    { model.set_plan(std::uint32_t{0}) } -> std::same_as<anira_status>;
    { view.rt_error() } -> std::same_as<anira_status>;
};

}  // namespace models_test
