// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.hpp>
#include <cstdint>
#include <memory>

namespace anira::models::core {

/// The id model entries name this engine by.
inline constexpr const char* k_onnx_engine_id = "org.anira-project.models.onnxruntime";

/// ONNX Runtime on the CPU with a thread count of its own: anira's ONNX Runtime adapter runs every
/// session on one thread (SetIntraOpNumThreads(1), right for a real-time model next to an audio
/// thread), which leaves a large job model (Parakeet's encoder, a diarization step) several times
/// slower than the machine allows. This engine is the same binding (by name, over anira's memory,
/// no copy) with `threads` intra-op threads per session (0: ONNX Runtime's default, one per
/// physical core) and spinning off, so idle workers sleep instead of burning cores the audio
/// thread may need. AsyncHandler runs every CPU ONNX Runtime entry on it.
///
/// Until anira takes a thread count per model (the request in example-models.md); then this goes.
/// Null when this build of anira carries no ONNX Runtime.
std::shared_ptr<anira::Engine> make_onnx_engine(std::uint32_t threads);

}  // namespace anira::models::core
