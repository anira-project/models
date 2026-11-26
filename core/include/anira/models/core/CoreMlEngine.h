// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.hpp>
#include <memory>

namespace anira::models::core {

/// The id model entries name this engine by.
inline constexpr const char* k_coreml_engine_id = "org.anira-project.models.coreml";

/// Core ML ML programs (`.mlpackage`) through Apple's Core ML framework, until anira has a Core ML
/// engine (its ExecuTorch carries no Core ML delegate either). Temporary: it lets the models'
/// tests and the examples run the Core ML exports, to validate them. Two providers: "cpu" (the CPU
/// alone) and "coreml" (every compute unit Core ML picks: CPU, GPU, Neural Engine). A Core ML input
/// of another data type than its spec (int64 lengths into int32) is converted on the way in, an
/// output on the way out. Not real-time safe: Objective-C allocations on every call, which anira
/// makes on its inference threads, not the audio thread.
///
/// One process-wide engine (handlers of one model share its load); null when this build has no
/// Core ML engine (ANIRA_MODELS_COREML_ENGINE off, or not macOS / iOS).
std::shared_ptr<anira::Engine> coreml_engine();

}  // namespace anira::models::core
