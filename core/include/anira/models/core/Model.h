// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/ModelSource.h>

#include <anira/anira.hpp>
#include <string_view>

namespace anira::models::core {

/// A model from its anira description (the text of a `*.model.json`, kept in this repository and
/// compiled into the library) and its weights (from the source the application passes in). The
/// description's model paths are relative to the weights' root: from a directory source anira opens
/// them there; from a bytes source every entry gets the source's bytes of that path, borrowed. An
/// entry for an engine this build does not carry may go without weights. anira checks the weights
/// against the description when the handler is prepared (data types, ranks, extents) and names the
/// tensor that does not match. Throws Error for missing weights or a description anira refuses.
anira::ModelConfig load_model(std::string_view model_json, const ModelSource& weights);

}  // namespace anira::models::core
