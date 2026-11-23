// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace anira::models::core {

namespace {

bool engine_enabled(anira_engine engine) {
    uint32_t count = 0;
    if (anira_enabled_engines(sizeof(anira_backend_id), &count, nullptr) != ANIRA_OK) {
        return false;
    }
    std::vector<anira_backend_id> ids(count,
                                      anira_backend_id{.struct_size = sizeof(anira_backend_id)});
    if (anira_enabled_engines(sizeof(anira_backend_id), &count, ids.data()) != ANIRA_OK) {
        return false;
    }
    return std::ranges::any_of(ids, [&](const anira_backend_id& id) {
        return id.engine == static_cast<uint32_t>(engine);
    });
}

// A bytes source: every entry gets the source's bytes of its path, borrowed. An entry of an engine
// this build does not carry may go without (an embedding keeps only what it can run).
void attach_bytes(anira::ModelConfig& config, const ModelSource& weights) {
    for (uint32_t i = 0; i < config.model_count(); ++i) {
        const char* path = anira_model_config_model_path(config.native(), i);
        if (path == nullptr || *path == '\0') { continue; }
        if (const auto bytes = weights.find_bytes(path)) {
            config.set_model_bytes(i, *bytes, ANIRA_BYTES_BORROW);
        } else if (engine_enabled(config.model_engine(i).kind)) {
            throw Error(std::string("no weights for '") + path + "', the model of entry " +
                        std::to_string(i));
        }
    }
}

}  // namespace

anira::ModelConfig load_model(std::string_view model_json, const ModelSource& weights) {
    try {
        anira::ModelConfig config =
            weights.is_directory()
                ? anira::ModelConfig::from_json(model_json, weights.root().string())
                : anira::ModelConfig::from_json(model_json);
        if (!weights.is_directory()) { attach_bytes(config, weights); }
        return config;
    } catch (const anira::Error& error) { throw Error(error.what()); }
}

}  // namespace anira::models::core
