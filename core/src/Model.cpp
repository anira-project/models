// SPDX-License-Identifier: Apache-2.0

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/status.h>
#include <anira/models/core/CoreMlEngine.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cstdint>
#include <nlohmann/json.hpp>
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

// The description without the entries of core::coreml_engine() where it cannot run them: a build
// without it, or weights that are bytes (a .mlpackage is a directory).
std::string runnable(std::string_view model_json, const ModelSource& weights) {
    if (model_json.find(k_coreml_engine_id) == std::string_view::npos ||
        (coreml_engine() != nullptr && weights.is_directory())) {
        return std::string(model_json);
    }
    auto description = nlohmann::json::parse(model_json);
    auto& models = description.at("models");
    for (auto entry = models.begin(); entry != models.end();) {
        entry = entry->value("engine", "") == k_coreml_engine_id ? models.erase(entry) : entry + 1;
    }
    return description.dump();
}

}  // namespace

anira::Pipeline model_pipeline(const anira::ModelConfig& model) {
    anira::Pipeline pipeline;
    if (auto engine = coreml_engine()) { pipeline.add_engine(engine); }
    pipeline.add_inference(model);
    return pipeline;
}

anira::ModelConfig load_model(std::string_view model_json, const ModelSource& weights) {
    try {
        const std::string text = runnable(model_json, weights);
        anira::ModelConfig config =
            weights.is_directory() ? anira::ModelConfig::from_json(text, weights.root().string())
                                   : anira::ModelConfig::from_json(text);
        if (!weights.is_directory()) { attach_bytes(config, weights); }
        return config;
    } catch (const anira::Error& error) { throw Error(error.what()); }
}

}  // namespace anira::models::core
