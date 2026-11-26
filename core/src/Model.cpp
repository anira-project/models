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
#include <filesystem>
#include <fstream>
#include <ios>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <string_view>
#include <utility>
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

// A bytes source's Core ML package (the files under `package`/) written into a directory of
// their own under the temporary directory, once per process and package: core::coreml_engine()
// opens a package by path. Left there for the system to clear (packages of compiled-in weights
// are small). Empty when the source has none of its files.
std::filesystem::path written_package(const ModelSource& weights, const std::string& package) {
    static std::mutex mutex;
    static std::map<std::pair<std::string, const std::byte*>, std::filesystem::path> written;
    const std::string prefix = package + "/";
    std::vector<const ModelSource::File*> files;
    for (const ModelSource::File& file : weights.files()) {
        if (file.m_path.starts_with(prefix)) { files.push_back(&file); }
    }
    if (files.empty()) { return {}; }
    const std::scoped_lock lock(mutex);
    const auto key = std::make_pair(package, files.front()->m_bytes.data());
    if (const auto found = written.find(key); found != written.end()) { return found->second; }
    // A directory per process (a random name) and package: processes never share one.
    static const std::string process = std::to_string(std::random_device{}());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "anira-models" /
                                      process / std::to_string(written.size()) / package;
    std::filesystem::remove_all(dir);
    for (const ModelSource::File* file : files) {
        const std::filesystem::path path = dir / file->m_path.substr(prefix.size());
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(file->m_bytes.data()),
                  static_cast<std::streamsize>(file->m_bytes.size()));
        if (!out) { throw Error("could not write the Core ML package to " + dir.string()); }
    }
    written.emplace(key, dir);
    return dir;
}

// The description as this build runs it: core::coreml_engine()'s entries dropped without the
// engine; from a bytes source, pointed at the package written to disk (written_package), or
// dropped when the source has none of its files.
std::string runnable(std::string_view model_json, const ModelSource& weights) {
    if (model_json.find(k_coreml_engine_id) == std::string_view::npos ||
        (coreml_engine() != nullptr && weights.is_directory())) {
        return std::string(model_json);
    }
    auto description = nlohmann::json::parse(model_json);
    auto& models = description.at("models");
    for (auto entry = models.begin(); entry != models.end();) {
        if (entry->value("engine", "") != k_coreml_engine_id) {
            ++entry;
            continue;
        }
        const std::filesystem::path dir =
            coreml_engine() != nullptr
                ? written_package(weights, entry->at("path").get<std::string>())
                : std::filesystem::path();
        if (dir.empty()) {
            entry = models.erase(entry);
        } else {
            (*entry)["path"] = dir.string();
            ++entry;
        }
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
