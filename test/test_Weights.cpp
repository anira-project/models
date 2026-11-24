// SPDX-License-Identifier: Apache-2.0

// Every description against its weights: each *.model.json prepares on anira's Hard path on every
// engine it names and every provider of it that this anira build has (CPU, Core ML, ...), from the
// weights of the HF repository its folder's catalog names. An entry pinned to a provider prepares
// on that provider only. anira checks at prepare that the weights fit the description (names,
// dtypes, ranks, extents). A description for the Async contract prepares through
// core::AsyncHandler until anira's own (M3). Opt-in: ANIRA_MODELS_TEST_WEIGHTS is a directory
// with one folder per HF repository (`<root>/<repo name>/onnx/...`); without it the test skips.

#include <anira/abi/config.h>
#include <anira/abi/context.h>
#include <anira/abi/enums.h>
#include <anira/abi/handler.h>
#include <anira/abi/status.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core = anira::models::core;
namespace fs = std::filesystem;

namespace {

constexpr double k_rate = 44100.0;
constexpr int64_t k_default_block = 512;

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

bool ends_with(const fs::path& path, const std::string& suffix) {
    const std::string name = path.filename().string();
    return name.size() > suffix.size() && name.ends_with(suffix);
}

constexpr std::array<std::pair<std::string_view, anira_engine>, 5> k_engines{
    {{"onnxruntime", ANIRA_ENGINE_ONNXRUNTIME},
     {"executorch", ANIRA_ENGINE_EXECUTORCH},
     {"litert", ANIRA_ENGINE_LITERT},
     {"libtorch", ANIRA_ENGINE_LIBTORCH},
     {"tflite", ANIRA_ENGINE_TFLITE}}};

constexpr std::array<std::pair<std::string_view, anira_provider>, 7> k_providers{
    {{"cpu", ANIRA_PROVIDER_CPU},
     {"cuda", ANIRA_PROVIDER_CUDA},
     {"webgpu", ANIRA_PROVIDER_WEBGPU},
     {"vulkan", ANIRA_PROVIDER_VULKAN},
     {"directml", ANIRA_PROVIDER_DIRECTML},
     {"coreml", ANIRA_PROVIDER_COREML},
     {"xnnpack", ANIRA_PROVIDER_XNNPACK}}};

uint32_t provider_named(std::string_view name) {
    for (const auto& [provider_name, provider] : k_providers) {
        if (provider_name == name) { return provider; }
    }
    return ANIRA_PROVIDER_NONE;
}

anira_engine engine_named(std::string_view name) {
    for (const auto& [engine_name, engine] : k_engines) {
        if (engine_name == name) { return engine; }
    }
    return ANIRA_ENGINE_NONE;
}

// The HF repository (its name, without the organisation) of each weight the folder's catalogs list.
std::map<std::string, std::string> repo_of_weights(const fs::path& folder) {
    std::map<std::string, std::string> repos;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!ends_with(entry.path(), ".catalog.json")) { continue; }
        const auto catalog = nlohmann::json::parse(read_text(entry.path()));
        const auto repo = catalog.at("repo").get<std::string>();
        for (const auto& file : catalog.at("files")) {
            repos[file.at("path").get<std::string>()] = repo.substr(repo.find('/') + 1);
        }
    }
    return repos;
}

// The host block: the window of the anchor, else of the first streamed input that has one.
int64_t host_block(const nlohmann::json& description) {
    const std::string anchor = description.value("anchor", "");
    for (const char* side : {"outputs", "inputs"}) {
        for (const auto& tensor : description.at(side)) {
            if (!tensor.contains("window")) { continue; }
            if (anchor.empty() ? (side == std::string("inputs") && tensor.at("role") == "streamed")
                               : tensor.at("name") == anchor) {
                return tensor.at("window").at("max").get<int64_t>();
            }
        }
    }
    return k_default_block;
}

// Prepares the description on the engine and provider; the error message, empty on success.
std::string prepare(anira::Context& context,
                    const anira::ModelConfig& model,
                    anira::ContractHandle& contract,
                    anira_engine engine,
                    anira_provider provider) {
    anira_error err{};
    anira_pipeline* pipeline = nullptr;
    anira_pipeline_create(&pipeline, &err);
    const std::array<const anira_model_config*, 1> variants{model.native()};
    const std::array<anira_backend_id, 1> candidates{
        anira_backend_id{.struct_size = sizeof(anira_backend_id),
                         .engine = static_cast<uint32_t>(engine),
                         .provider = static_cast<uint32_t>(provider)}};
    anira_pipeline_add_inference(pipeline, variants.data(), 1, candidates.data(), 1, &err);
    anira_handler* handler = nullptr;
    anira_status status = anira_handler_create(context.native(), pipeline, &handler, &err);
    anira_pipeline_destroy(pipeline);
    if (status == ANIRA_OK) { status = anira_handler_prepare(handler, contract.native(), &err); }
    anira_handler_destroy(handler);
    return status == ANIRA_OK ? std::string{} : std::string(err.message);
}

}  // namespace

TEST(Weights, EveryDescriptionPreparesOnEveryEngineBuiltIn) {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS is not set"; }

    anira::Context context{anira::ContextConfig{}};
    std::map<uint32_t, std::set<uint32_t>> providers;  // per engine of this build
    for (const anira::BackendId& backend : context.capabilities().backends()) {
        if (backend.engine != ANIRA_ENGINE_CUSTOM && backend.provider != ANIRA_PROVIDER_CUSTOM) {
            providers[backend.engine].insert(backend.provider);
        }
    }

    std::vector<fs::path> descriptions;
    for (const char* source : {ANIRA_MODELS_SOURCE_DIR, ANIRA_TEST_MODELS_SOURCE_DIR}) {
        // <root>/<model>/<file>: a model folder's descriptions sit at its top.
        for (const auto& folder : fs::directory_iterator(source)) {
            if (!folder.is_directory()) { continue; }
            for (const auto& entry : fs::directory_iterator(folder.path())) {
                descriptions.push_back(entry.path());
            }
        }
    }

    size_t prepared = 0;
    for (const fs::path& path : descriptions) {
        if (!ends_with(path, ".model.json")) { continue; }
        const std::string stem =
            path.string().substr(0, path.string().size() - std::string_view(".model.json").size());
        // Its own contract, or the folder's one contract that its models share.
        fs::path contract_path = stem + ".contract.json";
        if (!fs::exists(contract_path)) {
            std::vector<fs::path> shared;
            for (const auto& entry : fs::directory_iterator(path.parent_path())) {
                if (ends_with(entry.path(), ".contract.json")) { shared.push_back(entry.path()); }
            }
            if (shared.size() != 1) { continue; }
            contract_path = shared.front();
        }
        anira::ContractHandle contract = anira::ContractHandle::from_json(read_text(contract_path));
        const std::string text = read_text(path);
        const auto description = nlohmann::json::parse(text);
        if (anira_contract_get_kind(contract.native()) == ANIRA_CONTRACT_ASYNC) {
            // Until anira's Async contract (M3): core::AsyncHandler, on every engine of this build.
            const std::map<std::string, std::string> repos = repo_of_weights(path.parent_path());
            for (const auto& model_entry : description.at("models")) {
                const anira_engine engine =
                    engine_named(model_entry.at("engine").get<std::string>());
                if (!providers.contains(static_cast<uint32_t>(engine))) { continue; }
                const auto weight = model_entry.at("path").get<std::string>();
                const fs::path weights_dir = fs::path(root) / repos.at(weight);
                if (!fs::exists(weights_dir / weight)) {
                    ADD_FAILURE() << path << ": " << (weights_dir / weight) << " is missing";
                    break;
                }
                try {
                    core::AsyncHandler handler(context,
                                               text,
                                               core::ModelSource::directory(weights_dir));
                    handler.prepare(anira::Async{});
                    ++prepared;
                } catch (const std::exception& error) {
                    ADD_FAILURE() << path.filename() << ": " << error.what();
                }
                break;
            }
            continue;
        }
        const int64_t block = host_block(description);
        contract.hard_host_blocks(static_cast<double>(block), static_cast<double>(block), k_rate);
        const std::map<std::string, std::string> repos = repo_of_weights(path.parent_path());
        for (const auto& model_entry : description.at("models")) {
            const anira_engine engine = engine_named(model_entry.at("engine").get<std::string>());
            if (!providers.contains(static_cast<uint32_t>(engine))) { continue; }
            const auto weight = model_entry.at("path").get<std::string>();
            const fs::path weights_dir = fs::path(root) / repos.at(weight);
            if (!fs::exists(weights_dir / weight)) {
                ADD_FAILURE() << path << ": " << (weights_dir / weight) << " is missing";
                continue;
            }
            const anira::ModelConfig model =
                core::load_model(text, core::ModelSource::directory(weights_dir));
            const std::string pinned = model_entry.value("provider", "");
            for (const uint32_t provider : providers.at(static_cast<uint32_t>(engine))) {
                if (!pinned.empty() && provider_named(pinned) != provider) { continue; }
                EXPECT_EQ(prepare(context,
                                  model,
                                  contract,
                                  engine,
                                  static_cast<anira_provider>(provider)),
                          "")
                    << path.filename() << " on " << model_entry.at("engine") << ", provider "
                    << provider;
                ++prepared;
            }
        }
    }
    EXPECT_GT(prepared, 0U);
}
