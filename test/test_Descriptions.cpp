// SPDX-License-Identifier: Apache-2.0

// Every model folder at once: each *.model.json loads through anira's configuration layer, each
// *.contract.json is a contract, and every weight a description names is listed in a catalog in the
// same folder, so an application that fetches by the catalog has what the description needs.

#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>
#include <vector>

namespace core = anira::models::core;
namespace fs = std::filesystem;

namespace {

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

bool ends_with(const fs::path& path, const std::string& suffix) {
    const std::string name = path.filename().string();
    return name.size() > suffix.size() && name.ends_with(suffix);
}

std::vector<fs::path> files_ending(const std::string& suffix) {
    std::vector<fs::path> files;
    for (const char* root : {ANIRA_MODELS_SOURCE_DIR, ANIRA_TEST_MODELS_SOURCE_DIR}) {
        // <root>/<model>/<file>: a model folder's descriptions sit at its top.
        for (const auto& folder : fs::directory_iterator(root)) {
            if (!folder.is_directory()) { continue; }
            for (const auto& entry : fs::directory_iterator(folder.path())) {
                if (entry.is_regular_file() && ends_with(entry.path(), suffix)) {
                    files.push_back(entry.path());
                }
            }
        }
    }
    return files;
}

// The paths every catalog in the folder lists.
std::set<std::string> catalogued(const fs::path& folder) {
    std::set<std::string> paths;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!ends_with(entry.path(), ".catalog.json")) { continue; }
        const auto catalog = nlohmann::json::parse(read_text(entry.path()));
        for (const auto& file : catalog.at("files")) {
            paths.insert(file.at("path").get<std::string>());
        }
    }
    return paths;
}

}  // namespace

TEST(Descriptions, EveryModelLoadsAndItsWeightsAreCatalogued) {
    const std::vector<fs::path> models = files_ending(".model.json");
    ASSERT_FALSE(models.empty());
    for (const fs::path& path : models) {
        const std::string text = read_text(path);
        // No weights are opened before prepare: any directory serves as their root.
        EXPECT_NO_THROW((void)core::load_model(text, core::ModelSource::directory("."))) << path;
        const std::set<std::string> listed = catalogued(path.parent_path());
        // Kept: a range-for over parse(text).at(...) would iterate a destroyed temporary.
        const auto description = nlohmann::json::parse(text);
        for (const auto& entry : description.at("models")) {
            const auto weight = entry.at("path").get<std::string>();
            // A file, or a folder whose files are listed (a Core ML .mlpackage).
            const auto inside = listed.lower_bound(weight + "/");
            const bool folder = inside != listed.end() && inside->starts_with(weight + "/");
            EXPECT_TRUE(listed.contains(weight) || folder)
                << path << " names " << weight << ", which no catalog in its folder lists";
        }
    }
}

TEST(Descriptions, EveryContractLoads) {
    const std::vector<fs::path> contracts = files_ending(".contract.json");
    ASSERT_FALSE(contracts.empty());
    for (const fs::path& path : contracts) {
        EXPECT_NO_THROW((void)anira::ContractHandle::from_json(read_text(path))) << path;
    }
}
