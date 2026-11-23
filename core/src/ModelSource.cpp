// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/Error.h>
#include <anira/models/core/ModelSource.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace anira::models::core {

namespace {

// One spelling per file: "a/./b/../c.json" and "a/c.json" name the same file.
std::string normal_path(std::string_view path) {
    return std::filesystem::path(path).lexically_normal().generic_string();
}

}  // namespace

ModelSource ModelSource::directory(std::filesystem::path root) {
    ModelSource source;
    source.m_root = std::move(root);
    source.m_is_directory = true;
    return source;
}

ModelSource ModelSource::bytes(std::vector<File> files) {
    ModelSource source;
    for (File& file : files) { file.m_path = normal_path(file.m_path); }
    source.m_files = std::move(files);
    return source;
}

bool ModelSource::is_directory() const noexcept {
    return m_is_directory;
}

const std::filesystem::path& ModelSource::root() const noexcept {
    return m_root;
}

std::string ModelSource::read_text(std::string_view path) const {
    if (m_is_directory) {
        const std::filesystem::path file = m_root / normal_path(path);
        std::ifstream stream(file, std::ios::binary);
        if (!stream) { throw Error("model source: cannot read " + file.string()); }
        return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    }
    const std::optional<std::span<const std::byte>> bytes = find_bytes(path);
    if (!bytes) { throw Error("model source: no file " + normal_path(path)); }
    return {reinterpret_cast<const char*>(bytes->data()), bytes->size()};
}

std::optional<std::span<const std::byte>> ModelSource::find_bytes(std::string_view path) const {
    const std::string wanted = normal_path(path);
    for (const File& file : m_files) {
        if (file.m_path == wanted) { return file.m_bytes; }
    }
    return std::nullopt;
}

}  // namespace anira::models::core
