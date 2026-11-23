// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace anira::models::core {

// Where a library gets its model files from: the application passes it in, so it decides
// whether the files sit in a bundle, a user folder, a download cache, or inside the binary.
// Files are named by their path relative to the root, laid out like the model's Hugging Face
// repository (e.g. "onnx/hand_detector.onnx", "hand_detector.model.json").
class ModelSource {
public:
    // One file of a bytes source. The bytes are borrowed: they must outlive every library object
    // loaded from the source (embedded data lives as long as the binary).
    struct File {
        std::string m_path;
        std::span<const std::byte> m_bytes;
    };

    // The files under a directory on disk.
    static ModelSource directory(std::filesystem::path root);
    // Files in memory, e.g. embedded into the binary.
    static ModelSource bytes(std::vector<File> files);

    bool is_directory() const noexcept;
    // The directory root; empty for a bytes source.
    const std::filesystem::path& root() const noexcept;

    // The whole file as text (model and contract descriptions). Throws Error when it is missing.
    std::string read_text(std::string_view path) const;
    // A bytes source's file; std::nullopt when it has none of that name (and for a directory).
    std::optional<std::span<const std::byte>> find_bytes(std::string_view path) const;

private:
    ModelSource() = default;

    std::filesystem::path m_root;
    std::vector<File> m_files;
    bool m_is_directory = false;
};

}  // namespace anira::models::core
