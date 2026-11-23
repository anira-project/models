// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

namespace anira::models::handtracking {

struct Vec2 {
    float m_x = 0.0F;
    float m_y = 0.0F;
};

/// Byte order of a pixel in memory, independent of the host's naming convention.
enum class PixelFormat { Rgb, Bgr, Rgba, Bgra, Argb };

/// One camera frame or image. The tracker may keep a frame beyond the call: `m_pixels` must stay
/// valid while `m_keep_alive` holds a reference; copies only count references.
struct Frame {
    const std::uint8_t* m_pixels = nullptr;
    int m_width = 0;
    int m_height = 0;
    int m_stride = 0;        ///< bytes per row
    int m_pixel_stride = 0;  ///< bytes per pixel (3 or 4)
    PixelFormat m_format = PixelFormat::Bgra;
    std::shared_ptr<const void> m_keep_alive;

    [[nodiscard]] bool is_valid() const {
        return m_pixels != nullptr && m_width > 0 && m_height > 0;
    }
};

}  // namespace anira::models::handtracking
