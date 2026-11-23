// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/handtracking/Types.h>

#include <cmath>

// The y axis points down; a positive rotation is clockwise on screen. Affine2D composes left to
// right.
namespace anira::models::handtracking {

struct Rect {
    float m_x = 0.0F;
    float m_y = 0.0F;
    float m_width = 0.0F;
    float m_height = 0.0F;

    [[nodiscard]] float right() const { return m_x + m_width; }
    [[nodiscard]] float bottom() const { return m_y + m_height; }
    [[nodiscard]] float centre_x() const { return m_x + (m_width * 0.5F); }
    [[nodiscard]] float centre_y() const { return m_y + (m_height * 0.5F); }
    [[nodiscard]] bool contains(Vec2 point) const {
        return point.m_x >= m_x && point.m_x < right() && point.m_y >= m_y && point.m_y < bottom();
    }
};

inline float intersection_over_union(const Rect& a, const Rect& b) {
    const float overlap_w = std::fmin(a.right(), b.right()) - std::fmax(a.m_x, b.m_x);
    const float overlap_h = std::fmin(a.bottom(), b.bottom()) - std::fmax(a.m_y, b.m_y);
    const float intersection =
        (overlap_w > 0.0F && overlap_h > 0.0F) ? overlap_w * overlap_h : 0.0F;
    const float union_area = (a.m_width * a.m_height) + (b.m_width * b.m_height) - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

/// x' = m_00 * x + m_01 * y + m_02, y' = m_10 * x + m_11 * y + m_12
struct Affine2D {
    float m_00 = 1.0F, m_01 = 0.0F, m_02 = 0.0F;
    float m_10 = 0.0F, m_11 = 1.0F, m_12 = 0.0F;

    static Affine2D translation(float dx, float dy) {
        return {.m_00 = 1.0F, .m_01 = 0.0F, .m_02 = dx, .m_10 = 0.0F, .m_11 = 1.0F, .m_12 = dy};
    }

    /// This transform followed by `n` (the matrix product n * this).
    [[nodiscard]] Affine2D followed_by(const Affine2D& n) const {
        return {.m_00 = (n.m_00 * m_00) + (n.m_01 * m_10),
                .m_01 = (n.m_00 * m_01) + (n.m_01 * m_11),
                .m_02 = (n.m_00 * m_02) + (n.m_01 * m_12) + n.m_02,
                .m_10 = (n.m_10 * m_00) + (n.m_11 * m_10),
                .m_11 = (n.m_10 * m_01) + (n.m_11 * m_11),
                .m_12 = (n.m_10 * m_02) + (n.m_11 * m_12) + n.m_12};
    }
    [[nodiscard]] Affine2D translated(float dx, float dy) const {
        return followed_by(translation(dx, dy));
    }
    [[nodiscard]] Affine2D rotated(float radians) const {
        const float c = std::cos(radians);
        const float s = std::sin(radians);
        return followed_by(
            {.m_00 = c, .m_01 = -s, .m_02 = 0.0F, .m_10 = s, .m_11 = c, .m_12 = 0.0F});
    }
    [[nodiscard]] Affine2D scaled(float factor) const {
        return followed_by({.m_00 = factor,
                            .m_01 = 0.0F,
                            .m_02 = 0.0F,
                            .m_10 = 0.0F,
                            .m_11 = factor,
                            .m_12 = 0.0F});
    }
    [[nodiscard]] Affine2D inverted() const {
        const float det = (m_00 * m_11) - (m_01 * m_10);
        const float inv_det = det != 0.0F ? 1.0F / det : 0.0F;
        const float i00 = m_11 * inv_det;
        const float i01 = -m_01 * inv_det;
        const float i10 = -m_10 * inv_det;
        const float i11 = m_00 * inv_det;
        return {.m_00 = i00,
                .m_01 = i01,
                .m_02 = -((i00 * m_02) + (i01 * m_12)),
                .m_10 = i10,
                .m_11 = i11,
                .m_12 = -((i10 * m_02) + (i11 * m_12))};
    }
    [[nodiscard]] Vec2 transform(Vec2 point) const {
        return {.m_x = (m_00 * point.m_x) + (m_01 * point.m_y) + m_02,
                .m_y = (m_10 * point.m_x) + (m_11 * point.m_y) + m_12};
    }
};

}  // namespace anira::models::handtracking
