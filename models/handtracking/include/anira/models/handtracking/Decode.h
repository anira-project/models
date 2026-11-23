// SPDX-License-Identifier: Apache-2.0

#pragma once

// The MediaPipe gesture recognizer's calculators as plain functions over the models' tensors:
// pixels into the input tensors, output tensors into hands. No anira, no allocation except where
// a function returns a vector. Copied from anira-video-example's src/handtracking (MediaPipeOps.h,
// TrackerShared.h, InferenceConfigs.h).

#include <anira/models/handtracking/Geometry.h>
#include <anira/models/handtracking/Types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace anira::models::handtracking {

inline constexpr int k_max_hands = 4;
inline constexpr int k_detector_input_size = 192;  ///< hand_detector: image [1, 192, 192, 3]
inline constexpr int k_num_anchors = 2016;         ///< hand_detector: boxes [1, 2016, 18]
inline constexpr int k_num_detection_values = 18;  ///< 4 box + 7 keypoints x 2
inline constexpr int k_landmark_input_size = 224;  ///< hand_landmarks_detector: image [1, 224, 224,
                                                   ///< 3]
inline constexpr std::size_t k_num_landmarks = 21;
inline constexpr float k_nms_iou_threshold = 0.3F;
inline constexpr float k_min_gesture_confidence = 0.5F;

/// The canned gesture classifier's classes, in its output order.
inline constexpr std::array<const char*, 8> k_gesture_labels{"None",
                                                             "Closed_Fist",
                                                             "Open_Palm",
                                                             "Pointing_Up",
                                                             "Thumb_Down",
                                                             "Thumb_Up",
                                                             "Victory",
                                                             "ILoveYou"};

/// MediaPipe's hand skeleton: landmark index pairs.
inline constexpr std::array<std::array<int, 2>, 21> k_hand_connections{{
    {0, 1},   {1, 2},   {2, 3},   {3, 4},    // thumb
    {0, 5},   {5, 6},   {6, 7},   {7, 8},    // index
    {5, 9},   {9, 10},  {10, 11}, {11, 12},  // middle
    {9, 13},  {13, 14}, {14, 15}, {15, 16},  // ring
    {13, 17}, {17, 18}, {18, 19}, {19, 20},  // pinky
    {0, 17},                                 // palm
}};

inline float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-value));
}

/// Rotation that turns from -> to straight up (MediaPipe's convention, y axis down).
inline float rotation_from_points(Vec2 from, Vec2 to) {
    return (std::numbers::pi_v<float> * 0.5F) - std::atan2(-(to.m_y - from.m_y), to.m_x - from.m_x);
}

/// Moves a centre by shift * size along the rotated y axis.
inline Vec2 shift_along_rotated_y(Vec2 center, float rotation, float shift, float size) {
    return {.m_x = center.m_x - (std::sin(rotation) * shift * size),
            .m_y = center.m_y + (std::cos(rotation) * shift * size)};
}

/// The detector's SSD anchors: strides 8, 16, 16, 16; two anchors per cell on the 24 x 24 map, six
/// on the 12 x 12 map: 2016 centres in [0, 1].
inline std::vector<Vec2> ssd_anchors() {
    constexpr std::array<int, 4> k_strides{8, 16, 16, 16};
    std::vector<Vec2> anchors;
    anchors.reserve(k_num_anchors);
    size_t layer = 0;
    while (layer < k_strides.size()) {
        size_t last_same_stride = layer;
        int anchors_per_cell = 0;
        while (last_same_stride < k_strides.size() &&
               k_strides[last_same_stride] == k_strides[layer]) {
            ++last_same_stride;
            anchors_per_cell += 2;  // aspect ratio 1.0 + interpolated scale
        }
        const int map_size = k_detector_input_size / k_strides[layer];
        for (int y = 0; y < map_size; ++y) {
            for (int x = 0; x < map_size; ++x) {
                for (int anchor = 0; anchor < anchors_per_cell; ++anchor) {
                    anchors.push_back(
                        {.m_x = (static_cast<float>(x) + 0.5F) / static_cast<float>(map_size),
                         .m_y = (static_cast<float>(y) + 0.5F) / static_cast<float>(map_size)});
                }
            }
        }
        layer = last_same_stride;
    }
    return anchors;
}

/// A rotated square region of interest in frame pixels.
struct RegionOfInterest {
    float m_center_x = 0.0F;
    float m_center_y = 0.0F;
    float m_size = 0.0F;      ///< side length
    float m_rotation = 0.0F;  ///< radians that turn the hand up in the crop

    [[nodiscard]] Rect bounding_box() const {
        return {.m_x = m_center_x - (m_size * 0.5F),
                .m_y = m_center_y - (m_size * 0.5F),
                .m_width = m_size,
                .m_height = m_size};
    }
};

/// How a frame was fit into the detector's square input: needed to map detections back.
struct LetterboxGeometry {
    float m_side = 0.0F;  ///< side of the square the frame was fit into
    float m_pad_x = 0.0F;
    float m_pad_y = 0.0F;
};

inline LetterboxGeometry letterbox_geometry(const Frame& frame) {
    const auto width = static_cast<float>(frame.m_width);
    const auto height = static_cast<float>(frame.m_height);
    const float side = std::fmax(width, height);
    return {.m_side = side, .m_pad_x = (side - width) * 0.5F, .m_pad_y = (side - height) * 0.5F};
}

/// Detector input -> frame: letterboxes the whole frame into the square input.
inline Affine2D letterbox_dest_to_source(const LetterboxGeometry& geometry) {
    return Affine2D::translation(geometry.m_pad_x, geometry.m_pad_y)
        .scaled(static_cast<float>(k_detector_input_size) / geometry.m_side)
        .inverted();
}

/// Frame -> landmark model input for a hand's region of interest.
inline Affine2D roi_frame_to_crop(const RegionOfInterest& roi) {
    const auto side = static_cast<float>(k_landmark_input_size);
    return Affine2D::translation(-roi.m_center_x, -roi.m_center_y)
        .rotated(-roi.m_rotation)
        .scaled(side / roi.m_size)
        .translated(side * 0.5F, side * 0.5F);
}

/// Samples a transformed square of the frame into an interleaved RGB float tensor in [0, 1]
/// (bilinear; outside the frame is black). Writes dest_size * dest_size * 3 values; no allocation.
inline void sample_to_tensor_rgb(const Frame& source,
                                 const Affine2D& dest_to_source,
                                 int dest_size,
                                 std::span<float> destination) {
    int r_off = 2;  // BGR / BGRA
    int g_off = 1;
    int b_off = 0;
    switch (source.m_format) {
        case PixelFormat::Rgb:
        case PixelFormat::Rgba:
            r_off = 0;
            b_off = 2;
            break;
        case PixelFormat::Argb:
            r_off = 1;
            g_off = 2;
            b_off = 3;
            break;
        case PixelFormat::Bgr:
        case PixelFormat::Bgra: break;
    }
    struct Rgb {
        float m_r = 0.0F, m_g = 0.0F, m_b = 0.0F;
    };
    const auto sample = [&](int x, int y) -> Rgb {
        if (x < 0 || y < 0 || x >= source.m_width || y >= source.m_height) { return {}; }
        const std::uint8_t* pixel =
            source.m_pixels + (static_cast<size_t>(y) * static_cast<size_t>(source.m_stride)) +
            (static_cast<size_t>(x) * static_cast<size_t>(source.m_pixel_stride));
        constexpr float k_inv255 = 1.0F / 255.0F;
        return {.m_r = static_cast<float>(pixel[r_off]) * k_inv255,
                .m_g = static_cast<float>(pixel[g_off]) * k_inv255,
                .m_b = static_cast<float>(pixel[b_off]) * k_inv255};
    };
    size_t i = 0;
    for (int y = 0; y < dest_size; ++y) {
        for (int x = 0; x < dest_size; ++x) {
            const Vec2 point = dest_to_source.transform(
                {.m_x = static_cast<float>(x) + 0.5F, .m_y = static_cast<float>(y) + 0.5F});
            const float source_x = point.m_x - 0.5F;
            const float source_y = point.m_y - 0.5F;
            const int x0 = static_cast<int>(std::floor(source_x));
            const int y0 = static_cast<int>(std::floor(source_y));
            const float tx = source_x - static_cast<float>(x0);
            const float ty = source_y - static_cast<float>(y0);
            const Rgb c00 = sample(x0, y0);
            const Rgb c10 = sample(x0 + 1, y0);
            const Rgb c01 = sample(x0, y0 + 1);
            const Rgb c11 = sample(x0 + 1, y0 + 1);
            const auto lerp = [tx, ty](float v00, float v10, float v01, float v11) {
                const float top = v00 + ((v10 - v00) * tx);
                const float bottom = v01 + ((v11 - v01) * tx);
                return top + ((bottom - top) * ty);
            };
            destination[i++] = lerp(c00.m_r, c10.m_r, c01.m_r, c11.m_r);
            destination[i++] = lerp(c00.m_g, c10.m_g, c01.m_g, c11.m_g);
            destination[i++] = lerp(c00.m_b, c10.m_b, c01.m_b, c11.m_b);
        }
    }
}

/// The detector's outputs into hand regions: threshold, sort, greedy NMS, then MediaPipe's
/// palm_detection_detection_to_roi (rotate wrist -> middle finger up, scale 2.6, shift -0.5 along
/// the rotated y). Returns how many of `rois` it filled (at most max_hands).
inline size_t decode_palm_rois(std::span<const float> boxes,   ///< [k_num_anchors, 18]
                               std::span<const float> scores,  ///< [k_num_anchors]
                               std::span<const Vec2> anchors,
                               float min_detection_confidence,
                               int max_hands,
                               const LetterboxGeometry& letterbox,
                               std::array<RegionOfInterest, k_max_hands>& rois) {
    std::array<std::pair<float, int>, k_num_anchors> candidates;
    size_t num_candidates = 0;
    for (int i = 0; i < k_num_anchors; ++i) {
        const float score = sigmoid(std::clamp(scores[static_cast<size_t>(i)], -100.0F, 100.0F));
        if (score >= min_detection_confidence) { candidates[num_candidates++] = {score, i}; }
    }
    std::sort(candidates.begin(),
              candidates.begin() + static_cast<std::ptrdiff_t>(num_candidates),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // A palm box in normalised letterbox coordinates.
    const float inv_scale = 1.0F / static_cast<float>(k_detector_input_size);
    const auto detection_of = [&](int index) {
        return boxes.subspan(static_cast<size_t>(index) * k_num_detection_values,
                             k_num_detection_values);
    };
    const auto decode_box = [&](int index) {
        const auto detection = detection_of(index);
        const Vec2 anchor = anchors[static_cast<size_t>(index)];
        const float center_x = (detection[0] * inv_scale) + anchor.m_x;
        const float center_y = (detection[1] * inv_scale) + anchor.m_y;
        const float width = detection[2] * inv_scale;
        const float height = detection[3] * inv_scale;
        return Rect{.m_x = center_x - (width * 0.5F),
                    .m_y = center_y - (height * 0.5F),
                    .m_width = width,
                    .m_height = height};
    };
    const auto to_frame = [&](float x, float y) {
        return Vec2{.m_x = (x * letterbox.m_side) - letterbox.m_pad_x,
                    .m_y = (y * letterbox.m_side) - letterbox.m_pad_y};
    };

    std::array<Rect, k_max_hands> kept;
    size_t num_kept = 0;
    for (size_t c = 0; c < num_candidates && std::cmp_less(num_kept, max_hands); ++c) {
        const int index = candidates[c].second;
        const Rect box = decode_box(index);
        const bool suppressed =
            std::any_of(kept.begin(), kept.begin() + num_kept, [&](const Rect& k) {
                return intersection_over_union(box, k) > k_nms_iou_threshold;
            });
        if (suppressed) { continue; }

        const auto detection = detection_of(index);
        const Vec2 anchor = anchors[static_cast<size_t>(index)];
        const Vec2 box_center = to_frame(box.centre_x(), box.centre_y());
        const float box_width = box.m_width * letterbox.m_side;
        const float box_height = box.m_height * letterbox.m_side;
        // Keypoint 0 is the wrist, keypoint 2 the middle finger's MCP joint.
        const Vec2 wrist = to_frame((detection[4] * inv_scale) + anchor.m_x,
                                    (detection[5] * inv_scale) + anchor.m_y);
        const Vec2 middle = to_frame((detection[8] * inv_scale) + anchor.m_x,
                                     (detection[9] * inv_scale) + anchor.m_y);
        const float rotation = rotation_from_points(wrist, middle);
        const Vec2 center = shift_along_rotated_y(box_center, rotation, -0.5F, box_height);

        kept[num_kept] = box;
        rois[num_kept++] = {.m_center_x = center.m_x,
                            .m_center_y = center.m_y,
                            .m_size = std::fmax(box_width, box_height) * 2.6F,
                            .m_rotation = rotation};
    }
    return num_kept;
}

/// Whether a detected region is a hand already tracked. IoU alone is unreliable (detection and
/// tracking regions differ in size, 2.6 x and 2.0 x), so a centre inside also counts.
inline bool roi_matches_tracked(const RegionOfInterest& roi,
                                const RegionOfInterest& tracked,
                                float min_tracking_confidence) {
    const Rect tracked_box = tracked.bounding_box();
    return intersection_over_union(roi.bounding_box(), tracked_box) > min_tracking_confidence ||
           tracked_box.contains({.m_x = roi.m_center_x, .m_y = roi.m_center_y});
}

/// The next frame's region from this frame's landmarks (frame pixels): MediaPipe's
/// hand_landmark_landmarks_to_roi (rotate wrist -> middle finger MCP up, bounding box long side
/// x 2.0, shift -0.1).
inline RegionOfInterest roi_from_landmarks(const std::array<Vec2, k_num_landmarks>& landmarks) {
    float min_x = landmarks[0].m_x;
    float max_x = landmarks[0].m_x;
    float min_y = landmarks[0].m_y;
    float max_y = landmarks[0].m_y;
    for (const Vec2& point : landmarks) {
        min_x = std::fmin(min_x, point.m_x);
        max_x = std::fmax(max_x, point.m_x);
        min_y = std::fmin(min_y, point.m_y);
        max_y = std::fmax(max_y, point.m_y);
    }
    const float rotation = rotation_from_points(landmarks[0], landmarks[9]);
    const float size = std::fmax(max_x - min_x, max_y - min_y) * 2.0F;
    const Vec2 center =
        shift_along_rotated_y({.m_x = (min_x + max_x) * 0.5F, .m_y = (min_y + max_y) * 0.5F},
                              rotation,
                              -0.1F,
                              size);
    return {.m_center_x = center.m_x,
            .m_center_y = center.m_y,
            .m_size = size,
            .m_rotation = rotation};
}

/// The gesture embedder's landmark input: MediaPipe's LandmarksToMatrixCalculator (rotate the hand
/// up, origin at the wrist, scale by the larger bounding box side); 21 x (x, y, z) interleaved.
inline void normalize_landmarks_for_embedder(const std::array<Vec2, k_num_landmarks>& points,
                                             const std::array<float, k_num_landmarks>& z_values,
                                             float rotation,
                                             std::array<float, k_num_landmarks * 3>& destination) {
    const float cos_r = std::cos(rotation);
    const float sin_r = -std::sin(rotation);  // y axis points down
    std::array<Vec2, k_num_landmarks> rotated;
    for (size_t i = 0; i < rotated.size(); ++i) {
        rotated[i] = {.m_x = (points[i].m_x * cos_r) - (points[i].m_y * sin_r),
                      .m_y = (points[i].m_y * cos_r) + (points[i].m_x * sin_r)};
    }
    float min_x = rotated[0].m_x;
    float max_x = rotated[0].m_x;
    float min_y = rotated[0].m_y;
    float max_y = rotated[0].m_y;
    for (const Vec2& point : rotated) {
        min_x = std::fmin(min_x, point.m_x);
        max_x = std::fmax(max_x, point.m_x);
        min_y = std::fmin(min_y, point.m_y);
        max_y = std::fmax(max_y, point.m_y);
    }
    const float scale = std::fmax(max_x - min_x, max_y - min_y) + 1e-5F;
    const Vec2 origin = rotated[0];
    const float origin_z = z_values[0];
    for (size_t i = 0; i < rotated.size(); ++i) {
        destination[(i * 3) + 0] = (rotated[i].m_x - origin.m_x) / scale;
        destination[(i * 3) + 1] = (rotated[i].m_y - origin.m_y) / scale;
        destination[(i * 3) + 2] = (z_values[i] - origin_z) / scale;
    }
}

/// The classifier's scores into {label index, score}; index -1 when the best is "None" or below
/// k_min_gesture_confidence. Scores that are no probabilities (sum not 1) get a softmax first.
inline std::pair<int, float> classify_gesture(std::array<float, k_gesture_labels.size()> scores) {
    float sum = 0.0F;
    for (const float score : scores) { sum += score; }
    if (sum < 0.99F || sum > 1.01F) {
        const float max_score = *std::ranges::max_element(scores);
        sum = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
        }
        for (float& score : scores) { score /= sum; }
    }
    const auto best = std::ranges::max_element(scores);
    const auto index = static_cast<int>(best - scores.begin());
    if (index == 0 || *best < k_min_gesture_confidence) { return {-1, *best}; }
    return {index, *best};
}

}  // namespace anira::models::handtracking
