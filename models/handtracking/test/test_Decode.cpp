// SPDX-License-Identifier: Apache-2.0

#include <anira/models/handtracking/Decode.h>
#include <anira/models/handtracking/Geometry.h>
#include <anira/models/handtracking/Types.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace ht = anira::models::handtracking;

namespace {

constexpr float k_tolerance = 1e-4F;

void expect_near(ht::Vec2 actual, ht::Vec2 expected) {
    EXPECT_NEAR(actual.m_x, expected.m_x, k_tolerance);
    EXPECT_NEAR(actual.m_y, expected.m_y, k_tolerance);
}

// A hand pointing straight up, wrist at (cx, bottom), middle finger MCP above it.
std::array<ht::Vec2, ht::k_num_landmarks> upright_hand(float cx, float bottom, float height) {
    std::array<ht::Vec2, ht::k_num_landmarks> points;
    for (size_t i = 0; i < points.size(); ++i) {
        points[i] = {.m_x = cx + ((static_cast<float>(i % 5) - 2.0F) * height * 0.1F),
                     .m_y = bottom - (height * static_cast<float>(i) / 20.0F)};
    }
    points[0] = {.m_x = cx, .m_y = bottom};
    points[9] = {.m_x = cx, .m_y = bottom - (height * 0.5F)};
    return points;
}

}  // namespace

TEST(HandtrackingGeometry, IntersectionOverUnion) {
    const ht::Rect a{.m_x = 0.0F, .m_y = 0.0F, .m_width = 2.0F, .m_height = 2.0F};
    EXPECT_FLOAT_EQ(ht::intersection_over_union(a, a), 1.0F);
    EXPECT_FLOAT_EQ(ht::intersection_over_union(a, {3.0F, 3.0F, 1.0F, 1.0F}), 0.0F);
    EXPECT_NEAR(ht::intersection_over_union(a, {1.0F, 0.0F, 2.0F, 2.0F}), 2.0F / 6.0F, k_tolerance);
}

TEST(HandtrackingGeometry, AffineComposesAndInverts) {
    const ht::Affine2D t =
        ht::Affine2D::translation(3.0F, -2.0F).rotated(0.7F).scaled(1.5F).translated(10.0F, 4.0F);
    const ht::Vec2 point{.m_x = 5.0F, .m_y = 7.0F};
    expect_near(t.inverted().transform(t.transform(point)), point);
    // A quarter turn clockwise on screen (y down) takes +x to +y.
    expect_near(ht::Affine2D{}
                    .rotated(std::numbers::pi_v<float> * 0.5F)
                    .transform({.m_x = 1.0F, .m_y = 0.0F}),
                {.m_x = 0.0F, .m_y = 1.0F});
}

TEST(HandtrackingDecode, AnchorsCoverTheInput) {
    const std::vector<ht::Vec2> anchors = ht::ssd_anchors();
    ASSERT_EQ(anchors.size(), static_cast<size_t>(ht::k_num_anchors));
    expect_near(anchors.front(), {.m_x = 0.5F / 24.0F, .m_y = 0.5F / 24.0F});   // first cell of the
                                                                                // 24 x 24 map
    expect_near(anchors.back(), {.m_x = 11.5F / 12.0F, .m_y = 11.5F / 12.0F});  // last cell of the
                                                                                // 12 x 12 map
}

// One confident palm, pointing up, centred in a square frame: one region, upright, 2.6 x the box,
// shifted up by half the box height.
TEST(HandtrackingDecode, APalmBecomesARegion) {
    const std::vector<ht::Vec2> anchors = ht::ssd_anchors();
    std::vector<float> boxes(static_cast<size_t>(ht::k_num_anchors) * ht::k_num_detection_values);
    std::vector<float> scores(ht::k_num_anchors, -10.0F);

    // The anchor nearest the centre; a 48 x 48 px box on it (input pixels), wrist below the middle
    // finger.
    size_t index = 0;
    for (size_t i = 0; i < anchors.size(); ++i) {
        const auto distance = [](ht::Vec2 a) { return std::hypot(a.m_x - 0.5F, a.m_y - 0.5F); };
        if (distance(anchors[i]) < distance(anchors[index])) { index = i; }
    }
    float* detection = &boxes[index * ht::k_num_detection_values];
    detection[2] = 48.0F;
    detection[3] = 48.0F;
    detection[5] = 20.0F;   // wrist: 20 px below the anchor
    detection[9] = -20.0F;  // middle finger MCP: 20 px above
    scores[index] = 5.0F;

    const ht::Frame frame{.m_pixels = nullptr,
                          .m_width = 192,
                          .m_height = 192,
                          .m_stride = 0,
                          .m_pixel_stride = 4};
    std::array<ht::RegionOfInterest, ht::k_max_hands> rois;
    const size_t count =
        ht::decode_palm_rois(boxes, scores, anchors, 0.5F, 2, ht::letterbox_geometry(frame), rois);
    ASSERT_EQ(count, 1U);
    EXPECT_NEAR(rois[0].m_rotation, 0.0F, k_tolerance);
    EXPECT_NEAR(rois[0].m_size, 48.0F * 2.6F, 1e-3F);
    EXPECT_NEAR(rois[0].m_center_x, anchors[index].m_x * 192.0F, 1e-3F);
    EXPECT_NEAR(rois[0].m_center_y, (anchors[index].m_y * 192.0F) - 24.0F, 1e-3F);

    // Below the confidence nothing is decoded; a second box on top of the first is suppressed.
    EXPECT_EQ(ht::decode_palm_rois(boxes,
                                   scores,
                                   anchors,
                                   0.999F,
                                   2,
                                   ht::letterbox_geometry(frame),
                                   rois),
              0U);
    std::copy_n(detection,
                ht::k_num_detection_values,
                &boxes[(index + 1) * ht::k_num_detection_values]);
    scores[index + 1] = 4.0F;
    EXPECT_EQ(
        ht::decode_palm_rois(boxes, scores, anchors, 0.5F, 2, ht::letterbox_geometry(frame), rois),
        1U);
}

TEST(HandtrackingDecode, LandmarksBecomeTheNextRegion) {
    const auto hand = upright_hand(100.0F, 200.0F, 80.0F);
    const ht::RegionOfInterest roi = ht::roi_from_landmarks(hand);
    EXPECT_NEAR(roi.m_rotation, 0.0F, k_tolerance);
    EXPECT_NEAR(roi.m_size, 2.0F * 80.0F, 1e-3F);  // the long side of the bounding box, x 2
    EXPECT_TRUE(ht::roi_matches_tracked(roi, roi, 0.5F));

    // The crop transform puts the region's centre in the middle of the landmark model's input.
    const ht::Affine2D crop = ht::roi_frame_to_crop(roi);
    const float middle = static_cast<float>(ht::k_landmark_input_size) * 0.5F;
    expect_near(crop.transform({.m_x = roi.m_center_x, .m_y = roi.m_center_y}),
                {.m_x = middle, .m_y = middle});
}

TEST(HandtrackingDecode, EmbedderInputStartsAtTheWrist) {
    const auto hand = upright_hand(100.0F, 200.0F, 80.0F);
    const std::array<float, ht::k_num_landmarks> z{};
    std::array<float, ht::k_num_landmarks * 3> input{};
    ht::normalize_landmarks_for_embedder(hand, z, 0.0F, input);
    EXPECT_FLOAT_EQ(input[0], 0.0F);
    EXPECT_FLOAT_EQ(input[1], 0.0F);
    float extent = 0.0F;
    for (size_t i = 0; i < ht::k_num_landmarks; ++i) {
        extent = std::fmax(extent, std::fabs(input[(i * 3) + 1]));
    }
    EXPECT_NEAR(extent, 1.0F, 1e-3F);  // scaled by the larger bounding box side
}

TEST(HandtrackingDecode, GestureFromScores) {
    EXPECT_EQ(ht::classify_gesture({0.0F, 0.0F, 0.9F, 0.1F, 0.0F, 0.0F, 0.0F, 0.0F}).first, 2);
    EXPECT_EQ(ht::classify_gesture({0.9F, 0.0F, 0.1F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}).first, -1);
    // Logits get a softmax first.
    const auto [index, score] =
        ht::classify_gesture({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 9.0F, 0.0F, 0.0F});
    EXPECT_EQ(index, 5);
    EXPECT_GT(score, 0.99F);
}

TEST(HandtrackingDecode, PixelsBecomeAnRgbTensor) {
    // 2 x 2 BGRA: red, green / blue, white.
    const std::array<std::uint8_t, 16>
        pixels{0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255};
    const ht::Frame frame{.m_pixels = pixels.data(),
                          .m_width = 2,
                          .m_height = 2,
                          .m_stride = 8,
                          .m_pixel_stride = 4,
                          .m_format = ht::PixelFormat::Bgra};
    std::array<float, 12> tensor{};  // 2 x 2 pixels x RGB
    ht::sample_to_tensor_rgb(frame, ht::Affine2D{}, 2, tensor);
    const std::array<float, 12> expected{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_NEAR(tensor[i], expected[i], k_tolerance) << i;
    }

    // Outside the frame is black.
    ht::sample_to_tensor_rgb(frame, ht::Affine2D::translation(10.0F, 10.0F), 2, tensor);
    for (const float value : tensor) { EXPECT_EQ(value, 0.0F); }
}
