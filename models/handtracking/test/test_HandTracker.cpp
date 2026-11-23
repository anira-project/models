// SPDX-License-Identifier: Apache-2.0

// HandTracker on MediaPipe's own test images (mediapipe-assets: thumb_up, victory, fist,
// pointing_up, right_hands), as PPM in test/ of the weights' HF repository: each image's gesture
// and wrist against MediaPipe's GestureRecognizer, offline (run) and streamed (process, the camera
// thread's way); two hands; no hands in an empty frame. The
// tests need ANIRA_MODELS_TEST_WEIGHTS/mediapipe-gesture-recognizer; without it they skip.

#include <anira/models/core/ModelSource.h>
#include <anira/models/handtracking/Decode.h>
#include <anira/models/handtracking/HandTracker.h>
#include <anira/models/handtracking/Types.h>
#include <gtest/gtest.h>

#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace core = anira::models::core;
namespace ht = anira::models::handtracking;
namespace fs = std::filesystem;

namespace {

std::optional<fs::path> weights_dir() {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    const fs::path dir = fs::path(root) / "mediapipe-gesture-recognizer";
    if (!fs::exists(dir / "test")) { return std::nullopt; }
    return dir;
}

// A binary PPM (P6, 8 bits) as an RGB frame that owns its pixels.
ht::Frame read_ppm(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    int width = 0;
    int height = 0;
    int max_value = 0;
    file >> magic >> width >> height >> max_value;
    file.get();
    auto pixels = std::make_shared<std::vector<std::uint8_t>>(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    file.read(reinterpret_cast<char*>(pixels->data()), static_cast<std::streamsize>(pixels->size()));
    return {.m_pixels = pixels->data(),
            .m_width = width,
            .m_height = height,
            .m_stride = width * 3,
            .m_pixel_stride = 3,
            .m_format = ht::PixelFormat::Rgb,
            .m_keep_alive = pixels};
}

std::string_view gesture_of(const ht::Hands& hands) {
    if (hands.m_count == 0 || hands.m_hands[0].m_gesture < 0) { return "None"; }
    return ht::k_gesture_labels.at(static_cast<std::size_t>(hands.m_hands[0].m_gesture));
}

// MediaPipe's own GestureRecognizer on the same images (export/make_test_images.py, mediapipe
// 0.10, gesture_recognizer.task float16/1): the gesture and the wrist in normalised coordinates.
struct Case {
    const char* m_image;
    std::string_view m_gesture;
    ht::Vec2 m_wrist;
};
constexpr std::array<Case, 4> k_cases{
    {{.m_image = "thumb_up", .m_gesture = "Thumb_Up", .m_wrist = {.m_x = 0.639F, .m_y = 0.671F}},
     {.m_image = "victory", .m_gesture = "Victory", .m_wrist = {.m_x = 0.516F, .m_y = 0.804F}},
     {.m_image = "fist", .m_gesture = "Closed_Fist", .m_wrist = {.m_x = 0.477F, .m_y = 0.661F}},
     {.m_image = "pointing_up",
      .m_gesture = "Pointing_Up",
      .m_wrist = {.m_x = 0.479F, .m_y = 0.743F}}}};
// Tracking crops by the last landmarks, MediaPipe's image mode by the palm detection.
constexpr float k_wrist_tolerance = 0.03F;

}  // namespace

TEST(HandTracker, RunFindsEachImagesGesture) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/mediapipe-gesture-recognizer/test is missing"; }
    anira::Context context{anira::ContextConfig{}};
    for (const Case& c : k_cases) {
        const ht::Frame frame = read_ppm(*dir / "test" / (std::string(c.m_image) + ".ppm"));
        const ht::Hands hands =
            ht::HandTracker::run(context, core::ModelSource::directory(*dir), frame);
        ASSERT_GE(hands.m_count, 1U) << c.m_image;
        EXPECT_GT(hands.m_hands[0].m_presence, 0.5F) << c.m_image;
        EXPECT_EQ(gesture_of(hands), c.m_gesture) << c.m_image;
        EXPECT_NEAR(hands.m_hands[0].m_landmarks[0].m_x, c.m_wrist.m_x, k_wrist_tolerance) << c.m_image;
        EXPECT_NEAR(hands.m_hands[0].m_landmarks[0].m_y, c.m_wrist.m_y, k_wrist_tolerance) << c.m_image;
        EXPECT_GT(hands.m_hands[0].m_handedness_right, 0.5F) << c.m_image;  // MediaPipe: Right
    }
}

TEST(HandTracker, RunFindsTwoHands) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/mediapipe-gesture-recognizer/test is missing"; }
    anira::Context context{anira::ContextConfig{}};
    const ht::Frame frame = read_ppm(*dir / "test" / "right_hands.ppm");
    const ht::Hands hands = ht::HandTracker::run(context, core::ModelSource::directory(*dir), frame);
    EXPECT_EQ(hands.m_count, 2U);
}

TEST(HandTracker, StreamedFramesGiveTheGestureAndEveryFrame) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/mediapipe-gesture-recognizer/test is missing"; }
    anira::Context context{anira::ContextConfig{}};
    ht::HandTracker tracker(context, core::ModelSource::directory(*dir));
    tracker.prepare();
    const ht::Frame frame = read_ppm(*dir / "test" / "victory.ppm");
    constexpr int k_frames = 60;
    for (int i = 0; i < k_frames; ++i) {
        ASSERT_EQ(tracker.process(frame), ANIRA_OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // 30 fps
    }
    ht::Hands hands;
    std::int64_t expected = 0;
    while (tracker.pop(hands)) {
        EXPECT_EQ(hands.m_frame, expected++);
    }
    EXPECT_EQ(expected, k_frames);
    EXPECT_EQ(tracker.dropped(), 0U);
    const ht::Hands last = tracker.latest();
    EXPECT_EQ(last.m_frame, k_frames - 1);
    EXPECT_EQ(gesture_of(last), "Victory");
}

TEST(HandTracker, AnEmptyFrameHasNoHands) {
    const auto dir = weights_dir();
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/mediapipe-gesture-recognizer/test is missing"; }
    anira::Context context{anira::ContextConfig{}};
    auto pixels = std::make_shared<std::vector<std::uint8_t>>(640 * 480 * 3, std::uint8_t{40});
    const ht::Frame frame{.m_pixels = pixels->data(),
                          .m_width = 640,
                          .m_height = 480,
                          .m_stride = 640 * 3,
                          .m_pixel_stride = 3,
                          .m_format = ht::PixelFormat::Rgb,
                          .m_keep_alive = pixels};
    const ht::Hands hands = ht::HandTracker::run(context, core::ModelSource::directory(*dir), frame);
    EXPECT_EQ(hands.m_count, 0U);
}
