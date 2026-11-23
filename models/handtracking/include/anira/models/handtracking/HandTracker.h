// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Latest.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/SpscQueue.h>
#include <anira/models/handtracking/Decode.h>
#include <anira/models/handtracking/Geometry.h>
#include <anira/models/handtracking/Types.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace anira::models::handtracking {

/// MediaPipe's GestureRecognizerOptions.
struct Options {
    int m_num_hands = 2;                         ///< 1 .. k_max_hands
    float m_min_hand_detection_confidence = 0.5F;  ///< the palm detector's score
    float m_min_hand_presence_confidence = 0.5F;   ///< the landmark model's presence
    float m_min_tracking_confidence = 0.5F;        ///< IoU of a detection with a tracked hand
};

struct Hand {
    int m_slot = 0;          ///< stable while the hand is tracked
    bool m_tracking = false;  ///< its region came from the last landmarks, not the detector
    float m_presence = 0.0F;
    float m_handedness_right = 0.0F;  ///< the probability that it is a right hand
    int m_gesture = -1;               ///< index into k_gesture_labels, -1: none
    float m_gesture_score = 0.0F;
    std::array<Vec2, k_num_landmarks> m_landmarks{};  ///< normalised frame coordinates
    std::array<Vec2, 4> m_roi_corners{};               ///< normalised frame coordinates
};

/// The hands of one frame. Trivially copyable: no reference to the frame.
struct Hands {
    std::int64_t m_frame = -1;  ///< the frame's index since reset(), -1: none yet
    std::array<Hand, k_max_hands> m_hands{};
    std::size_t m_count = 0;
};

/// MediaPipe's gesture recognizer on anira: the palm detector while hands are missing, the
/// landmark model per hand (its region from the hand's last landmarks), the gesture embedder and
/// classifier per hand. Each model runs as its own job (core::AsyncHandler, anira's Async contract
/// from M3), one job per model in flight: process() never waits; a hand's landmarks of frame n
/// arrive with a later frame and are mapped back through frame n's crop.
///
///   - Camera thread: process(frame) per frame.
///   - Reader thread: pop(hands) gives every frame's hands, latest() the newest.
///   - Offline: run(frame) waits for each model and gives the frame's hands.
class HandTracker {
public:
    /// The four models' weights (laid out like anira-project/mediapipe-gesture-recognizer).
    HandTracker(anira::Context& context, const core::ModelSource& weights, Options options = {});
    /// Waits for the jobs in flight: they write into the tracker's buffers.
    ~HandTracker();
    HandTracker(const HandTracker&) = delete;
    HandTracker& operator=(const HandTracker&) = delete;
    HandTracker(HandTracker&&) = delete;
    HandTracker& operator=(HandTracker&&) = delete;

    /// Main thread, no process() running: every model's Async contract with `deadline` per job
    /// (a job not done by then is dropped; nullopt: none). The plans of the landmark model.
    anira::PlanReport prepare(
        std::optional<std::chrono::nanoseconds> deadline = std::chrono::milliseconds(200));
    /// A new stream: no hands, the frame index from 0, the queue empty.
    void reset();
    /// Any thread; takes effect with the next frame.
    void set_options(const Options& options) noexcept;

    /// Camera thread: the frame's hands as far as the models are done. ANIRA_OK, or
    /// ANIRA_ERROR_INVALID_ARGUMENT for an invalid frame.
    anira_status process(const Frame& frame) noexcept;

    /// Reader thread: the oldest frame's hands not taken yet.
    bool pop(Hands& hands) noexcept { return m_queue.pop(hands); }
    /// Reader thread (one): the newest frame's hands.
    [[nodiscard]] Hands latest() noexcept { return m_latest.read(); }
    /// Frames whose hands the reader did not take in time.
    [[nodiscard]] std::size_t dropped() const noexcept {
        return m_dropped.load(std::memory_order_relaxed);
    }

    /// One image's hands (every model waited for).
    static Hands run(anira::Context& context,
                     const core::ModelSource& weights,
                     const Frame& frame,
                     Options options = {});

private:
    struct Slot {
        std::unique_ptr<core::AsyncHandler> m_landmarks;
        std::unique_ptr<core::AsyncHandler> m_embedder;
        std::unique_ptr<core::AsyncHandler> m_classifier;
        std::vector<float> m_crop;
        std::array<float, k_num_landmarks * 3> m_out_landmarks{};
        float m_out_presence = 0.0F;
        float m_out_handedness = 0.0F;
        std::array<float, k_num_landmarks * 3> m_out_world{};
        std::array<float, k_num_landmarks * 3> m_embed_hand{};
        float m_embed_handedness = 0.0F;
        std::array<float, k_num_landmarks * 3> m_embed_world{};
        std::array<float, 128> m_embedding{};
        std::array<float, 128> m_classify_in{};
        std::array<float, k_gesture_labels.size()> m_scores{};

        core::Ticket m_landmarks_job;
        core::Ticket m_embedder_job;
        core::Ticket m_classifier_job;
        Affine2D m_crop_to_frame;  // of the landmark job in flight
        float m_crop_size = 0.0F;
        float m_frame_width = 0.0F;
        float m_frame_height = 0.0F;

        bool m_active = false;
        bool m_tracking = false;
        RegionOfInterest m_roi;
        std::optional<Hand> m_hand;  // the last landmarks
    };

    void take_detector(const Options& options) noexcept;
    void take_landmarks(Slot& slot, int index, const Options& options) noexcept;
    void take_gestures(Slot& slot) noexcept;
    void submit_detector(const Frame& frame) noexcept;
    void submit_landmarks(Slot& slot, const Frame& frame) noexcept;
    void wait_idle();

    anira::Context& m_context;
    std::unique_ptr<core::AsyncHandler> m_detector;
    std::array<Slot, k_max_hands> m_slots;
    std::vector<Vec2> m_anchors;

    std::vector<float> m_detector_input;
    std::vector<float> m_boxes;
    std::vector<float> m_detector_scores;
    core::Ticket m_detector_job;
    LetterboxGeometry m_letterbox;  // of the detector job in flight

    std::atomic<int> m_num_hands;
    std::atomic<float> m_min_detection;
    std::atomic<float> m_min_presence;
    std::atomic<float> m_min_tracking;

    std::int64_t m_frame = 0;
    core::SpscQueue<Hands> m_queue;
    core::Latest<Hands> m_latest;
    std::atomic<std::size_t> m_dropped{0};
};

}  // namespace anira::models::handtracking
