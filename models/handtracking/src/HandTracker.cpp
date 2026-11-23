// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/handtracking/Decode.h>
#include <anira/models/handtracking/Descriptions.h>
#include <anira/models/handtracking/Geometry.h>
#include <anira/models/handtracking/HandTracker.h>
#include <anira/models/handtracking/Types.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>

namespace anira::models::handtracking {

namespace {

constexpr std::size_t k_queue = 256;  // frames a reader may fall behind: about 8 s at 30 fps

anira::Tensor host(float* data, std::initializer_list<std::int64_t> shape) noexcept {
    return anira::Tensor::from_host(
        data, ANIRA_DTYPE_F32, std::span<const std::int64_t>(shape.begin(), shape.size()));
}

bool done(const core::Ticket& ticket) noexcept {
    const core::TicketStatus status = ticket.status();
    return status != core::TicketStatus::Pending;
}

bool succeeded(const core::Ticket& ticket) noexcept {
    const core::TicketStatus status = ticket.status();
    return status == core::TicketStatus::Met || status == core::TicketStatus::Late;
}

// A job slot is free when no ticket is held or the held one is done.
bool idle(const core::Ticket& ticket) noexcept {
    return ticket.status() == core::TicketStatus::Failed || done(ticket);
}

}  // namespace

HandTracker::HandTracker(anira::Context& context, const core::ModelSource& weights, Options options)
    : m_context(context)
    , m_detector(std::make_unique<core::AsyncHandler>(context, k_hand_detector_model_json, weights))
    , m_anchors(ssd_anchors())
    , m_detector_input(static_cast<std::size_t>(k_detector_input_size) * k_detector_input_size * 3)
    , m_boxes(static_cast<std::size_t>(k_num_anchors) * k_num_detection_values)
    , m_detector_scores(k_num_anchors)
    , m_num_hands(std::clamp(options.m_num_hands, 1, k_max_hands))
    , m_min_detection(options.m_min_hand_detection_confidence)
    , m_min_presence(options.m_min_hand_presence_confidence)
    , m_min_tracking(options.m_min_tracking_confidence) {
    for (Slot& slot : m_slots) {
        slot.m_landmarks = std::make_unique<core::AsyncHandler>(
            context, k_hand_landmarks_detector_model_json, weights);
        slot.m_embedder =
            std::make_unique<core::AsyncHandler>(context, k_gesture_embedder_model_json, weights);
        slot.m_classifier = std::make_unique<core::AsyncHandler>(
            context, k_canned_gesture_classifier_model_json, weights);
        slot.m_crop.assign(
            static_cast<std::size_t>(k_landmark_input_size) * k_landmark_input_size * 3, 0.0F);
    }
    m_queue.reserve(k_queue);
}

HandTracker::~HandTracker() { wait_idle(); }

anira::PlanReport HandTracker::prepare(std::optional<std::chrono::nanoseconds> deadline) {
    const anira::Async contract{.deadline = deadline,
                                .on_late = deadline ? ANIRA_LATE_DROP : ANIRA_LATE_FINISH,
                                .max_in_flight = 1};
    m_detector_job = {};
    m_detector->prepare(contract);
    m_detector->bind_output(0, host(m_boxes.data(), {1, k_num_anchors, k_num_detection_values}));
    m_detector->bind_output(1, host(m_detector_scores.data(), {1, k_num_anchors, 1}));
    std::optional<anira::PlanReport> report;
    for (Slot& slot : m_slots) {
        slot.m_landmarks_job = {};
        slot.m_embedder_job = {};
        slot.m_classifier_job = {};
        anira::PlanReport slot_report = slot.m_landmarks->prepare(contract);
        if (!report) { report.emplace(slot_report); }
        slot.m_landmarks->bind_output(0, host(slot.m_out_landmarks.data(), {1, 63}));
        slot.m_landmarks->bind_output(1, host(&slot.m_out_presence, {1, 1}));
        slot.m_landmarks->bind_output(2, host(&slot.m_out_handedness, {1, 1}));
        slot.m_landmarks->bind_output(3, host(slot.m_out_world.data(), {1, 63}));
        slot.m_embedder->prepare(contract);
        slot.m_embedder->bind_output(0, host(slot.m_embedding.data(), {1, 128}));
        slot.m_classifier->prepare(contract);
        slot.m_classifier->bind_output(
            0, host(slot.m_scores.data(), {1, static_cast<std::int64_t>(k_gesture_labels.size())}));
    }
    reset();
    return *report;
}

void HandTracker::reset() {
    wait_idle();
    m_detector_job = {};
    for (Slot& slot : m_slots) {
        slot.m_landmarks_job = {};
        slot.m_embedder_job = {};
        slot.m_classifier_job = {};
        slot.m_active = false;
        slot.m_tracking = false;
        slot.m_hand.reset();
    }
    m_frame = 0;
    m_queue.clear();
    m_latest.reset();
    m_dropped.store(0, std::memory_order_relaxed);
}

void HandTracker::set_options(const Options& options) noexcept {
    m_num_hands.store(std::clamp(options.m_num_hands, 1, k_max_hands), std::memory_order_relaxed);
    m_min_detection.store(options.m_min_hand_detection_confidence, std::memory_order_relaxed);
    m_min_presence.store(options.m_min_hand_presence_confidence, std::memory_order_relaxed);
    m_min_tracking.store(options.m_min_tracking_confidence, std::memory_order_relaxed);
}

anira_status HandTracker::process(const Frame& frame) noexcept {
    if (!frame.is_valid()) { return ANIRA_ERROR_INVALID_ARGUMENT; }
    const Options options{.m_num_hands = m_num_hands.load(std::memory_order_relaxed),
                          .m_min_hand_detection_confidence =
                              m_min_detection.load(std::memory_order_relaxed),
                          .m_min_hand_presence_confidence =
                              m_min_presence.load(std::memory_order_relaxed),
                          .m_min_tracking_confidence = m_min_tracking.load(std::memory_order_relaxed)};

    // What finished since the last frame.
    take_detector(options);
    for (std::size_t i = 0; i < m_slots.size(); ++i) {
        // Gestures first: a finished embedder job goes on to the classifier before new landmarks
        // start the next one.
        take_gestures(m_slots[i]);
        take_landmarks(m_slots[i], static_cast<int>(i), options);
    }

    // New jobs: the detector while hands are missing, the landmark model per tracked hand.
    int live = 0;
    for (Slot& slot : m_slots) {
        if (!slot.m_active) { continue; }
        if (++live > options.m_num_hands) {  // num_hands lowered
            slot.m_active = false;
            slot.m_hand.reset();
        }
    }
    if (live < options.m_num_hands && idle(m_detector_job)) { submit_detector(frame); }
    for (Slot& slot : m_slots) {
        if (slot.m_active && idle(slot.m_landmarks_job)) { submit_landmarks(slot, frame); }
    }

    Hands hands;
    hands.m_frame = m_frame++;
    for (const Slot& slot : m_slots) {
        if (slot.m_active && slot.m_hand) { hands.m_hands[hands.m_count++] = *slot.m_hand; }
    }
    m_latest.write(hands);
    if (!m_queue.push(hands)) { m_dropped.fetch_add(1, std::memory_order_relaxed); }
    return ANIRA_OK;
}

void HandTracker::take_detector(const Options& options) noexcept {
    if (!done(m_detector_job) || m_detector_job.status() == core::TicketStatus::Failed) { return; }
    const bool ok = succeeded(m_detector_job);
    m_detector_job = {};
    if (!ok) { return; }
    std::array<RegionOfInterest, k_max_hands> rois;
    const std::size_t count = decode_palm_rois(m_boxes,
                                               m_detector_scores,
                                               m_anchors,
                                               options.m_min_hand_detection_confidence,
                                               options.m_num_hands,
                                               m_letterbox,
                                               rois);
    for (std::size_t r = 0; r < count; ++r) {
        const bool tracked = std::ranges::any_of(m_slots, [&](const Slot& slot) {
            return slot.m_active &&
                   roi_matches_tracked(rois[r], slot.m_roi, options.m_min_tracking_confidence);
        });
        if (tracked) { continue; }
        for (Slot& slot : m_slots) {
            if (!slot.m_active) {
                slot.m_active = true;
                slot.m_tracking = false;
                slot.m_roi = rois[r];
                slot.m_hand.reset();
                break;
            }
        }
    }
}

void HandTracker::take_landmarks(Slot& slot, int index, const Options& options) noexcept {
    if (!done(slot.m_landmarks_job) || slot.m_landmarks_job.status() == core::TicketStatus::Failed) {
        return;
    }
    const bool ok = succeeded(slot.m_landmarks_job);
    slot.m_landmarks_job = {};
    if (!ok || !slot.m_active) { return; }
    if (slot.m_out_presence < options.m_min_hand_presence_confidence) {
        slot.m_active = false;  // lost: the detector looks for it again
        slot.m_hand.reset();
        return;
    }

    Hand hand;
    hand.m_slot = index;
    hand.m_tracking = slot.m_tracking;
    hand.m_presence = slot.m_out_presence;
    hand.m_handedness_right = slot.m_out_handedness;
    if (slot.m_hand) {
        hand.m_gesture = slot.m_hand->m_gesture;
        hand.m_gesture_score = slot.m_hand->m_gesture_score;
    }
    const auto side = static_cast<float>(k_landmark_input_size);
    const std::array<Vec2, 4> corners{
        Vec2{0.0F, 0.0F}, Vec2{side, 0.0F}, Vec2{side, side}, Vec2{0.0F, side}};
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const Vec2 corner = slot.m_crop_to_frame.transform(corners[i]);
        hand.m_roi_corners[i] = {.m_x = corner.m_x / slot.m_frame_width,
                                 .m_y = corner.m_y / slot.m_frame_height};
    }
    // Crop pixels to frame pixels through the crop of the frame the job ran on.
    std::array<Vec2, k_num_landmarks> pixels;
    std::array<float, k_num_landmarks> z_values{};
    const float z_scale = slot.m_crop_size / side;
    for (std::size_t i = 0; i < k_num_landmarks; ++i) {
        pixels[i] = slot.m_crop_to_frame.transform(
            {.m_x = slot.m_out_landmarks[i * 3], .m_y = slot.m_out_landmarks[(i * 3) + 1]});
        z_values[i] = slot.m_out_landmarks[(i * 3) + 2] * z_scale;
        hand.m_landmarks[i] = {.m_x = pixels[i].m_x / slot.m_frame_width,
                               .m_y = pixels[i].m_y / slot.m_frame_height};
    }
    slot.m_roi = roi_from_landmarks(pixels);
    slot.m_tracking = true;
    slot.m_hand = hand;

    // The gesture of these landmarks, when the embedder is free. As MediaPipe's gesture graph:
    // rotated by the image's region (the whole frame, rotation 0), not the hand's; x and y in
    // units of the longer side, z of the frame's width (MediaPipe's normalised landmarks).
    if (!idle(slot.m_embedder_job)) { return; }
    slot.m_embedder_job = {};
    const float z_to_long_side =
        std::max(slot.m_frame_width, slot.m_frame_height) / slot.m_frame_width;
    for (float& z : z_values) { z *= z_to_long_side; }
    normalize_landmarks_for_embedder(pixels, z_values, 0.0F, slot.m_embed_hand);
    std::array<Vec2, k_num_landmarks> world_points;
    std::array<float, k_num_landmarks> world_z{};
    for (std::size_t i = 0; i < k_num_landmarks; ++i) {
        world_points[i] = {.m_x = slot.m_out_world[i * 3], .m_y = slot.m_out_world[(i * 3) + 1]};
        world_z[i] = slot.m_out_world[(i * 3) + 2];
    }
    normalize_landmarks_for_embedder(world_points, world_z, 0.0F, slot.m_embed_world);
    slot.m_embed_handedness = slot.m_out_handedness;
    const std::array<anira::Tensor, 3> inputs{host(slot.m_embed_hand.data(), {1, 21, 3}),
                                              host(&slot.m_embed_handedness, {1, 1}),
                                              host(slot.m_embed_world.data(), {1, 21, 3})};
    slot.m_embedder_job = slot.m_embedder->submit(inputs);
}

void HandTracker::take_gestures(Slot& slot) noexcept {
    if (succeeded(slot.m_classifier_job)) {
        slot.m_classifier_job = {};
        const auto [gesture, score] = classify_gesture(slot.m_scores);
        if (slot.m_hand) {
            slot.m_hand->m_gesture = gesture;
            slot.m_hand->m_gesture_score = score;
        }
    } else if (done(slot.m_classifier_job)) {
        slot.m_classifier_job = {};
    }
    if (succeeded(slot.m_embedder_job) && idle(slot.m_classifier_job)) {
        slot.m_embedder_job = {};
        slot.m_classify_in = slot.m_embedding;
        const std::array<anira::Tensor, 1> inputs{host(slot.m_classify_in.data(), {1, 128})};
        slot.m_classifier_job = slot.m_classifier->submit(inputs);
    } else if (done(slot.m_embedder_job) && !succeeded(slot.m_embedder_job)) {
        slot.m_embedder_job = {};
    }
}

void HandTracker::submit_detector(const Frame& frame) noexcept {
    m_letterbox = letterbox_geometry(frame);
    sample_to_tensor_rgb(
        frame, letterbox_dest_to_source(m_letterbox), k_detector_input_size, m_detector_input);
    const std::array<anira::Tensor, 1> inputs{host(
        m_detector_input.data(), {1, k_detector_input_size, k_detector_input_size, 3})};
    m_detector_job = m_detector->submit(inputs);
}

void HandTracker::submit_landmarks(Slot& slot, const Frame& frame) noexcept {
    slot.m_crop_to_frame = roi_frame_to_crop(slot.m_roi).inverted();
    slot.m_crop_size = slot.m_roi.m_size;
    slot.m_frame_width = static_cast<float>(frame.m_width);
    slot.m_frame_height = static_cast<float>(frame.m_height);
    sample_to_tensor_rgb(frame, slot.m_crop_to_frame, k_landmark_input_size, slot.m_crop);
    const std::array<anira::Tensor, 1> inputs{
        host(slot.m_crop.data(), {1, k_landmark_input_size, k_landmark_input_size, 3})};
    slot.m_landmarks_job = slot.m_landmarks->submit(inputs);
}

void HandTracker::wait_idle() {
    m_detector_job.wait();
    for (Slot& slot : m_slots) {
        slot.m_landmarks_job.wait();
        slot.m_embedder_job.wait();
        slot.m_classifier_job.wait();
    }
}

Hands HandTracker::run(anira::Context& context,
                       const core::ModelSource& weights,
                       const Frame& frame,
                       Options options) {
    HandTracker tracker(context, weights, options);
    tracker.prepare(std::nullopt);
    // Detector, landmarks, embedder, classifier: each frame takes what the last one started.
    for (int step = 0; step < 5; ++step) {
        tracker.process(frame);
        tracker.wait_idle();
    }
    tracker.process(frame);
    return tracker.latest();
}

}  // namespace anira::models::handtracking
