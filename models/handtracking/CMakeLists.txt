# SPDX-License-Identifier: Apache-2.0

# anira::models::handtracking: MediaPipe's gesture recognizer. The four models' anira
# descriptions and the contract are the JSON files next to this file, compiled into
# <anira/models/handtracking/Descriptions.h>; the weights are on Hugging Face
# (mediapipe-gesture-recognizer.catalog.json). Decode.h turns pixels into the models' input tensors
# and their output tensors into hands; HandTracker runs the four models in MediaPipe's loop, each
# as jobs (core::AsyncHandler, anira's Async contract from M3).

add_library(anira_models_handtracking STATIC
    src/HandTracker.cpp
)
add_library(anira::models::handtracking ALIAS anira_models_handtracking)

set_target_properties(anira_models_handtracking PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_handtracking PUBLIC cxx_std_20)
target_include_directories(anira_models_handtracking
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_handtracking_descriptions
    NAMESPACE anira::models::handtracking
    HEADER anira/models/handtracking/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/hand_detector.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/hand_landmarks_detector.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/gesture_embedder.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/canned_gesture_classifier.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/handtracking.contract.json)

target_link_libraries(anira_models_handtracking
    PUBLIC anira::models::core anira_models_handtracking_descriptions)

anira_models_license(anira_models_handtracking Apache-2.0)
