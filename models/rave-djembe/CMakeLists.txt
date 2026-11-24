# SPDX-License-Identifier: Apache-2.0

# anira::models::rave_djembe: RAVE djembe (weights CC BY-NC 4.0). Header-only,
# <anira/models/rave_djembe/RaveDjembe.h>: RaveDjembe (the forward pass, core::AudioEffect),
# RaveDjembeEncoder (core::LatentEncoder) and RaveDjembeDecoder (core::LatentDecoder), with the
# descriptions compiled into <anira/models/rave_djembe/Descriptions.h>; the weights are on Hugging
# Face (rave-djembe.catalog.json).

add_library(anira_models_rave_djembe INTERFACE)
add_library(anira::models::rave_djembe ALIAS anira_models_rave_djembe)
target_include_directories(anira_models_rave_djembe
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_rave_djembe_descriptions
    NAMESPACE anira::models::rave_djembe
    HEADER anira/models/rave_djembe/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-forward.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-forward.contract.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-encoder.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-encoder.contract.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-decoder.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-decoder.contract.json)
target_link_libraries(anira_models_rave_djembe INTERFACE anira::models::core anira_models_rave_djembe_descriptions)
anira_models_license(anira_models_rave_djembe Apache-2.0)
