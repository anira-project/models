# SPDX-License-Identifier: Apache-2.0

# anira::models::steerable_nafx: steerable-nafx (faressc/steerable-nafx), a dilated convolutional effect in three sizes. Header-only: SteerableNafx (<anira/models/steerable_nafx/SteerableNafx.h>) on
# core::AudioEffect, with the description compiled into <anira/models/steerable_nafx/Descriptions.h>; the
# weights are on Hugging Face (steerable-nafx.catalog.json).

add_library(anira_models_steerable_nafx INTERFACE)
add_library(anira::models::steerable_nafx ALIAS anira_models_steerable_nafx)
target_include_directories(anira_models_steerable_nafx
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_steerable_nafx_descriptions
    NAMESPACE anira::models::steerable_nafx
    HEADER anira/models/steerable_nafx/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx.contract.json
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx-3-blocks.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx-3-blocks.contract.json
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx-2-blocks.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/steerable-nafx-2-blocks.contract.json)
target_link_libraries(anira_models_steerable_nafx INTERFACE anira::models::core anira_models_steerable_nafx_descriptions)
anira_models_license(anira_models_steerable_nafx Apache-2.0)
