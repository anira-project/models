# SPDX-License-Identifier: Apache-2.0

# anira::models::rave_djembe: RAVE djembe (RAVE, weights CC BY-NC 4.0), its streaming forward pass (encode and decode) as an effect. Header-only: RaveDjembe (<anira/models/rave_djembe/RaveDjembe.h>) on
# core::AudioEffect, with the description compiled into <anira/models/rave_djembe/Descriptions.h>; the
# weights are on Hugging Face (rave-djembe.catalog.json).

add_library(anira_models_rave_djembe INTERFACE)
add_library(anira::models::rave_djembe ALIAS anira_models_rave_djembe)
target_include_directories(anira_models_rave_djembe
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_rave_djembe_descriptions
    NAMESPACE anira::models::rave_djembe
    HEADER anira/models/rave_djembe/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-forward.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-djembe-forward.contract.json)
target_link_libraries(anira_models_rave_djembe INTERFACE anira::models::core anira_models_rave_djembe_descriptions)
anira_models_license(anira_models_rave_djembe Apache-2.0)
