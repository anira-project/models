# SPDX-License-Identifier: Apache-2.0

# anira::models::rave_funk_drum: RAVE funk drum (RAVE, weights CC BY-NC 4.0), its streaming forward pass (encode and decode) as an effect. Header-only: RaveFunkDrum (<anira/models/rave_funk_drum/RaveFunkDrum.h>) on
# core::AudioEffect, with the description compiled into <anira/models/rave_funk_drum/Descriptions.h>; the
# weights are on Hugging Face (rave-funk-drum.catalog.json).

add_library(anira_models_rave_funk_drum INTERFACE)
add_library(anira::models::rave_funk_drum ALIAS anira_models_rave_funk_drum)
target_include_directories(anira_models_rave_funk_drum
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_rave_funk_drum_descriptions
    NAMESPACE anira::models::rave_funk_drum
    HEADER anira/models/rave_funk_drum/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-funk-drum-forward.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-funk-drum-forward.contract.json)
target_link_libraries(anira_models_rave_funk_drum INTERFACE anira::models::core anira_models_rave_funk_drum_descriptions)
anira_models_license(anira_models_rave_funk_drum Apache-2.0)
