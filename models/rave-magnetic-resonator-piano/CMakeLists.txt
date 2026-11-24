# SPDX-License-Identifier: Apache-2.0

# anira::models::rave_magnetic_resonator_piano: RAVE magnetic resonator piano (RAVE, weights CC BY-NC 4.0), its streaming forward pass (encode and decode) as an effect. Header-only: RaveMagneticResonatorPiano (<anira/models/rave_magnetic_resonator_piano/RaveMagneticResonatorPiano.h>) on
# core::AudioEffect, with the description compiled into <anira/models/rave_magnetic_resonator_piano/Descriptions.h>; the
# weights are on Hugging Face (rave-magnetic-resonator-piano.catalog.json).

add_library(anira_models_rave_magnetic_resonator_piano INTERFACE)
add_library(anira::models::rave_magnetic_resonator_piano ALIAS anira_models_rave_magnetic_resonator_piano)
target_include_directories(anira_models_rave_magnetic_resonator_piano
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_rave_magnetic_resonator_piano_descriptions
    NAMESPACE anira::models::rave_magnetic_resonator_piano
    HEADER anira/models/rave_magnetic_resonator_piano/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-magnetic-resonator-piano-forward.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/rave-magnetic-resonator-piano-forward.contract.json)
target_link_libraries(anira_models_rave_magnetic_resonator_piano INTERFACE anira::models::core anira_models_rave_magnetic_resonator_piano_descriptions)
anira_models_license(anira_models_rave_magnetic_resonator_piano Apache-2.0)
