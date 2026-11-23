# SPDX-License-Identifier: Apache-2.0

# anira::models::silero_vad: Silero VAD. Vad is the handler-shaped wrapper on core::AudioModel
# (speech probability aligned in process, speech spans from pop); VadSegmenter turns the
# probabilities into spans. The description and contract are compiled into
# <anira/models/silero_vad/Descriptions.h>; the weights are on Hugging Face (silero-vad.catalog.json).

add_library(anira_models_silero_vad STATIC
    src/Vad.cpp
    src/VadSegmenter.cpp
)
add_library(anira::models::silero_vad ALIAS anira_models_silero_vad)

set_target_properties(anira_models_silero_vad PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_silero_vad PUBLIC cxx_std_20)
target_include_directories(anira_models_silero_vad
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_silero_vad_descriptions
    NAMESPACE anira::models::silero_vad
    HEADER anira/models/silero_vad/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/silero-vad.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/silero-vad.contract.json)

target_link_libraries(anira_models_silero_vad
    PUBLIC anira::models::core
    PRIVATE anira_models_silero_vad_descriptions)
anira_models_license(anira_models_silero_vad Apache-2.0)
