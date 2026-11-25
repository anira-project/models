# SPDX-License-Identifier: Apache-2.0

# anira::models::nemotron_diarization: Nemotron 3 Diarization (streaming Sortformer). Diarizer runs a
# step every 720 ms (core::AsyncHandler) with SpeakerCache, the streaming speaker cache, in between.
# The description is compiled into <anira/models/nemotron_diarization/Descriptions.h>; the weights are
# on Hugging Face (nemotron-3-diarization.catalog.json).

add_library(anira_models_nemotron_diarization STATIC
    src/Diarizer.cpp
    src/SpeakerCache.cpp
)
add_library(anira::models::nemotron_diarization ALIAS anira_models_nemotron_diarization)

set_target_properties(anira_models_nemotron_diarization PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_nemotron_diarization PUBLIC cxx_std_20)
target_include_directories(anira_models_nemotron_diarization
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_nemotron_diarization_descriptions
    NAMESPACE anira::models::nemotron_diarization
    HEADER anira/models/nemotron_diarization/Descriptions.h
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/nemotron-diarization.model.json)
target_link_libraries(anira_models_nemotron_diarization
    PUBLIC anira::models::core tanh::DSP
    PRIVATE anira_models_nemotron_diarization_descriptions)
anira_models_license(anira_models_nemotron_diarization Apache-2.0)
