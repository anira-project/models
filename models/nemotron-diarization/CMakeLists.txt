# SPDX-License-Identifier: Apache-2.0

# anira::models::nemotron_diarization: Nemotron 3 diarization. SpeakerCache is the streaming
# speaker cache between the model's calls. The weights are on Hugging Face
# (nemotron-3-diarization.catalog.json); the descriptions and the handler-shaped wrapper follow.

add_library(anira_models_nemotron_diarization STATIC
    src/SpeakerCache.cpp
)
add_library(anira::models::nemotron_diarization ALIAS anira_models_nemotron_diarization)

set_target_properties(anira_models_nemotron_diarization PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_nemotron_diarization PUBLIC cxx_std_20)
target_include_directories(anira_models_nemotron_diarization
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(anira_models_nemotron_diarization PUBLIC anira::models::core)
anira_models_license(anira_models_nemotron_diarization Apache-2.0)
