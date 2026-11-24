# SPDX-License-Identifier: Apache-2.0

# anira::models::nemotron_asr_streaming: Nemotron 3.5 ASR Streaming 0.6B (int4). NemotronAsr runs an
# encoder step every 560 ms and the greedy RNNT loop (decoder, joint), each a job (core::AsyncHandler),
# over core::LogMel. The descriptions are compiled into
# <anira/models/nemotron_asr_streaming/Descriptions.h>; the weights are on Hugging Face
# (nemotron-3.5-asr-streaming-int4.catalog.json).

add_library(anira_models_nemotron_asr_streaming STATIC
    src/NemotronAsr.cpp
)
add_library(anira::models::nemotron_asr_streaming ALIAS anira_models_nemotron_asr_streaming)

set_target_properties(anira_models_nemotron_asr_streaming PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_nemotron_asr_streaming PUBLIC cxx_std_20)
target_include_directories(anira_models_nemotron_asr_streaming
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
file(GLOB _nemotron_asr_descriptions CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.model.json)
anira_models_embed_text(anira_models_nemotron_asr_streaming_descriptions
    NAMESPACE anira::models::nemotron_asr_streaming
    HEADER anira/models/nemotron_asr_streaming/Descriptions.h
    FILES ${_nemotron_asr_descriptions})
target_link_libraries(anira_models_nemotron_asr_streaming
    PUBLIC anira::models::core tanh::DspBasics
    PRIVATE anira_models_nemotron_asr_streaming_descriptions)
anira_models_license(anira_models_nemotron_asr_streaming Apache-2.0)
