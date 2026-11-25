# SPDX-License-Identifier: Apache-2.0

# anira::models::moonshine_streaming: Moonshine AI's streaming speech to text (tiny and small,
# English and German), two fixed-size graphs per model written for anira's Async contract (run
# through core::AsyncHandler until M3). The descriptions are compiled into
# <anira/models/moonshine_streaming/Descriptions.h>; the weights and tokenizers are on Hugging Face
# (moonshine-streaming.catalog.json).

add_library(anira_models_moonshine_streaming STATIC
    src/Moonshine.cpp
    src/SegmentDecoder.cpp
)
add_library(anira::models::moonshine_streaming ALIAS anira_models_moonshine_streaming)

set_target_properties(anira_models_moonshine_streaming PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_moonshine_streaming PUBLIC cxx_std_20)
target_include_directories(anira_models_moonshine_streaming
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
file(GLOB _moonshine_descriptions CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.model.json)
anira_models_embed_text(anira_models_moonshine_streaming_descriptions
    NAMESPACE anira::models::moonshine_streaming
    HEADER anira/models/moonshine_streaming/Descriptions.h
    FILES ${_moonshine_descriptions})

target_link_libraries(anira_models_moonshine_streaming
    PUBLIC anira::models::core tanh::DSP
    PRIVATE anira_models_moonshine_streaming_descriptions nlohmann_json::nlohmann_json)
anira_models_license(anira_models_moonshine_streaming Apache-2.0)
