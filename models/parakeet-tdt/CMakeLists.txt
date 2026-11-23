# SPDX-License-Identifier: Apache-2.0

# anira::models::parakeet_tdt: Parakeet TDT 0.6B v3. TdtGreedyDecoder decodes the joint network's
# tokens and durations, Vocabulary turns tokens into text, Words gives the words with their positions.
# Parakeet runs a segment through the preprocessor, the encoder (in the smallest of four windows that
# holds it) and the TDT search (core::AsyncHandler). The weights are on Hugging Face
# (parakeet-tdt-0.6b-v3.catalog.json).

add_library(anira_models_parakeet_tdt STATIC
    src/Parakeet.cpp
    src/TdtGreedyDecoder.cpp
    src/Vocabulary.cpp
    src/Words.cpp
)
add_library(anira::models::parakeet_tdt ALIAS anira_models_parakeet_tdt)

set_target_properties(anira_models_parakeet_tdt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_parakeet_tdt PUBLIC cxx_std_20)
target_include_directories(anira_models_parakeet_tdt
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
file(GLOB _parakeet_descriptions CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.model.json)
anira_models_embed_text(anira_models_parakeet_tdt_descriptions
    NAMESPACE anira::models::parakeet_tdt
    HEADER anira/models/parakeet_tdt/Descriptions.h
    FILES ${_parakeet_descriptions})
target_link_libraries(anira_models_parakeet_tdt
    PUBLIC anira::models::core
    PRIVATE anira_models_parakeet_tdt_descriptions tanh::DspBasics)
anira_models_license(anira_models_parakeet_tdt Apache-2.0)
