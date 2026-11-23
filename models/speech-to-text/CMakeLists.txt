# SPDX-License-Identifier: Apache-2.0

# anira::models::stt: speech to text.
#
# The building blocks below carry no anira and no model: copied unchanged in substance from
# anira-project/speech-to-text 277238a (engines/ and app/; the same files as anira-cam/src/stt),
# with the namespace anira::models::stt, the includes under anira/models/stt/ and the
# real-time macro of tanh-lib. The models (JSON first) and their handler-shaped wrappers come on top:
# Vad (Silero VAD) on core::AudioModel.

add_library(anira_models_stt STATIC
    src/common/TdtGreedyDecoder.cpp
    src/common/VadSegmenter.cpp
    src/common/Vocabulary.cpp
    src/common/Words.cpp
    src/diarization/SpeakerCache.cpp
    src/live/SpeakerTrack.cpp
    src/live/TranscriptMerger.cpp
    src/vad/Vad.cpp
)
add_library(anira::models::stt ALIAS anira_models_stt)

set_target_properties(anira_models_stt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_stt PUBLIC cxx_std_20)
target_include_directories(anira_models_stt
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# The models' anira descriptions, compiled in (<anira/models/stt/Descriptions.h>).
anira_models_embed_text(anira_models_stt_descriptions
    NAMESPACE anira::models::stt
    HEADER anira/models/stt/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/silero-vad.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/silero-vad.contract.json)

target_link_libraries(anira_models_stt
    PUBLIC anira::models::core tanh::DspBasics
    PRIVATE anira_models_stt_descriptions)
anira_models_license(anira_models_stt Apache-2.0)
