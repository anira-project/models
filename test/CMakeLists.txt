# SPDX-License-Identifier: Apache-2.0

include(${PROJECT_SOURCE_DIR}/cmake/tanh/test-deps.cmake)
tanh_fetch_googletest(VERSION v1.17.0)

# The gain model as a library would carry it: its description compiled in, its weights from the
# test-models catalog (fetched and embedded, as an application would).
anira_models_embed_text(models_test_gain_descriptions
    NAMESPACE models_test
    HEADER GainDescriptions.h
    FILES
        ${PROJECT_SOURCE_DIR}/core/test/data/gain/gain.model.json
        ${PROJECT_SOURCE_DIR}/core/test/data/gain/gain.contract.json)
anira_models_fetch(${PROJECT_SOURCE_DIR}/test-models/simple-gain/test-models.catalog.json
    OUT_DIR _gain_dir
    OUT_FILES _gain_files
    ENGINES onnxruntime)
# The stateful accumulator too (the same HF repository and revision: the same directory).
anira_models_fetch(${PROJECT_SOURCE_DIR}/test-models/stateful-accumulator/test-models.catalog.json
    OUT_DIR _accumulator_dir
    ENGINES onnxruntime)
anira_models_embed(models_test_gain_models
    NAMESPACE models_test
    SOURCE_DIR ${_gain_dir}
    FILES ${_gain_files}
    FUNCTION gain_models
    HEADER GainModels.h)

add_executable(anira-models-tests
    test_anira.cpp
    test_Descriptions.cpp
    test_Weights.cpp
    ${PROJECT_SOURCE_DIR}/core/test/test_AsyncHandler.cpp
    ${PROJECT_SOURCE_DIR}/core/test/test_Core.cpp
    ${PROJECT_SOURCE_DIR}/core/test/test_Embed.cpp
)
target_compile_definitions(anira-models-tests
    PRIVATE MODELS_TEST_GAIN_DIR="${_gain_dir}"
            ANIRA_MODELS_SOURCE_DIR="${PROJECT_SOURCE_DIR}/models"
            ANIRA_TEST_MODELS_SOURCE_DIR="${PROJECT_SOURCE_DIR}/test-models")
target_link_libraries(anira-models-tests
    PRIVATE anira::models::core models_test_gain_models models_test_gain_descriptions
            anira::anira nlohmann_json::nlohmann_json gtest_main)

if(TARGET anira::models::handtracking)
    target_sources(anira-models-tests PRIVATE
        ${PROJECT_SOURCE_DIR}/models/handtracking/test/test_Decode.cpp
        ${PROJECT_SOURCE_DIR}/models/handtracking/test/test_Models.cpp
        ${PROJECT_SOURCE_DIR}/models/handtracking/test/test_HandTracker.cpp)
    target_link_libraries(anira-models-tests PRIVATE anira::models::handtracking)
endif()
if(TARGET anira::models::silero_vad)
    target_sources(anira-models-tests PRIVATE
        ${PROJECT_SOURCE_DIR}/models/silero-vad/test/test_Vad.cpp)
    target_link_libraries(anira-models-tests PRIVATE anira::models::silero_vad)
endif()
foreach(_model parakeet_tdt nemotron_diarization)
    if(TARGET anira::models::${_model})
        target_link_libraries(anira-models-tests PRIVATE anira::models::${_model})
    endif()
endforeach()
gtest_discover_tests(anira-models-tests)

# anira_models_fetch in script mode: local directory, cache, a bad hash, a missing file.
foreach(_case local provider cache bad_hash_fails missing_local_fails)
    add_test(NAME Fetch.${_case}
        COMMAND ${CMAKE_COMMAND} -DCASE=${_case} -DREPO_ROOT=${PROJECT_SOURCE_DIR}
                -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/fetch/${_case}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/fetch/fetch.cmake)
    if(_case MATCHES "_fails$")
        set_tests_properties(Fetch.${_case} PROPERTIES WILL_FAIL TRUE)
    endif()
endforeach()

# Network: Silero VAD (2 MB and its licence) from Hugging Face into an empty cache, by its catalog.
option(ANIRA_MODELS_TEST_DOWNLOAD "Add the test that downloads from Hugging Face" OFF)
if(ANIRA_MODELS_TEST_DOWNLOAD)
    add_test(NAME Fetch.download
        COMMAND ${CMAKE_COMMAND} -DCASE=download -DREPO_ROOT=${PROJECT_SOURCE_DIR}
                -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/fetch/download
                -P ${CMAKE_CURRENT_SOURCE_DIR}/fetch/fetch.cmake)
endif()

# The licence check: an Apache-2.0 target that links an AGPL one fails configure.
foreach(_case clean agpl_fails)
    set(_link OFF)
    if(_case STREQUAL "agpl_fails")
        set(_link ON)
    endif()
    add_test(NAME Licenses.${_case}
        COMMAND ${CMAKE_COMMAND} -S ${CMAKE_CURRENT_SOURCE_DIR}/licenses
                -B ${CMAKE_CURRENT_BINARY_DIR}/licenses/${_case} -DLINK_AGPL=${_link})
endforeach()
set_tests_properties(Licenses.agpl_fails PROPERTIES WILL_FAIL TRUE)
