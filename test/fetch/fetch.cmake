# SPDX-License-Identifier: Apache-2.0

# anira_models_fetch in script mode, one case per run:
#   cmake -DCASE=<case> -DREPO_ROOT=<repo> -DWORK_DIR=<scratch dir> -P fetch.cmake
# Cases ending in _fails must fail (the test is registered WILL_FAIL).

cmake_minimum_required(VERSION 3.25)
include(${REPO_ROOT}/cmake/anira-models.cmake)

set(_data "${REPO_ROOT}/core/test/data")
set(_catalog "${_data}/gain/gain.catalog.json")
file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

function(expect_equal actual expected what)
    if(NOT "${actual}" STREQUAL "${expected}")
        message(FATAL_ERROR "${what}: got '${actual}', expected '${expected}'")
    endif()
endfunction()

if(CASE STREQUAL "local")
    anira_models_fetch(${_catalog} OUT_DIR _dir OUT_FILES _files LOCAL_DIR ${_data}
                           ENGINES onnxruntime)
    expect_equal("${_dir}" "${_data}/gain" "OUT_DIR")
    expect_equal("${_files}" "simple_gain_network_mono.onnx" "OUT_FILES")
    # Only the files of the enabled engines: without ONNX Runtime the weights stay out.
    anira_models_fetch(${_catalog} OUT_DIR _dir OUT_FILES _files LOCAL_DIR ${_data}
                           ENGINES litert)
    expect_equal("${_files}" "" "OUT_FILES without onnxruntime")

elseif(CASE STREQUAL "provider")
    # A file built for one provider comes only where that provider is.
    file(READ "${_catalog}" _json)
    string(JSON _json SET "${_json}" files 0 provider "\"coreml\"")
    file(MAKE_DIRECTORY "${WORK_DIR}/catalog")
    file(WRITE "${WORK_DIR}/catalog/catalog.json" "${_json}")
    anira_models_fetch("${WORK_DIR}/catalog/catalog.json" OUT_DIR _dir OUT_FILES _files
                           LOCAL_DIR ${_data} ENGINES onnxruntime PROVIDERS cuda)
    expect_equal("${_files}" "" "OUT_FILES without Core ML")
    anira_models_fetch("${WORK_DIR}/catalog/catalog.json" OUT_DIR _dir OUT_FILES _files
                           LOCAL_DIR ${_data} ENGINES onnxruntime PROVIDERS coreml)
    expect_equal("${_files}" "simple_gain_network_mono.onnx" "OUT_FILES with Core ML")

elseif(CASE STREQUAL "cache")
    # Files already in the cache are used without a download and stamped once verified.
    set(ANIRA_MODELS_CACHE_DIR "${WORK_DIR}/cache")
    set(_cached "${WORK_DIR}/cache/gain/local")
    file(COPY "${_data}/gain/" DESTINATION "${_cached}")
    anira_models_fetch(${_catalog} OUT_DIR _dir OUT_FILES _files ENGINES onnxruntime)
    expect_equal("${_dir}" "${_cached}" "OUT_DIR")
    if(NOT EXISTS "${_cached}/simple_gain_network_mono.onnx.sha256")
        message(FATAL_ERROR "no stamp after verifying")
    endif()
    anira_models_fetch(${_catalog} OUT_DIR _dir OUT_FILES _files ENGINES onnxruntime)
    expect_equal("${_files}" "simple_gain_network_mono.onnx" "OUT_FILES from the stamps")

elseif(CASE STREQUAL "bad_hash_fails")
    file(READ "${_catalog}" _json)
    string(JSON _json SET "${_json}" files 0 sha256
           "\"0000000000000000000000000000000000000000000000000000000000000000\"")
    file(MAKE_DIRECTORY "${WORK_DIR}/catalog")
    file(WRITE "${WORK_DIR}/catalog/catalog.json" "${_json}")
    anira_models_fetch("${WORK_DIR}/catalog/catalog.json" OUT_DIR _dir LOCAL_DIR ${_data}
                           ENGINES onnxruntime)

elseif(CASE STREQUAL "missing_local_fails")
    anira_models_fetch(${_catalog} OUT_DIR _dir LOCAL_DIR "${WORK_DIR}/empty"
                           ENGINES onnxruntime)

elseif(CASE STREQUAL "download")
    # Network: Silero VAD from Hugging Face into an empty cache, by the repository's catalog.
    set(ANIRA_MODELS_CACHE_DIR "${WORK_DIR}/cache")
    anira_models_fetch(${REPO_ROOT}/models/speech-to-text/silero-vad.catalog.json
                           OUT_DIR _dir OUT_FILES _files ENGINES onnxruntime)
    expect_equal("${_files}" "LICENSE;onnx/silero_vad_16k.onnx" "OUT_FILES")
    if(NOT EXISTS "${_dir}/onnx/silero_vad_16k.onnx")
        message(FATAL_ERROR "not downloaded")
    endif()
    # A damaged file in the cache is downloaded again, not an error.
    file(WRITE "${_dir}/onnx/silero_vad_16k.onnx" "damaged")
    anira_models_fetch(${REPO_ROOT}/models/speech-to-text/silero-vad.catalog.json
                           OUT_DIR _dir ENGINES onnxruntime)
    file(SHA256 "${_dir}/onnx/silero_vad_16k.onnx" _repaired)
    expect_equal("${_repaired}" "165af836d0a0442bfe5d5d91c3e715a995d55ac21ecc1179a96301fe140e7e2c"
                 "the repaired file")

else()
    message(FATAL_ERROR "unknown CASE '${CASE}'")
endif()
