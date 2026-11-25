# SPDX-License-Identifier: Apache-2.0

# Model files for applications: optional helpers around a library's models/catalog.json.
#
#   anira_models_fetch(<catalog> OUT_DIR <var> [OUT_FILES <var>] [ENGINES <engine>...]
#                          [PROVIDERS <provider>...] [LOCAL_DIR <dir>])
#       The catalog's files for the enabled engines (and every file without an engine: model and
#       contract descriptions), verified by SHA256. <var> is a directory laid out like the
#       Hugging Face repository, ready for core::ModelSource::directory; OUT_FILES lists the
#       files' paths relative to it. The files come from, in order:
#         LOCAL_DIR / ANIRA_MODELS_LOCAL_DIR   <dir>/<repo>/<path> (offline, CI caches)
#         ANIRA_MODELS_CACHE_DIR               <dir>/<repo>/<revision>/<path>, shared by builds
#         https://huggingface.co/<repo>/resolve/<revision>/<path>   into the cache
#       ENGINES overrides the enabled engines (anira's JSON names: onnxruntime, litert, tflite,
#       libtorch, executorch, coreml); by default an engine is enabled when ANIRA_WITH_<ENGINE> is on or
#       the target anira::<engine> exists. A file built for one provider ("provider": "coreml",
#       an ExecuTorch program lowered to Core ML) comes only where that provider is: PROVIDERS
#       overrides the default, coreml on macOS and iOS.
#
#   anira_models_embed(<target> NAMESPACE <ns> SOURCE_DIR <dir> FILES <path>...
#                          [FUNCTION <name>] [HEADER <name>])
#       A static library <target> holding the files as byte arrays and a function
#       `anira::models::core::ModelSource <ns>::<name>()` (default embedded_models) that
#       returns them as a bytes source under their relative paths. <target> links
#       anira::models::core; the function is declared in <HEADER> (default EmbeddedModels.h)
#       on its include path.
#
#   anira_models_embed_text(<target> NAMESPACE <ns> HEADER <path/Name.h> FILES <file>...)
#       An INTERFACE target <target> whose generated header holds each file's text as
#       `inline constexpr std::string_view k_<file name as a C identifier>` in <ns>: the model and
#       contract descriptions a library compiles in (hand_detector.model.json -> k_hand_detector_model_json).

include_guard(GLOBAL)

set(ANIRA_MODELS_LOCAL_DIR "" CACHE PATH
    "Take model files from <dir>/<repo>/<path> instead of the cache or a download")
if(DEFINED ENV{LOCALAPPDATA})
    set(_anira_models_default_cache "$ENV{LOCALAPPDATA}/anira-models")
else()
    set(_anira_models_default_cache "$ENV{HOME}/.cache/anira-models")
endif()
set(ANIRA_MODELS_CACHE_DIR "${_anira_models_default_cache}" CACHE PATH
    "Where downloaded model files are kept, shared by every build and project")
unset(_anira_models_default_cache)

function(_anira_models_enabled_engines out)
    # Here, not at file scope: a project that includes this file from a parent directory calls
    # the functions where a directory variable of models' is not visible.
    set(_enabled "")
    foreach(_engine IN ITEMS onnxruntime litert tflite libtorch executorch coreml)
        string(TOUPPER "${_engine}" _upper)
        if(ANIRA_WITH_${_upper} OR TARGET anira::${_engine})
            list(APPEND _enabled ${_engine})
        endif()
    endforeach()
    set(${out} ${_enabled} PARENT_SCOPE)
endfunction()

# The file's engines: "engine": "<name>" or "engines": ["<name>", …]; none = always needed.
function(_anira_models_file_engines json index out)
    set(_engines "")
    string(JSON _one ERROR_VARIABLE _no_one GET "${json}" files ${index} engine)
    if(NOT _no_one)
        list(APPEND _engines "${_one}")
    endif()
    string(JSON _count ERROR_VARIABLE _no_list LENGTH "${json}" files ${index} engines)
    if(NOT _no_list AND _count GREATER 0)
        math(EXPR _last "${_count} - 1")
        foreach(_i RANGE ${_last})
            string(JSON _engine GET "${json}" files ${index} engines ${_i})
            list(APPEND _engines "${_engine}")
        endforeach()
    endif()
    set(${out} ${_engines} PARENT_SCOPE)
endfunction()

function(_anira_models_download url file sha256)
    get_filename_component(_dir "${file}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    set(_part "${file}.part")
    foreach(_attempt 1 2)
        file(DOWNLOAD "${url}" "${_part}" EXPECTED_HASH SHA256=${sha256} TLS_VERIFY ON
             STATUS _status)
        list(GET _status 0 _code)
        if(_code EQUAL 0)
            file(RENAME "${_part}" "${file}")
            return()
        endif()
        file(REMOVE "${_part}")
        list(GET _status 1 _message)
        if(_attempt EQUAL 1)
            message(STATUS "anira_models_fetch: ${url}: ${_message}; retrying")
            execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 2)
        endif()
    endforeach()
    message(FATAL_ERROR "anira_models_fetch: downloading ${url} failed: ${_message}")
endfunction()

function(anira_models_fetch catalog)
    cmake_parse_arguments(PARSE_ARGV 1 _arg "" "OUT_DIR;OUT_FILES;LOCAL_DIR" "ENGINES;PROVIDERS")
    if(NOT _arg_OUT_DIR)
        message(FATAL_ERROR "anira_models_fetch: OUT_DIR is required")
    endif()
    if(NOT IS_ABSOLUTE "${catalog}")
        get_filename_component(catalog "${catalog}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
    if(NOT CMAKE_SCRIPT_MODE_FILE)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${catalog}")
    endif()
    file(READ "${catalog}" _json)
    string(JSON _repo GET "${_json}" repo)
    string(JSON _revision GET "${_json}" revision)
    string(JSON _count LENGTH "${_json}" files)

    if(DEFINED _arg_ENGINES)
        set(_enabled ${_arg_ENGINES})
    else()
        _anira_models_enabled_engines(_enabled)
    endif()
    if(DEFINED _arg_PROVIDERS)
        set(_providers ${_arg_PROVIDERS})
    elseif(TANH_OPERATING_SYSTEM MATCHES "^(macOS|iOS)$")
        set(_providers coreml)
    else()
        set(_providers "")
    endif()
    set(_local "${_arg_LOCAL_DIR}")
    if(NOT _local)
        set(_local "${ANIRA_MODELS_LOCAL_DIR}")
    endif()
    if(_local)
        set(_root "${_local}/${_repo}")
        set(_from "${_root}")
    else()
        set(_root "${ANIRA_MODELS_CACHE_DIR}/${_repo}/${_revision}")
        set(_from "huggingface.co/${_repo}@${_revision}, cached in ${_root}")
    endif()

    set(_files "")
    math(EXPR _last "${_count} - 1")
    foreach(_i RANGE ${_last})
        string(JSON _path GET "${_json}" files ${_i} path)
        string(JSON _sha256 GET "${_json}" files ${_i} sha256)
        string(TOLOWER "${_sha256}" _sha256)
        _anira_models_file_engines("${_json}" ${_i} _file_engines)
        if(_file_engines)
            set(_wanted FALSE)
            foreach(_engine IN LISTS _file_engines)
                if(_engine IN_LIST _enabled)
                    set(_wanted TRUE)
                endif()
            endforeach()
            if(NOT _wanted)
                continue()
            endif()
        endif()
        string(JSON _provider ERROR_VARIABLE _no_provider GET "${_json}" files ${_i} provider)
        if(NOT _no_provider AND NOT _provider IN_LIST _providers)
            continue()
        endif()

        set(_file "${_root}/${_path}")
        set(_stamp "${_file}.sha256")
        if(_local)
            if(NOT EXISTS "${_file}")
                message(FATAL_ERROR "anira_models_fetch: ${_file} is missing (${catalog})")
            endif()
            file(SHA256 "${_file}" _actual)
        else()
            # The stamp records the hash a cached file was verified against when it was stored.
            set(_actual "")
            if(EXISTS "${_file}" AND EXISTS "${_stamp}" AND NOT "${_file}" IS_NEWER_THAN "${_stamp}")
                file(READ "${_stamp}" _actual)
            elseif(EXISTS "${_file}")
                file(SHA256 "${_file}" _actual)
            endif()
            if(NOT _actual STREQUAL _sha256)
                # Missing, damaged, or not what the catalog pins: download it again (verified on the way).
                file(REMOVE "${_file}" "${_stamp}")
                _anira_models_download(
                    "https://huggingface.co/${_repo}/resolve/${_revision}/${_path}" "${_file}" "${_sha256}")
                set(_actual "${_sha256}")
            endif()
            if(NOT EXISTS "${_stamp}")
                file(WRITE "${_stamp}" "${_actual}")
            endif()
        endif()
        if(NOT _actual STREQUAL _sha256)
            message(FATAL_ERROR
                "anira_models_fetch: ${_file} has SHA256 ${_actual}, the catalog says ${_sha256} "
                "(${catalog})")
        endif()
        list(APPEND _files "${_path}")
    endforeach()

    list(LENGTH _files _num)
    message(STATUS "anira_models_fetch: ${_repo}: ${_num} files (${_enabled}) from ${_from}")
    set(${_arg_OUT_DIR} "${_root}" PARENT_SCOPE)
    if(_arg_OUT_FILES)
        set(${_arg_OUT_FILES} ${_files} PARENT_SCOPE)
    endif()
endfunction()

function(anira_models_embed target)
    cmake_parse_arguments(PARSE_ARGV 1 _arg "" "NAMESPACE;SOURCE_DIR;FUNCTION;HEADER" "FILES")
    foreach(_required NAMESPACE SOURCE_DIR FILES)
        if(NOT _arg_${_required})
            message(FATAL_ERROR "anira_models_embed(${target}): ${_required} is required")
        endif()
    endforeach()
    if(NOT _arg_FUNCTION)
        set(_arg_FUNCTION embedded_models)
    endif()
    if(NOT _arg_HEADER)
        set(_arg_HEADER EmbeddedModels.h)
    endif()
    include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tanh/binary-data.cmake)

    # tanh_add_binary_data names each array after the file's base name, which repeats across
    # directories ("tiny-en/encoder.onnx", "small-en/encoder.onnx"), so each file is staged under
    # its index: "f3_encoder.onnx" -> array f3_encoder_onnx.
    set(_stage "${CMAKE_CURRENT_BINARY_DIR}/${target}_stage")
    file(MAKE_DIRECTORY "${_stage}")
    set(_staged "")
    set(_entries "")
    set(_index 0)
    foreach(_path IN LISTS _arg_FILES)
        get_filename_component(_name "${_path}" NAME)
        set(_flat "f${_index}_${_name}")
        math(EXPR _index "${_index} + 1")
        file(COPY_FILE "${_arg_SOURCE_DIR}/${_path}" "${_stage}/${_flat}" ONLY_IF_DIFFERENT)
        list(APPEND _staged "${_stage}/${_flat}")
        string(MAKE_C_IDENTIFIER "${_flat}" _symbol)
        string(APPEND _entries
            "        {\"${_path}\", std::span(reinterpret_cast<const std::byte*>(data::${_symbol}), "
            "static_cast<std::size_t>(data::${_symbol}Size))},\n")
    endforeach()

    tanh_add_binary_data(${target}_data
        NAMESPACE ${_arg_NAMESPACE}::data
        HEADER_NAME ${target}_data.h
        SOURCES ${_staged})

    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/${target}_generated")
    file(WRITE "${_generated}/include/${_arg_HEADER}.in"
"// Generated by anira_models_embed(${target}).
#pragma once

#include <anira/models/core/ModelSource.h>

namespace ${_arg_NAMESPACE} {

// The embedded model files as a bytes source, under their paths relative to the model root.
anira::models::core::ModelSource ${_arg_FUNCTION}();

}  // namespace ${_arg_NAMESPACE}
")
    configure_file("${_generated}/include/${_arg_HEADER}.in" "${_generated}/include/${_arg_HEADER}" COPYONLY)
    file(WRITE "${_generated}/${target}.cpp.in"
"// Generated by anira_models_embed(${target}).
#include \"${_arg_HEADER}\"
#include \"${target}_data.h\"

#include <cstddef>
#include <span>

namespace ${_arg_NAMESPACE} {

anira::models::core::ModelSource ${_arg_FUNCTION}() {
    return anira::models::core::ModelSource::bytes({
${_entries}    });
}

}  // namespace ${_arg_NAMESPACE}
")
    configure_file("${_generated}/${target}.cpp.in" "${_generated}/${target}.cpp" COPYONLY)

    add_library(${target} STATIC "${_generated}/${target}.cpp")
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_features(${target} PUBLIC cxx_std_20)
    target_include_directories(${target} PUBLIC "${_generated}/include")
    target_link_libraries(${target} PUBLIC anira::models::core PRIVATE ${target}_data)
endfunction()

function(anira_models_embed_text target)
    cmake_parse_arguments(PARSE_ARGV 1 _arg "" "NAMESPACE;HEADER" "FILES")
    foreach(_required NAMESPACE HEADER FILES)
        if(NOT _arg_${_required})
            message(FATAL_ERROR "anira_models_embed_text(${target}): ${_required} is required")
        endif()
    endforeach()
    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}_generated")
    set(_content "// Generated by anira_models_embed_text(${target}).\n#pragma once\n\n")
    string(APPEND _content "#include <string_view>\n\nnamespace ${_arg_NAMESPACE} {\n")
    foreach(_file IN LISTS _arg_FILES)
        get_filename_component(_name "${_file}" NAME)
        string(MAKE_C_IDENTIFIER "${_name}" _id)
        file(READ "${_file}" _text)
        string(APPEND _content "\ninline constexpr std::string_view k_${_id} = R\"anira(${_text})anira\";\n")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_file}")
    endforeach()
    string(APPEND _content "\n}  // namespace ${_arg_NAMESPACE}\n")
    file(WRITE "${_dir}/${_arg_HEADER}.in" "${_content}")
    configure_file("${_dir}/${_arg_HEADER}.in" "${_dir}/${_arg_HEADER}" COPYONLY)
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE "${_dir}")
endfunction()
