# SPDX-License-Identifier: Apache-2.0

# anira as a dependency. A process holds one copy of anira (one core, one thread pool), so the
# application's anira is used when it has one; otherwise the commit this repo is tested with is
# fetched, or an installed anira 3 is found first. A different anira than the tested one gives
# a warning: it is not an error, since this repo compiles against whatever anira it gets.
#
#   ANIRA_MODELS_ANIRA_GIT_TAG   the commit FetchContent fetches
#   FETCHCONTENT_SOURCE_DIR_ANIRA    a local anira checkout instead (offline)

include_guard(GLOBAL)
include(FetchContent)

# v3.0.0-alpha.3.
set(ANIRA_MODELS_ANIRA_GIT_TAG c9c146b2555215fbe3d5e81564b0c71558649c42
    CACHE STRING "anira commit fetched when the parent project has no anira")

# ANIRA_VERSION_STRING of the tested commit (anira's generated <anira/abi/build_info.h>).
set(_anira_models_tested_version "3.0.0-alpha.3")

# tanh-lib with the two Apache-2.0 components only: Core and DspBasics (resampling, mel features).
# anira builds its own private copy of tanh-lib's core (anira_tanh_lib, in thl::anira) whatever is
# declared here, so the two never meet and need not be the same commit.
if(NOT TARGET tanh::Core)
    set(TANH_BUILD_CORE ON)
    set(TANH_BUILD_DSP_BASICS ON)
    set(TANH_BUILD_STATE OFF)
    set(TANH_BUILD_DSP OFF)
    set(TANH_BUILD_MODULATION OFF)
    set(TANH_BUILD_AUDIO_IO OFF)
    set(TANH_BUILD_NET OFF)
    set(TANH_WITH_TESTS OFF)
    set(TANH_WITH_EXAMPLES OFF)
    set(TANH_WITH_DOCS OFF)
    set(TANH_WITH_JOURNALD OFF)
    set(TANH_LOG_COMPILED_MAX_LEVEL 4)
    FetchContent_Declare(tanh-lib
        GIT_REPOSITORY https://github.com/tanh-lab/tanh-lib.git
        # tanh-lab/tanh-lib feat/dsp-basics: DspBasics on top of tanh-tooling v0.3.0. Local only so far: build with FETCHCONTENT_SOURCE_DIR_TANH-LIB until it is
        # merged, then pin the merge commit here.
        GIT_TAG 1566356a3d6d490716ab79fd5840a1c60aaf7e26)
    FetchContent_MakeAvailable(tanh-lib)
endif()
if(NOT TARGET tanh::DspBasics)
    message(FATAL_ERROR
        "anira-models needs tanh-lib's DspBasics component (Apache-2.0): the tanh-lib in this "
        "build has none. Set TANH_BUILD_DSP_BASICS ON where tanh-lib is added.")
endif()

if(TARGET anira::anira)
    set(_anira_models_source "the parent project")
else()
    FetchContent_Declare(anira
        GIT_REPOSITORY https://github.com/anira-project/anira.git
        GIT_TAG ${ANIRA_MODELS_ANIRA_GIT_TAG}
        FIND_PACKAGE_ARGS 3)
    FetchContent_MakeAvailable(anira)
    if(anira_FOUND)
        set(_anira_models_source "find_package (${anira_DIR})")
    elseif(FETCHCONTENT_SOURCE_DIR_ANIRA)
        set(_anira_models_source "FETCHCONTENT_SOURCE_DIR_ANIRA (${FETCHCONTENT_SOURCE_DIR_ANIRA})")
    else()
        set(_anira_models_source "FetchContent (${ANIRA_MODELS_ANIRA_GIT_TAG})")
    endif()
endif()

# Reads the version and the ABI of the anira target from its <anira/abi/build_info.h>: in the
# build tree's generated include directory for a subproject, in the include directory of an
# installed anira.
function(_anira_models_read_build_info out_found out_version out_abi)
    set(${out_found} FALSE PARENT_SCOPE)
    get_target_property(_target anira::anira ALIASED_TARGET)
    if(NOT _target)
        set(_target anira::anira)
    endif()
    get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _dirs)
        return()
    endif()
    foreach(_dir IN LISTS _dirs)
        string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" _dir "${_dir}")
        if(_dir MATCHES "\\$<" OR NOT EXISTS "${_dir}/anira/abi/build_info.h")
            continue()
        endif()
        file(STRINGS "${_dir}/anira/abi/build_info.h" _lines
             REGEX "^#define ANIRA_(VERSION_STRING|ABI_MAJOR|ABI_MINOR) ")
        foreach(_line IN LISTS _lines)
            if(_line MATCHES "^#define ANIRA_VERSION_STRING \"(.*)\"")
                set(_version "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "^#define ANIRA_ABI_MAJOR ([0-9]+)")
                set(_major "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "^#define ANIRA_ABI_MINOR ([0-9]+)")
                set(_minor "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        set(${out_found} TRUE PARENT_SCOPE)
        set(${out_version} "${_version}" PARENT_SCOPE)
        set(${out_abi} "${_major}.${_minor}" PARENT_SCOPE)
        return()
    endforeach()
endfunction()

_anira_models_read_build_info(_found _version _abi)
if(NOT _found)
    # anira 2.x has no build_info.h; this repo needs the 3.x API.
    message(FATAL_ERROR
        "anira-models needs anira 3 (tested with ${_anira_models_tested_version}); "
        "the anira from ${_anira_models_source} has no <anira/abi/build_info.h>.")
endif()
if(NOT _version STREQUAL _anira_models_tested_version)
    message(WARNING
        "anira-models is tested with anira ${_anira_models_tested_version}; this build uses "
        "anira ${_version} (ABI ${_abi}) from ${_anira_models_source}. Different anira versions "
        "are not tested.")
else()
    message(STATUS "anira-models: anira ${_version} (ABI ${_abi}) from ${_anira_models_source}")
endif()

# nlohmann_json for this repo's own JSON files (sidecar descriptions, upstream model configs, test
# data); anira's model and contract files are parsed by anira, whose copy is private.
if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        FIND_PACKAGE_ARGS 3.12)
    FetchContent_MakeAvailable(nlohmann_json)
endif()
