# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25)

# Version from the nearest vX.Y.Z tag (0.0.0 without one) — before project().
include(cmake/tanh/git-version.cmake)
tanh_git_version(${CMAKE_CURRENT_SOURCE_DIR})

project(anira-models VERSION ${TANH_VERSION_SHORT} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ------------------------------------------------------------------------------
# Options. The inference engines are anira's (ANIRA_WITH_*): this repo never sets them.
# ------------------------------------------------------------------------------
option(ANIRA_MODELS_WITH_HANDTRACKING "Build anira::models::handtracking" ${PROJECT_IS_TOP_LEVEL})
option(ANIRA_MODELS_WITH_SILERO_VAD "Build anira::models::silero_vad" ${PROJECT_IS_TOP_LEVEL})
option(ANIRA_MODELS_WITH_PARAKEET_TDT "Build anira::models::parakeet_tdt" ${PROJECT_IS_TOP_LEVEL})
option(ANIRA_MODELS_WITH_NEMOTRON_DIARIZATION "Build anira::models::nemotron_diarization" ${PROJECT_IS_TOP_LEVEL})
option(ANIRA_MODELS_WITH_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(ANIRA_MODELS_LINK_COMPILE_COMMANDS
       "Link build/compile_commands.json, which clangd reads, to this tree's (top-level builds only)" OFF)

# Platform axes: TANH_OPERATING_SYSTEM, TANH_BINARY_FORMAT. Every OS decision keys on these.
include(cmake/tanh/platform.cmake)

if(ANIRA_MODELS_LINK_COMPILE_COMMANDS)
    include(cmake/tanh/compile-commands.cmake)
    tanh_link_compile_commands()
endif()

message(STATUS "anira-models ${TANH_VERSION_FULL} on ${TANH_OPERATING_SYSTEM} (${TANH_BINARY_FORMAT})")

# ------------------------------------------------------------------------------
# anira: the application's when it has one, else the pinned commit (cmake/dependencies.cmake).
# ------------------------------------------------------------------------------
include(cmake/dependencies.cmake)

# anira_models_fetch / anira_models_embed for applications (cmake/anira-models.cmake).
include(cmake/anira-models.cmake)
# One licence per target; an Apache-2.0 target never links an AGPL one (cmake/licenses.cmake).
include(cmake/licenses.cmake)

# ------------------------------------------------------------------------------
# The libraries: core (always), then one add_subdirectory per model behind
# ANIRA_MODELS_WITH_<MODEL>.
# ------------------------------------------------------------------------------
add_subdirectory(core)
if(ANIRA_MODELS_WITH_HANDTRACKING)
    add_subdirectory(models/handtracking)
endif()
if(ANIRA_MODELS_WITH_SILERO_VAD)
    add_subdirectory(models/silero-vad)
endif()
if(ANIRA_MODELS_WITH_PARAKEET_TDT)
    add_subdirectory(models/parakeet-tdt)
endif()
if(ANIRA_MODELS_WITH_NEMOTRON_DIARIZATION)
    add_subdirectory(models/nemotron-diarization)
endif()

anira_models_check_licenses()

if(ANIRA_MODELS_WITH_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
