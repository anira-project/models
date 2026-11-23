# SPDX-License-Identifier: Apache-2.0

# An application that adds anira itself, then anira-models: one anira, the application's.
#   cmake -S test/consumer -B build/consumer -DANIRA_SOURCE_DIR=<anira checkout> [engine options]
cmake_minimum_required(VERSION 3.25)
project(anira-models-consumer LANGUAGES C CXX)

set(ANIRA_SOURCE_DIR "" CACHE PATH "The application's anira checkout")
if(NOT ANIRA_SOURCE_DIR)
    message(FATAL_ERROR "set ANIRA_SOURCE_DIR to an anira checkout")
endif()
add_subdirectory(${ANIRA_SOURCE_DIR} anira)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. anira-models)

