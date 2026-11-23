# SPDX-License-Identifier: Apache-2.0

# anira_models_check_licenses on its own: configures cleanly with LINK_AGPL OFF, fails with it ON
# (an Apache-2.0 target that links an AGPL one through another target).
#   cmake -S test/licenses -B <dir> [-DLINK_AGPL=ON]
cmake_minimum_required(VERSION 3.25)
project(anira-models-licenses LANGUAGES NONE)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/licenses.cmake)

add_library(apache_lib INTERFACE)
add_library(apache_dep INTERFACE)
add_library(agpl_model INTERFACE)
anira_models_license(apache_lib Apache-2.0)
anira_models_license(apache_dep Apache-2.0)
anira_models_license(agpl_model AGPL-3.0-only)
target_link_libraries(apache_lib INTERFACE apache_dep)
target_link_libraries(agpl_model INTERFACE apache_lib)  # AGPL may use Apache-2.0
if(LINK_AGPL)
    target_link_libraries(apache_dep INTERFACE agpl_model)
endif()
anira_models_check_licenses()
