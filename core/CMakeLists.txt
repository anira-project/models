# SPDX-License-Identifier: Apache-2.0

# anira::models::core: where model files come from (ModelSource) and loading their anira
# descriptions from it (anira checks the weights against the descriptions); AudioModel, what every
# audio model's wrapper shares (resampling, the clock, the anira handler), and its helpers;
# AsyncHandler, the jobs of a model written for anira's Async contract (Hard inside until M3).

add_library(anira_models_core STATIC
    src/AsyncHandler.cpp
    src/AudioModel.cpp
    src/Model.cpp
    src/ModelSource.cpp
)
add_library(anira::models::core ALIAS anira_models_core)

set_target_properties(anira_models_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(anira_models_core PUBLIC cxx_std_20)
target_include_directories(anira_models_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(anira_models_core PUBLIC anira::anira tanh::DspBasics
    PRIVATE nlohmann_json::nlohmann_json)
anira_models_license(anira_models_core Apache-2.0)
