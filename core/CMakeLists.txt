# SPDX-License-Identifier: Apache-2.0

# anira::models::core: where model files come from (ModelSource) and loading their anira
# descriptions from it (anira checks the weights against the descriptions); AudioModel, what every
# audio model's wrapper shares (resampling, the clock, the anira handler), and its helpers;
# AsyncHandler, the jobs of a model written for anira's Async contract (Hard inside until M3).

add_library(anira_models_core STATIC
    src/AsyncHandler.cpp
    src/AudioEffect.cpp
    src/AudioModel.cpp
    src/LogMel.cpp
    src/Model.cpp
    src/ModelSource.cpp
    src/OnnxEngine.cpp
    src/Vocabulary.cpp
    src/Words.cpp
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
# OnnxEngine: ONNX Runtime with a thread count per model, over anira's own ONNX Runtime library
# (anira::onnxruntime, the one anira links), until anira takes a thread count per model. ONNX
# Runtime's symbols stay private to this library.
if(TARGET anira::onnxruntime)
    target_link_libraries(anira_models_core PRIVATE anira::onnxruntime)
    target_compile_definitions(anira_models_core PRIVATE ANIRA_MODELS_WITH_ONNX_ENGINE)
    set_target_properties(anira_models_core PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
endif()
anira_models_license(anira_models_core Apache-2.0)
