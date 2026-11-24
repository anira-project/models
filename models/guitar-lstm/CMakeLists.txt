# SPDX-License-Identifier: Apache-2.0

# anira::models::guitar_lstm: GuitarLSTM (faressc/GuitarLSTM), a hybrid convolutional LSTM amp model. Header-only: GuitarLstm (<anira/models/guitar_lstm/GuitarLstm.h>) on
# core::AudioEffect, with the description compiled into <anira/models/guitar_lstm/Descriptions.h>; the
# weights are on Hugging Face (guitar-lstm.catalog.json).

add_library(anira_models_guitar_lstm INTERFACE)
add_library(anira::models::guitar_lstm ALIAS anira_models_guitar_lstm)
target_include_directories(anira_models_guitar_lstm
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_guitar_lstm_descriptions
    NAMESPACE anira::models::guitar_lstm
    HEADER anira/models/guitar_lstm/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/guitar-lstm.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/guitar-lstm.contract.json)
target_link_libraries(anira_models_guitar_lstm INTERFACE anira::models::core anira_models_guitar_lstm_descriptions)
anira_models_license(anira_models_guitar_lstm Apache-2.0)
