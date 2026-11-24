# SPDX-License-Identifier: Apache-2.0

# anira::models::stateful_lstm: the stateful LSTM (vackva/stateful-lstm), an LSTM effect with its state carried by anira. Header-only: StatefulLstm (<anira/models/stateful_lstm/StatefulLstm.h>) on
# core::AudioEffect, with the description compiled into <anira/models/stateful_lstm/Descriptions.h>; the
# weights are on Hugging Face (stateful-lstm.catalog.json).

add_library(anira_models_stateful_lstm INTERFACE)
add_library(anira::models::stateful_lstm ALIAS anira_models_stateful_lstm)
target_include_directories(anira_models_stateful_lstm
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
anira_models_embed_text(anira_models_stateful_lstm_descriptions
    NAMESPACE anira::models::stateful_lstm
    HEADER anira/models/stateful_lstm/Descriptions.h
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/stateful-lstm.model.json
        ${CMAKE_CURRENT_SOURCE_DIR}/stateful-lstm.contract.json)
target_link_libraries(anira_models_stateful_lstm INTERFACE anira::models::core anira_models_stateful_lstm_descriptions)
anira_models_license(anira_models_stateful_lstm Apache-2.0)
