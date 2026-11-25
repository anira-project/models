# SPDX-License-Identifier: Apache-2.0

# One licence per library target, and an Apache-2.0 target never links an AGPL one: an application
# that links only Apache-2.0 targets gets no AGPL code.
#
#   anira_models_license(<target> <Apache-2.0|AGPL-3.0-only>)   tags one of this repo's targets
#   anira_models_check_licenses()                                fails configure on a violation
#
# tanh-lib's AGPL components count as AGPL and its Core is Apache-2.0. Its DSP is licensed per file
# (BaseProcessor and utils/Resampler are Apache-2.0, the rest AGPL): this repo uses only the
# Apache-2.0 files, so tanh_dsp may be linked as a static library (the linker takes only the objects
# used), never as a shared one, which would ship its AGPL code.

include_guard(GLOBAL)

set(_anira_models_agpl_targets
    tanh_state tanh_resonator tanh_modulation tanh_audio_io tanh_net)
set(_anira_models_per_file_targets tanh_dsp)

function(anira_models_license target license)
    if(NOT license MATCHES "^(Apache-2.0|AGPL-3.0-only)$")
        message(FATAL_ERROR "anira_models_license(${target}): unknown licence '${license}'")
    endif()
    set_property(TARGET ${target} PROPERTY ANIRA_MODELS_LICENSE ${license})
    set_property(GLOBAL APPEND PROPERTY ANIRA_MODELS_LICENSED_TARGETS ${target})
endfunction()

function(_anira_models_is_agpl target out)
    get_target_property(_real ${target} ALIASED_TARGET)
    if(_real)
        set(target ${_real})
    endif()
    set(_agpl FALSE)
    if(target IN_LIST _anira_models_agpl_targets)
        set(_agpl TRUE)
    elseif(target IN_LIST _anira_models_per_file_targets)
        get_target_property(_type ${target} TYPE)
        if(NOT _type STREQUAL "STATIC_LIBRARY")
            set(_agpl TRUE)
        endif()
    else()
        get_target_property(_license ${target} ANIRA_MODELS_LICENSE)
        if(_license STREQUAL "AGPL-3.0-only")
            set(_agpl TRUE)
        endif()
    endif()
    set(${out} ${_agpl} PARENT_SCOPE)
endfunction()

# Every target <target> links, directly or through other targets.
function(_anira_models_links target out)
    set(_seen "")
    set(_queue ${target})
    while(_queue)
        list(POP_FRONT _queue _current)
        get_target_property(_real ${_current} ALIASED_TARGET)
        if(_real)
            set(_current ${_real})
        endif()
        foreach(_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_links ${_current} ${_property})
            foreach(_link IN LISTS _links)
                string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" _link "${_link}")
                if(TARGET ${_link} AND NOT _link IN_LIST _seen)
                    list(APPEND _seen ${_link})
                    list(APPEND _queue ${_link})
                endif()
            endforeach()
        endforeach()
    endwhile()
    set(${out} ${_seen} PARENT_SCOPE)
endfunction()

function(anira_models_check_licenses)
    get_property(_targets GLOBAL PROPERTY ANIRA_MODELS_LICENSED_TARGETS)
    foreach(_target IN LISTS _targets)
        get_target_property(_license ${_target} ANIRA_MODELS_LICENSE)
        if(NOT _license STREQUAL "Apache-2.0")
            continue()
        endif()
        _anira_models_links(${_target} _links)
        foreach(_link IN LISTS _links)
            _anira_models_is_agpl(${_link} _agpl)
            if(_agpl)
                message(FATAL_ERROR
                    "${_target} is Apache-2.0 but links ${_link}, which is AGPL-3.0: an Apache-2.0 "
                    "library must not link AGPL code.")
            endif()
        endforeach()
    endforeach()
endfunction()
