# SPDX-License-Identifier: Apache-2.0
# Copies a target's runtime DLLs next to its binary, as anira's tests do; run as a POST_BUILD step:
#   cmake -DDLLS=<;-list from $<TARGET_RUNTIME_DLLS:...>> -DDEST=<dir> -P copy-runtime-dlls.cmake
# A no-op for an empty list (a script rather than `cmake -E copy`, which errors on zero files).
if(DLLS)
    file(COPY ${DLLS} DESTINATION "${DEST}")
endif()
