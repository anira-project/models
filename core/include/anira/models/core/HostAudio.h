// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/anira.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <span>

namespace anira::models::core {

// Host audio: [channels, n] or [n] float host tensors (planar or packed), as anira takes them.

/// Channel c of a [channels, n] or [n] float host tensor, nullptr when the tensor is none of those.
inline float* channel(const anira::Tensor& audio, std::uint32_t c) noexcept {
    if (audio.dtype != ANIRA_DTYPE_F32) { return nullptr; }
    if ((audio.flags & ANIRA_TENSOR_PLANAR) != 0U) {
        return static_cast<float*>(anira_tensor_plane(&audio, c, ANIRA_DTYPE_F32));
    }
    float* data = anira_tensor_data_f32(&audio);
    if (data == nullptr) { return nullptr; }
    return audio.ndim == 1 ? data : data + (static_cast<std::size_t>(c) * audio.shape[1]);
}

inline std::uint32_t channels(const anira::Tensor& audio) noexcept {
    if (audio.ndim == 1) { return 1; }
    return audio.ndim == 2 ? static_cast<std::uint32_t>(audio.shape[0]) : 0;
}

inline std::size_t samples(const anira::Tensor& audio) noexcept {
    return audio.ndim == 1 || audio.ndim == 2
               ? static_cast<std::size_t>(audio.shape[audio.ndim - 1])
               : 0;
}

/// The mean of `audio`'s channels into `mono` (at least samples(audio) long); false for a tensor
/// that is not float host audio.
inline bool downmix(const anira::Tensor& audio, std::span<float> mono) noexcept {
    const std::uint32_t count = channels(audio);
    const std::size_t n = samples(audio);
    if (count == 0 || mono.size() < n) { return false; }
    std::fill_n(mono.begin(), n, 0.0F);
    for (std::uint32_t c = 0; c < count; ++c) {
        const float* source = channel(audio, c);
        if (source == nullptr) { return false; }
        for (std::size_t i = 0; i < n; ++i) { mono[i] += source[i]; }
    }
    if (count > 1) {
        const float scale = 1.0F / static_cast<float>(count);
        for (std::size_t i = 0; i < n; ++i) { mono[i] *= scale; }
    }
    return true;
}

/// `mono` to every channel of `audio`; false for a tensor that is not float host audio.
inline bool spread(std::span<const float> mono, const anira::Tensor& audio) noexcept {
    const std::uint32_t count = channels(audio);
    const std::size_t n = samples(audio);
    if (count == 0 || mono.size() < n) { return false; }
    for (std::uint32_t c = 0; c < count; ++c) {
        float* target = channel(audio, c);
        if (target == nullptr) { return false; }
        std::copy_n(mono.begin(), n, target);
    }
    return true;
}

}  // namespace anira::models::core
