// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace anira::models::core {

/// The most latent dimensions a latent model of this library has.
inline constexpr std::size_t k_max_latents = 16;

/// One latent frame of an audio autoencoder (RAVE): its first m_count values, and the host sample
/// the audio it describes starts at (host samples since reset()).
struct Latents {
    std::int64_t m_position = 0;
    std::array<float, k_max_latents> m_values{};
    std::uint32_t m_count = 0;
};

}  // namespace anira::models::core
