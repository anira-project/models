// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace anira::models::core::detail {

/// The model's audio at the host's rate, not given to the host yet: a ring that starts with
/// `prime` zeros, so that the host never finds it dry (the most the model path ever owes it).
class HostDelay {
public:
    /// Non-real-time.
    void assign(std::size_t prime, std::size_t capacity) {
        m_prime = prime;
        m_ring.assign(prime + capacity + 1, 0.0F);
        reset();
    }
    void reset() noexcept {
        std::ranges::fill(m_ring, 0.0F);
        m_read = 0;
        m_fill = m_prime;
    }
    void write(std::span<const float> samples) noexcept {
        const std::size_t size = m_ring.size();
        for (std::size_t i = 0; i < samples.size() && m_fill < size; ++i, ++m_fill) {
            m_ring[(m_read + m_fill) % size] = samples[i];
        }
    }
    void read(std::span<float> out) noexcept {
        for (float& x : out) {
            if (m_fill == 0) {
                x = 0.0F;  // never: the prime covers the most the path owes
                continue;
            }
            x = m_ring[m_read];
            m_read = (m_read + 1) % m_ring.size();
            --m_fill;
        }
    }
    [[nodiscard]] bool empty() const noexcept { return m_ring.empty(); }

private:
    std::vector<float> m_ring;
    std::size_t m_prime = 0;
    std::size_t m_read = 0;
    std::size_t m_fill = 0;
};

}  // namespace anira::models::core::detail
