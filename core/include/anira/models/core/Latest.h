// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

namespace anira::models::core {

/// The newest value from one writer thread for one reader thread, of any size: a triple buffer,
/// wait-free on both sides, no allocation. The reader gets the newest value written (or the last
/// it got, when nothing new came); values in between are skipped.
template <class T>
class Latest {
public:
    /// Writer: publishes `value`.
    void write(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        m_slots[m_write] = value;
        const std::uint8_t previous =
            m_shared.exchange(static_cast<std::uint8_t>(m_write | k_fresh),
                              std::memory_order_acq_rel);
        m_write = previous & k_index;
    }
    /// Reader: the newest value.
    [[nodiscard]] const T& read() noexcept {
        if ((m_shared.load(std::memory_order_relaxed) & k_fresh) != 0) {
            const std::uint8_t previous = m_shared.exchange(m_read, std::memory_order_acq_rel);
            m_read = previous & k_index;
        }
        return m_slots[m_read];
    }
    /// With neither thread using it: every slot back to `value`.
    void reset(const T& value = T{}) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        m_slots.fill(value);
        m_write = 0;
        m_shared.store(1, std::memory_order_relaxed);
        m_read = 2;
    }

private:
    static constexpr std::uint8_t k_index = 0x3;
    static constexpr std::uint8_t k_fresh = 0x4;

    std::array<T, 3> m_slots{};
    std::uint8_t m_write = 0;
    std::atomic<std::uint8_t> m_shared{1};
    std::uint8_t m_read = 2;
};

}  // namespace anira::models::core
