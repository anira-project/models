// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace anira::models::core {

/// A bounded queue between one producer thread (the audio thread) and one consumer thread (the
/// reader): wait-free on both sides, no allocation after reserve(). A full queue refuses the entry
/// (push returns false); the producer counts what it could not deliver.
template <class T>
class SpscQueue {
public:
    /// Non-real-time, with neither thread using the queue: room for `capacity` entries.
    void reserve(std::size_t capacity) {
        m_slots.assign(capacity + 1, T{});
        clear();
    }
    /// With neither thread using the queue.
    void clear() noexcept {
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
    }

    /// Producer.
    bool push(const T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        const std::size_t tail = m_tail.load(std::memory_order_relaxed);
        const std::size_t next = advance(tail);
        if (m_slots.empty() || next == m_head.load(std::memory_order_acquire)) { return false; }
        m_slots[tail] = value;
        m_tail.store(next, std::memory_order_release);
        return true;
    }
    /// Consumer: the oldest entry, false when there is none.
    bool pop(T& value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
        const std::size_t head = m_head.load(std::memory_order_relaxed);
        if (head == m_tail.load(std::memory_order_acquire)) { return false; }
        value = m_slots[head];
        m_head.store(advance(head), std::memory_order_release);
        return true;
    }

private:
    std::size_t advance(std::size_t index) const noexcept {
        return index + 1 == m_slots.size() ? 0 : index + 1;
    }

    std::vector<T> m_slots;
    std::atomic<std::size_t> m_head{0};
    std::atomic<std::size_t> m_tail{0};
};

}  // namespace anira::models::core
