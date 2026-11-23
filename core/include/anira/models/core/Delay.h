// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tanh/utils/RealtimeSanitizer.h>

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace anira::models::core {

/// A fixed delay over any number of channels, for audio that should line up with a model's aligned
/// output: set it to the model's get_latency(). prepare() allocates; process() is real-time safe.
class Delay {
public:
    void prepare(std::size_t num_channels, std::size_t delay) {
        m_delay = delay;
        m_lines.assign(num_channels, std::vector<float>(delay, 0.0f));
        m_index = 0;
    }
    void reset() noexcept {
        for (auto& line : m_lines) { std::ranges::fill(line, 0.0f); }
        m_index = 0;
    }
    [[nodiscard]] std::size_t delay() const noexcept { return m_delay; }

    /// Delays `num_samples` samples of every channel in place (at most the prepared channels).
    void process(std::span<float* const> channels,
                 std::size_t num_samples) noexcept TANH_NONBLOCKING_FUNCTION {
        if (m_delay == 0) { return; }
        const std::size_t count = std::min(channels.size(), m_lines.size());
        std::size_t index = m_index;
        for (std::size_t c = 0; c < count; ++c) {
            float* line = m_lines[c].data();
            float* samples = channels[c];
            index = m_index;
            for (std::size_t i = 0; i < num_samples; ++i) {
                const float delayed = line[index];
                line[index] = samples[i];
                samples[i] = delayed;
                index = index + 1 == m_delay ? 0 : index + 1;
            }
        }
        m_index = count > 0 ? index : (m_index + num_samples) % m_delay;
    }

private:
    std::size_t m_delay = 0;
    std::vector<std::vector<float>> m_lines;
    std::size_t m_index = 0;
};

}  // namespace anira::models::core
