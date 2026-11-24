// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tanh/dsp-basics/Resampler.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

private:
    std::vector<float> m_ring;
    std::size_t m_prime = 0;
    std::size_t m_read = 0;
    std::size_t m_fill = 0;
};

/// The model samples due after `host` host samples: floor(host * model rate / host rate).
inline std::int64_t model_samples_by(std::int64_t host, double ratio) noexcept {
    return static_cast<std::int64_t>(std::floor(static_cast<double>(host) * ratio));
}

/// A resampled path's timing, found on resamplers of its own one host sample at a time (a
/// resampler's output count depends only on how much it was given, never on the blocks): the most
/// samples the path ever owes the host (the delay's prime), and where an impulse comes out behind
/// those zeros, anira's latency included.
struct PathTiming {
    std::size_t m_prime = 0;
    std::size_t m_latency = 0;
};

/// host -> model -> host (an effect): anira gives as many samples as it gets, `model_latency` late.
inline PathTiming effect_timing(double host_rate, double model_rate, std::uint32_t model_latency) {
    thl::dsp_basics::Resampler down;
    thl::dsp_basics::Resampler up;
    down.prepare(host_rate, model_rate, 1);
    const std::size_t model_max = std::max<std::size_t>(down.max_output(1), model_latency);
    up.prepare(model_rate, host_rate, model_max);
    std::vector<float> model(model_max, 0.0F);
    std::vector<float> host(up.max_output(model_max));
    const auto length = static_cast<std::size_t>(
        (static_cast<double>(model_latency) * host_rate / model_rate) + (host_rate * 0.25));

    std::vector<float> out;
    out.reserve(length + host.size());
    const std::size_t primed =
        up.process(std::span<const float>(model.data(), model_latency), host);
    out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(primed));
    for (std::size_t i = 0; i < length; ++i) {
        const float sample = i == 0 ? 1.0F : 0.0F;  // the impulse at host sample 0
        const std::size_t to_model = down.process(std::span<const float>(&sample, 1), model);
        const std::size_t to_host =
            up.process(std::span<const float>(model.data(), to_model), host);
        out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(to_host));
    }
    // What the path owes, on fresh filters (anira's priming zeros are inside its own count).
    down.reset();
    up.reset();
    PathTiming timing;
    std::int64_t owed = 0;
    for (std::size_t i = 0; i < length; ++i) {
        const float zero = 0.0F;
        const std::size_t to_model = down.process(std::span<const float>(&zero, 1), model);
        owed += 1 - static_cast<std::int64_t>(
                        up.process(std::span<const float>(model.data(), to_model), host));
        timing.m_prime =
            std::max(timing.m_prime, static_cast<std::size_t>(std::max<std::int64_t>(owed, 0)));
    }
    const auto peak = std::ranges::max_element(out, {}, [](float x) { return std::abs(x); });
    timing.m_latency = timing.m_prime + static_cast<std::size_t>(peak - out.begin());
    return timing;
}

/// model -> host (a generator): model_samples_by() model samples per host sample, anira's first
/// `model_latency` of them its priming.
inline PathTiming generator_timing(double host_rate,
                                   double model_rate,
                                   std::uint32_t model_latency) {
    const double ratio = model_rate / host_rate;
    thl::dsp_basics::Resampler up;
    const std::size_t model_max = static_cast<std::size_t>(std::ceil(ratio)) + 1;
    up.prepare(model_rate, host_rate, model_max);
    std::vector<float> model(model_max, 0.0F);
    std::vector<float> host(up.max_output(model_max));
    const auto length = static_cast<std::int64_t>((static_cast<double>(model_latency) / ratio) +
                                                  (host_rate * 0.25));

    PathTiming timing;
    std::int64_t owed = 0;
    for (std::int64_t n = 1; n <= length; ++n) {
        const auto m =
            static_cast<std::size_t>(model_samples_by(n, ratio) - model_samples_by(n - 1, ratio));
        owed += 1 - static_cast<std::int64_t>(
                        up.process(std::span<const float>(model.data(), m), host));
        timing.m_prime =
            std::max(timing.m_prime, static_cast<std::size_t>(std::max<std::int64_t>(owed, 0)));
    }
    up.reset();
    std::vector<float> out;
    std::int64_t given = 0;
    for (std::int64_t n = 1; n <= length; ++n) {
        const std::int64_t due = model_samples_by(n, ratio);
        const auto m = static_cast<std::size_t>(due - given);
        for (std::size_t i = 0; i < m; ++i) {
            model[i] = given + static_cast<std::int64_t>(i) == model_latency ? 1.0F : 0.0F;
        }
        given = due;
        const std::size_t made = up.process(std::span<const float>(model.data(), m), host);
        out.insert(out.end(), host.begin(), host.begin() + static_cast<std::ptrdiff_t>(made));
    }
    const auto peak = std::ranges::max_element(out, {}, [](float x) { return std::abs(x); });
    timing.m_latency = timing.m_prime + static_cast<std::size_t>(peak - out.begin());
    return timing;
}

}  // namespace anira::models::core::detail
