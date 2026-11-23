// SPDX-License-Identifier: Apache-2.0

// Speech-like audio without a recording, at 16 kHz: syllables of a 120 Hz pulse train through three
// formant resonators, 0.25 s on and 0.15 s off, four per phrase, two phrases with silence around
// them. data/make_vad_golden.py writes the same signal and Silero's probabilities on it.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace models_test {

inline std::vector<float> vad_signal() {
    constexpr std::size_t k_rate = 16000;
    constexpr std::size_t k_syllable = k_rate / 4;
    constexpr std::size_t k_period = k_rate / 120;
    constexpr std::array<std::pair<double, double>, 3> k_formants{
        {{700.0, 80.0}, {1200.0, 90.0}, {2600.0, 120.0}}};

    std::vector<double> out;
    const auto silence = [&](double seconds) {
        out.insert(out.end(), static_cast<std::size_t>(seconds * k_rate), 0.0);
    };
    const auto phrase = [&] {
        for (int s = 0; s < 4; ++s) {
            std::vector<double> x(k_syllable, 0.0);
            for (std::size_t i = 0; i < k_syllable; i += k_period) { x[i] = 1.0; }
            for (const auto& [formant, bandwidth] : k_formants) {
                const double r = std::exp(-std::numbers::pi * bandwidth / k_rate);
                const double a1 = -2.0 * r * std::cos(2.0 * std::numbers::pi * formant / k_rate);
                const double a2 = r * r;
                double y1 = 0.0;
                double y2 = 0.0;
                double peak = 0.0;
                for (double& sample : x) {
                    const double y0 = sample - (a1 * y1) - (a2 * y2);
                    y2 = y1;
                    y1 = y0;
                    sample = y0;
                    peak = std::max(peak, std::abs(y0));
                }
                for (double& sample : x) { sample /= peak; }
            }
            for (std::size_t i = 0; i < k_syllable; ++i) {
                const double hann =
                    0.5 - (0.5 * std::cos(2.0 * std::numbers::pi * static_cast<double>(i) /
                                          static_cast<double>(k_syllable - 1)));
                out.push_back(0.3 * x[i] * hann);
            }
            silence(0.15);
        }
    };
    silence(0.5);
    phrase();
    silence(1.0);
    phrase();
    silence(1.0);
    return {out.begin(), out.end()};
}

}  // namespace models_test
