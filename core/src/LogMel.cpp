// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/LogMel.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace anira::models::core {

namespace {

constexpr std::size_t k_bins = (LogMel::k_fft / 2) + 1;
constexpr double k_sample_rate = 16000.0;
constexpr double k_f_sp = 200.0 / 3.0;  // Slaney: linear below 1 kHz, logarithmic above
constexpr double k_min_log_hz = 1000.0;

double hz_to_mel(double hz) {
    const double min_log_mel = k_min_log_hz / k_f_sp;
    const double log_step = std::log(6.4) / 27.0;
    return hz < k_min_log_hz ? hz / k_f_sp : min_log_mel + (std::log(hz / k_min_log_hz) / log_step);
}

double mel_to_hz(double mel) {
    const double min_log_mel = k_min_log_hz / k_f_sp;
    const double log_step = std::log(6.4) / 27.0;
    return mel < min_log_mel ? mel * k_f_sp
                             : k_min_log_hz * std::exp(log_step * (mel - min_log_mel));
}

// In place, radix 2.
void fft(std::vector<float>& re, std::vector<float>& im) noexcept {
    const std::size_t n = re.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1U;
        for (; (j & bit) != 0U; bit >>= 1U) { j ^= bit; }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (std::size_t length = 2; length <= n; length <<= 1U) {
        const double angle = -2.0 * std::numbers::pi / static_cast<double>(length);
        const auto w_re = static_cast<float>(std::cos(angle));
        const auto w_im = static_cast<float>(std::sin(angle));
        for (std::size_t i = 0; i < n; i += length) {
            float c_re = 1.0F;
            float c_im = 0.0F;
            for (std::size_t k = 0; k < length / 2; ++k) {
                const std::size_t a = i + k;
                const std::size_t b = a + (length / 2);
                const float v_re = (re[b] * c_re) - (im[b] * c_im);
                const float v_im = (re[b] * c_im) + (im[b] * c_re);
                re[b] = re[a] - v_re;
                im[b] = im[a] - v_im;
                re[a] += v_re;
                im[a] += v_im;
                const float next = (c_re * w_re) - (c_im * w_im);
                c_im = (c_re * w_im) + (c_im * w_re);
                c_re = next;
            }
        }
    }
}

}  // namespace

LogMel::LogMel()
    : m_window(k_fft, 0.0F)
    , m_melbank(k_mels * k_bins, 0.0F)
    , m_re(k_fft)
    , m_im(k_fft)
    , m_power(k_bins) {
    // Periodic Hann of 400 samples centred in the 512-sample frame (torch.stft pads the window).
    const std::size_t left = (k_fft - k_window) / 2;
    for (std::size_t i = 0; i < k_window; ++i) {
        m_window[left + i] = static_cast<float>(
            0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * static_cast<double>(i) / k_window)));
    }
    // Slaney-normalised triangular filters (librosa.filters.mel).
    const double mel_min = hz_to_mel(0.0);
    const double mel_max = hz_to_mel(k_sample_rate / 2.0);
    std::vector<double> edges(k_mels + 2);
    for (std::size_t m = 0; m < edges.size(); ++m) {
        edges[m] = mel_to_hz(mel_min + ((mel_max - mel_min) * static_cast<double>(m) /
                                        static_cast<double>(k_mels + 1)));
    }
    for (std::size_t m = 0; m < k_mels; ++m) {
        const double low = edges[m];
        const double mid = edges[m + 1];
        const double high = edges[m + 2];
        const double norm = 2.0 / (high - low);
        for (std::size_t bin = 0; bin < k_bins; ++bin) {
            const double hz = static_cast<double>(bin) * k_sample_rate / static_cast<double>(k_fft);
            const double weight =
                std::max(0.0, std::min((hz - low) / (mid - low), (high - hz) / (high - mid)));
            m_melbank[(m * k_bins) + bin] = static_cast<float>(weight * norm);
        }
    }
    prepare(4096);
}

void LogMel::prepare(std::size_t max_block) {
    m_pending.assign(k_fft + max_block, 0.0F);
    reset();
}

void LogMel::reset() noexcept {
    m_pending_size = 0;
    m_previous = 0.0F;
    m_started = false;
}

std::size_t LogMel::process(std::span<const float> samples, std::span<float> frames) noexcept {
    if (!m_started) {  // half a frame of zeros on the left: frame k centred on sample k * hop
        std::fill_n(m_pending.begin(), k_fft / 2, 0.0F);
        m_pending_size = k_fft / 2;
        m_started = true;
    }
    const std::size_t take = std::min(samples.size(), m_pending.size() - m_pending_size);
    for (std::size_t i = 0; i < take; ++i) {
        m_pending[m_pending_size++] = samples[i] - (k_preemphasis * m_previous);
        m_previous = samples[i];
    }
    std::size_t count = 0;
    std::size_t offset = 0;
    while (m_pending_size - offset >= k_fft && (count + 1) * k_mels <= frames.size()) {
        frame(m_pending.data() + offset, frames.data() + (count * k_mels));
        offset += k_hop;
        ++count;
    }
    if (offset > 0) {
        std::copy(m_pending.begin() + static_cast<std::ptrdiff_t>(offset),
                  m_pending.begin() + static_cast<std::ptrdiff_t>(m_pending_size),
                  m_pending.begin());
        m_pending_size -= offset;
    }
    return count;
}

void LogMel::frame(const float* samples, float* out) noexcept {
    for (std::size_t i = 0; i < k_fft; ++i) {
        m_re[i] = samples[i] * m_window[i];
        m_im[i] = 0.0F;
    }
    fft(m_re, m_im);
    for (std::size_t bin = 0; bin < k_bins; ++bin) {
        m_power[bin] = (m_re[bin] * m_re[bin]) + (m_im[bin] * m_im[bin]);
    }
    static const float k_guard = std::ldexp(1.0F, -24);
    for (std::size_t m = 0; m < k_mels; ++m) {
        const float* bank = m_melbank.data() + (m * k_bins);
        float sum = 0.0F;
        for (std::size_t bin = 0; bin < k_bins; ++bin) { sum += bank[bin] * m_power[bin]; }
        out[m] = std::log(sum + k_guard);
    }
}

}  // namespace anira::models::core
