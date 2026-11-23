// SPDX-License-Identifier: Apache-2.0

#include "SpeakerCache.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace anira::models::stt {

namespace {

constexpr float k_infinity = std::numeric_limits<float>::infinity();
const float k_log_half = std::log(0.5f);

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

SpeakerCache::SpeakerCache(SpeakerCacheConfig config) : m_config(config) {
    // Every speaker's share of the cache, minus its silence slots, and the frame counts the
    // score policy spends it on.
    const std::size_t share = m_config.m_cache_frames / k_speakers;  // floor, as in the reference
    const auto budget =
        static_cast<double>(share) - static_cast<double>(m_config.m_silence_frames_per_speaker);
    m_min_positive_scores =
        static_cast<std::size_t>(std::floor(budget * m_config.m_min_positive_scores_rate));
    m_strong_boosted = static_cast<std::size_t>(std::floor(budget * m_config.m_strong_boost_rate));
    m_weak_boosted = static_cast<std::size_t>(std::floor(budget * m_config.m_weak_boost_rate));
}

std::vector<float> SpeakerCache::embeds() const {
    std::vector<float> out(m_cache.begin(), m_cache.end());
    out.insert(out.end(), m_fifo.begin(), m_fifo.end());
    return out;
}

void SpeakerCache::reset() {
    m_cache.clear();
    m_cache_probs.clear();
    m_fifo.clear();
    m_cache_frames = 0;
    m_fifo_frames = 0;
    m_compressed = false;
}

void SpeakerCache::update(std::span<const float> input_embeds,
                          std::span<const float> logits,
                          std::span<const float> silence_embeds,
                          std::size_t chunk_frames) {
    // Speaker probabilities per encoder frame: the logit frames' sigmoid, averaged.
    const auto input_frames = logits.size() / (k_subsampling * k_speakers);
    Rows probs(input_frames * k_speakers, 0.0f);
    for (std::size_t frame = 0; frame < input_frames; ++frame) {
        for (std::size_t s = 0; s < k_speakers; ++s) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < k_subsampling; ++k) {
                sum += sigmoid(logits[(((frame * k_subsampling) + k) * k_speakers) + s]);
            }
            probs[(frame * k_speakers) + s] = sum / static_cast<float>(k_subsampling);
        }
    }

    const auto chunk_start = m_cache_frames + m_fifo_frames;
    Rows fifo = m_fifo;
    fifo.insert(fifo.end(),
                input_embeds.begin() + static_cast<std::ptrdiff_t>(chunk_start * k_hidden),
                input_embeds.begin() +
                    static_cast<std::ptrdiff_t>((chunk_start + chunk_frames) * k_hidden));
    auto fifo_frames = fifo.size() / k_hidden;

    std::size_t popped = 0;
    if (fifo_frames > m_config.m_fifo_frames) {
        popped = std::min(std::max(m_config.m_update_period, fifo_frames - m_config.m_fifo_frames),
                          fifo_frames);
    }
    if (popped > 0) {
        // An uncompressed cache still holds plain chunk frames, whose probabilities this step
        // re-estimates; a compressed one is out of order, so its stored ones are kept.
        Rows cache = m_cache;
        Rows cache_probs =
            m_compressed
                ? m_cache_probs
                : Rows(probs.begin(),
                       probs.begin() + static_cast<std::ptrdiff_t>(m_cache_frames * k_speakers));
        cache.insert(cache.end(),
                     fifo.begin(),
                     fifo.begin() + static_cast<std::ptrdiff_t>(popped * k_hidden));
        const auto fifo_probs = m_cache_frames * k_speakers;
        cache_probs.insert(
            cache_probs.end(),
            probs.begin() + static_cast<std::ptrdiff_t>(fifo_probs),
            probs.begin() + static_cast<std::ptrdiff_t>(fifo_probs + (popped * k_speakers)));
        fifo.erase(fifo.begin(), fifo.begin() + static_cast<std::ptrdiff_t>(popped * k_hidden));
        fifo_frames -= popped;
        if (cache.size() / k_hidden > m_config.m_cache_frames) {
            compress(cache, cache_probs, silence_embeds);
            m_compressed = true;
        }
        m_cache = std::move(cache);
        m_cache_probs = std::move(cache_probs);
        m_cache_frames = m_cache.size() / k_hidden;
    }
    m_fifo = std::move(fifo);
    m_fifo_frames = fifo_frames;
}

SpeakerCache::Rows SpeakerCache::frame_scores(const Rows& probs, std::size_t frames) const {
    const float threshold = m_config.m_prediction_score_threshold;
    Rows scores(frames * k_speakers);
    std::vector<std::size_t> positive(k_speakers, 0);
    for (std::size_t f = 0; f < frames; ++f) {
        float complements = 0.0f;
        for (std::size_t s = 0; s < k_speakers; ++s) {
            complements += std::log(std::max(1.0f - probs[(f * k_speakers) + s], threshold));
        }
        for (std::size_t s = 0; s < k_speakers; ++s) {
            const float p = probs[(f * k_speakers) + s];
            float score = std::log(std::max(p, threshold)) -
                          std::log(std::max(1.0f - p, threshold)) + complements - k_log_half;
            if (p <= 0.5f) { score = -k_infinity; }
            if (score > 0.0f) { ++positive[s]; }
            scores[(f * k_speakers) + s] = score;
        }
    }
    // A speaker with enough positive frames keeps only those.
    for (std::size_t f = 0; f < frames; ++f) {
        for (std::size_t s = 0; s < k_speakers; ++s) {
            auto& score = scores[(f * k_speakers) + s];
            const bool speech = probs[(f * k_speakers) + s] > 0.5f;
            if (speech && score <= 0.0f && positive[s] >= m_min_positive_scores) {
                score = -k_infinity;
            }
        }
    }
    return scores;
}

void SpeakerCache::compress(Rows& embeds,
                            Rows& probs,
                            std::span<const float> silence_embeds) const {
    // Keeps the m_cache_frames most important frames, grouped by speaker, in their original
    // order within a speaker; m_silence_frames_per_speaker slots per speaker hold silence.
    const auto frames = embeds.size() / k_hidden;
    auto scores = frame_scores(probs, frames);
    for (std::size_t f = m_config.m_cache_frames; f < frames; ++f) {  // popped from the FIFO
        for (std::size_t s = 0; s < k_speakers; ++s) {
            scores[(f * k_speakers) + s] += m_config.m_latest_frames_score_boost;
        }
    }
    const auto boost = [&](std::size_t count, float amount) {
        std::vector<std::size_t> order(frames);
        for (std::size_t s = 0; s < k_speakers; ++s) {
            std::iota(order.begin(), order.end(), std::size_t{0});
            const auto k = std::min(count, frames);
            std::partial_sort(order.begin(),
                              order.begin() + static_cast<std::ptrdiff_t>(k),
                              order.end(),
                              [&](std::size_t a, std::size_t b) {
                                  return scores[(a * k_speakers) + s] >
                                         scores[(b * k_speakers) + s];
                              });
            for (std::size_t i = 0; i < k; ++i) { scores[(order[i] * k_speakers) + s] += amount; }
        }
    };
    boost(m_strong_boosted, -2.0f * k_log_half);
    boost(m_weak_boosted, -k_log_half);

    // Speaker-major flat scores over the frames plus the silence slots (always kept).
    const auto silence = m_config.m_silence_frames_per_speaker;
    const auto scored = frames + silence;
    std::vector<float> flat(scored * k_speakers, k_infinity);
    for (std::size_t s = 0; s < k_speakers; ++s) {
        for (std::size_t f = 0; f < frames; ++f) {
            flat[(s * scored) + f] = scores[(f * k_speakers) + s];
        }
    }
    std::vector<std::size_t> top(flat.size());
    std::iota(top.begin(), top.end(), std::size_t{0});
    const auto keep = std::min(m_config.m_cache_frames, top.size());
    std::partial_sort(top.begin(),
                      top.begin() + static_cast<std::ptrdiff_t>(keep),
                      top.end(),
                      [&](std::size_t a, std::size_t b) {
                          return flat[a] > flat[b] || (flat[a] == flat[b] && a < b);
                      });
    top.resize(keep);
    // A slot without a usable frame (-inf) and every silence slot hold the silence embedding.
    const auto sentinel = flat.size();
    for (auto& index : top) {
        if (flat[index] == -k_infinity) { index = sentinel; }
    }
    std::ranges::sort(top);

    Rows kept_embeds;
    Rows kept_probs;
    kept_embeds.reserve(keep * k_hidden);
    kept_probs.reserve(keep * k_speakers);
    for (const auto index : top) {
        const auto frame = index == sentinel ? frames : std::min(index % scored, frames);
        if (frame == frames) {
            kept_embeds.insert(kept_embeds.end(), silence_embeds.begin(), silence_embeds.end());
            kept_probs.insert(kept_probs.end(), k_speakers, 0.0f);
        } else {
            kept_embeds.insert(
                kept_embeds.end(),
                embeds.begin() + static_cast<std::ptrdiff_t>(frame * k_hidden),
                embeds.begin() + static_cast<std::ptrdiff_t>((frame + 1) * k_hidden));
            kept_probs.insert(
                kept_probs.end(),
                probs.begin() + static_cast<std::ptrdiff_t>(frame * k_speakers),
                probs.begin() + static_cast<std::ptrdiff_t>((frame + 1) * k_speakers));
        }
    }
    embeds = std::move(kept_embeds);
    probs = std::move(kept_probs);
}

}  // namespace anira::models::stt
