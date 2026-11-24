// SPDX-License-Identifier: Apache-2.0
// From anira (extras/models/hybrid-nn/HybridNNProcessor.h, Apache-2.0), unchanged but for its name.

#pragma once

#include <anira/anira.h>

#include <anira/anira.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace anira::models::guitar_lstm {

/// The windows GuitarLSTM takes, as the pre_process of an anira::Processor: the model's input is
/// `batches` windows of 150 samples, each one sample further than the last ([256, 1, 150]), its
/// output one sample per window, so one inference consumes 256 new samples with 149 samples of
/// history ahead of them. anira's default fill writes one window per channel; these batches are
/// one RingView::pop_windows of 256 windows of 149 old and 1 new sample. The output side is
/// anira's default push.
class GuitarLstmProcessor final : public anira::Processor {
public:
    uint32_t phases() const noexcept override { return k_pre_process; }
    // pre_process runs on the audio thread under a Hard contract: it pops and copies, nothing
    // else.
    uint32_t realtime_phases() const noexcept override { return phases(); }

    std::unique_ptr<Prepared> prepare(const anira::PrepareInfo& info) override {
        const anira::Tensor& input = info.inputs()[0];
        const anira::Tensor& output = info.outputs()[0];
        const auto batches = static_cast<uint32_t>(input.extent(0));
        const std::size_t window = batches > 0 ? input.num_elements() / batches : 0;
        const std::size_t num_new = batches > 0 ? output.num_elements() / batches : 0;
        if (batches == 0 || num_new == 0 || num_new > window) {
            throw anira::Error(ANIRA_ERROR_CONFIG,
                               "GuitarLstmProcessor: the model is not [batches, 1, window] -> "
                               "[batches, new samples]");
        }
        return std::make_unique<Windows>(batches, window - num_new, num_new);
    }

private:
    class Windows final : public Prepared {
    public:
        Windows(uint32_t batches, std::size_t num_old, std::size_t num_new) noexcept
            : m_batches(batches), m_num_old(num_old), m_num_new(num_new) {}

        anira_status pre_process(anira::ProcessorContext& ctx) noexcept override {
            anira::RingView ring;
            const anira_status has_ring = ctx.input_ring(0, ring);
            if (has_ring != ANIRA_OK) { return has_ring; }
            anira::Tensor tensor;
            const anira_status exposed = ctx.input_tensor(0, tensor);
            if (exposed != ANIRA_OK) { return exposed; }
            float* samples = tensor.data_f32();
            if (samples == nullptr) { return ANIRA_ERROR_CONFIG; }
            const std::span<float> windows(samples, tensor.num_elements());
            const std::size_t written =
                ring.pop_windows(0, windows, m_num_new, m_num_old, 0, m_batches);
            return written == 0 ? ANIRA_ERROR_CONFIG : ANIRA_OK;
        }

    private:
        uint32_t m_batches;
        std::size_t m_num_old;
        std::size_t m_num_new;
    };
};

}  // namespace anira::models::guitar_lstm
