// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anira/models/core/ModelSource.h>

#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace anira::models::core {

/// A job's state; anira_ticket_status once anira's Async contract lands (M3).
enum class TicketStatus : std::uint8_t {
    Pending,  ///< queued or running
    Met,      ///< done within its deadline (or without one)
    Late,     ///< done past its deadline (on_late finish)
    Dropped,  ///< not done by its deadline (on_late drop): the bound outputs hold zeros
    Failed,   ///< refused at submit (a full queue: error() is ANIRA_ERROR_CAPACITY) or by anira
};

class AsyncHandler;

/// One submitted job. Move-only; releases its slot when destroyed (the job still runs to its end;
/// its slot is reused once it has). A refused submit gives a ticket that is failed from the start.
class Ticket {
public:
    Ticket() = default;
    ~Ticket();
    Ticket(Ticket&& other) noexcept;
    Ticket& operator=(Ticket&& other) noexcept;
    Ticket(const Ticket&) = delete;
    Ticket& operator=(const Ticket&) = delete;

    [[nodiscard]] TicketStatus status() const noexcept;
    /// Blocks until the job is done.
    TicketStatus wait();
    /// Blocks until the job is done or the time is up; pending when it is.
    TicketStatus wait_for(std::chrono::nanoseconds timeout);
    /// What submit was given.
    [[nodiscard]] void* user_data() const noexcept;
    /// Why a failed job failed: ANIRA_ERROR_CAPACITY for a full queue, else anira's status.
    [[nodiscard]] anira_status error() const noexcept;

private:
    friend class AsyncHandler;
    Ticket(AsyncHandler* owner, std::uint32_t slot) noexcept : m_owner(owner), m_slot(slot) {}
    explicit Ticket(anira_status refused) noexcept : m_refused(refused) {}
    void release() noexcept;

    AsyncHandler* m_owner = nullptr;
    std::uint32_t m_slot = 0;
    anira_status m_refused = ANIRA_OK;
};

/// Jobs on a model written for anira's Async contract (Buffer tensors: a whole frame, a step's
/// audio, a cache; State and Static tensors as anira has them), with the surface anira fixes for
/// it (anira-v3-architecture.md, section 6, "The Async half"), so that anira's Async contract
/// (M3) replaces the inside and not the callers.
///
/// Until then the inside is anira's Hard contract: the constructor rewrites a copy of the
/// description, every Buffer tensor a Streamed one flattened onto its Time axis (one window = the
/// whole tensor; the workshop's way); prepare derives the Hard contract (one window per push, the
/// deadline as the budget, zeros on a miss, every output's latency 0, no warm-up); a worker thread
/// per lane, each with a handler of its own, pushes a job and waits for its outputs: without limit
/// under on_late finish (never misses), until the deadline under on_late drop (a job not done by
/// then is dropped and its outputs are zeros; anira delivers no late block afterwards, so the
/// stream stays in step). A model with State runs on one lane (each lane would keep its own state).
///
/// Threads: submit from one thread, never blocks and never allocates (a semaphore wakes the
/// worker); poll and the tickets from any one thread.
class AsyncHandler {
public:
    /// The model's description (the text of a `*.model.json` written for Async) and its weights.
    /// Its CPU ONNX Runtime entries run on OnnxEngine with `threads` intra-op threads (0: one per
    /// physical core); a job is not real-time, so it may take more than anira's one thread.
    /// Throws Error for a description anira refuses or a Buffer tensor without an axis of 1.
    AsyncHandler(anira::Context& context,
                 std::string_view model_json,
                 const ModelSource& weights,
                 std::uint32_t threads = 0);
    ~AsyncHandler();
    AsyncHandler(const AsyncHandler&) = delete;
    AsyncHandler& operator=(const AsyncHandler&) = delete;
    AsyncHandler(AsyncHandler&&) = delete;
    AsyncHandler& operator=(AsyncHandler&&) = delete;

    /// deadline, on_late, lanes (0: one), max_in_flight (0: four). Main thread; outstanding
    /// tickets must be released first. Throws anira's Error.
    anira::PlanReport prepare(const anira::Async& contract);
    /// Where every job's output `slot` goes (a host tensor of the output's dtype and shape; a
    /// State output has none). Before submit; the memory must outlive the jobs.
    void bind_output(std::uint32_t slot, const anira::Tensor& out);
    /// Called by poll for every job done since the last poll, in completion order.
    void on_complete(std::function<void(void* user_data, TicketStatus status)> callback);

    /// One tensor per input slot of the model, in its order: a Buffer input in the model's shape
    /// (contiguous host memory), a Static input its value, a State input an empty tensor. Borrowed
    /// until the job is done. JobOptions are taken for anira's surface and not used until M3.
    Ticket submit(std::span<const anira::Tensor> inputs,
                  const anira::JobOptions& options = {},
                  std::optional<std::chrono::steady_clock::time_point> deadline = {},
                  void* user_data = nullptr) noexcept;
    /// Runs on_complete for the jobs done since the last poll; their number.
    std::uint32_t poll();

    /// A new stream: every lane's State back to zero (anira_handler_reset, which each lane's
    /// worker calls before its next job: anira's driver thread). No job in flight.
    void reset() noexcept;
    anira_status set_plan(std::uint32_t plan) noexcept;
    [[nodiscard]] anira::PlanReport plan_report() const;
    /// The description as the Hard handlers run it (for tests and logs).
    [[nodiscard]] const std::string& hard_description() const noexcept { return m_hard_json; }

private:
    friend class Ticket;
    static constexpr std::size_t k_max_tensors = 16;

    enum class Role : std::uint8_t { Streamed, Static, State };
    struct Spec {
        Role m_role = Role::Static;
        anira_dtype m_dtype = ANIRA_DTYPE_F32;
        std::int64_t m_window = 0;  // elements, a Streamed (rewritten Buffer) tensor
    };
    struct Job {
        std::array<anira::Tensor, k_max_tensors> m_inputs{};
        std::optional<std::chrono::steady_clock::time_point> m_deadline;
        void* m_user_data = nullptr;
        std::atomic<TicketStatus> m_status{TicketStatus::Pending};
        anira_status m_error = ANIRA_OK;
        std::atomic<int> m_holders{0};  // ticket, poll
        // Its worker has not finished it. Cleared after the status is published, so a ticket that
        // saw the status can wait the last moment out and its next submit finds the slot free.
        std::atomic<bool> m_running{false};
    };
    struct Lane {
        std::unique_ptr<anira::InferenceHandler> m_handler;
        std::thread m_thread;
        std::uint64_t m_resets = 0;  // the worker's: the resets it has made
    };

    void run(Lane& lane) noexcept;
    void execute(anira::InferenceHandler& handler, Job& job) noexcept;
    void finish(std::uint32_t slot) noexcept;
    void drop_holder(std::uint32_t slot) noexcept;
    void stop();

    anira::Context& m_context;
    std::shared_ptr<anira::Engine> m_engine;  // OnnxEngine; null without ONNX Runtime
    std::string m_hard_json;
    anira::ModelConfig m_model;
    std::vector<Spec> m_inputs;
    std::vector<Spec> m_outputs;
    std::array<anira::Tensor, k_max_tensors> m_bound{};
    bool m_has_state = false;
    anira::Async m_contract{};

    std::vector<Lane> m_lanes;
    std::unique_ptr<Job[]> m_jobs;  // NOLINT(modernize-avoid-c-arrays): atomics do not move
    std::uint32_t m_capacity = 0;
    std::atomic<std::uint64_t> m_tail{0};  // next job to submit
    std::atomic<std::uint64_t> m_head{0};  // next job a worker takes
    std::counting_semaphore<> m_queued{0};
    std::atomic<bool> m_stopping{false};
    std::atomic<std::uint64_t> m_resets{0};  // the resets asked for

    // Completed slots, in order, for poll: a ring as long as the job slots.
    std::unique_ptr<std::atomic<std::uint32_t>[]> m_completed;  // NOLINT(modernize-avoid-c-arrays)
    std::atomic<std::uint64_t> m_completed_tail{0};
    std::uint64_t m_completed_head = 0;
    std::function<void(void*, TicketStatus)> m_on_complete;
};

}  // namespace anira::models::core
