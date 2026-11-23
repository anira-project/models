// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/Error.h>
#include <anira/models/core/Model.h>
#include <anira/models/core/ModelSource.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace anira::models::core {

namespace {

constexpr std::uint32_t k_no_slot = std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t k_default_in_flight = 4;
// Without a deadline: the budget anira checks against.
constexpr std::chrono::milliseconds k_default_budget{1000};

anira_dtype dtype_named(const std::string& name) {
    static const std::map<std::string, anira_dtype> k_dtypes{
        {"float32", ANIRA_DTYPE_F32},
        {"float64", ANIRA_DTYPE_F64},
        {"float16", ANIRA_DTYPE_F16},
        {"bfloat16", ANIRA_DTYPE_BF16},
        {"int8", ANIRA_DTYPE_I8},
        {"uint8", ANIRA_DTYPE_U8},
        {"int16", ANIRA_DTYPE_I16},
        {"int32", ANIRA_DTYPE_I32},
        {"int64", ANIRA_DTYPE_I64},
        {"bool", ANIRA_DTYPE_BOOL8}};
    const auto found = k_dtypes.find(name);
    if (found == k_dtypes.end()) { throw Error("AsyncHandler: unknown dtype '" + name + "'"); }
    return found->second;
}

// A Buffer tensor as a Streamed one with the same shape: an axis of 1 is its channel, the last
// other axis its Time axis, and one window is the whole tensor.
std::int64_t to_streamed(nlohmann::json& tensor) {
    auto& axes = tensor.at("axes");
    std::int64_t window = 1;
    std::optional<std::size_t> channel;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        const auto extent = axes[i].at(1).get<std::int64_t>();
        window *= extent;
        if (!channel && extent == 1) { channel = i; }
    }
    if (!channel) {
        throw Error("AsyncHandler: Buffer tensor '" + tensor.at("name").get<std::string>() +
                    "' has no axis of 1 to stream it on");
    }
    std::optional<std::size_t> time;
    for (std::size_t i = axes.size(); i-- > 0;) {
        if (i != *channel) {
            time = i;
            break;
        }
    }
    for (std::size_t i = 0; i < axes.size(); ++i) {
        axes[i][0] = i == *channel ? "channel" : (time && i == *time ? "time" : "any");
    }
    if (!time) { axes[*channel][0] = "time"; }  // a single axis of 1
    tensor["role"] = "streamed";
    tensor["window"] = {{"min", window}, {"max", window}};
    return window;
}

std::byte* host_data(const anira::Tensor& tensor) noexcept {
    return static_cast<std::byte*>(anira_tensor_data(&tensor, tensor.dtype));
}

anira::Tensor flat(std::byte* data, anira_dtype dtype, std::int64_t window) noexcept {
    const std::array<std::int64_t, 2> shape{1, window};
    return anira::Tensor::from_host(data, dtype, shape);
}

anira::Tensor empty(anira_dtype dtype) noexcept {
    const std::array<std::int64_t, 1> shape{0};
    anira::Tensor tensor{};
    anira_tensor_init_host(&tensor, nullptr, dtype, 1, shape.data());
    return tensor;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Ticket

Ticket::~Ticket() { release(); }

Ticket::Ticket(Ticket&& other) noexcept
    : m_owner(std::exchange(other.m_owner, nullptr))
    , m_slot(other.m_slot)
    , m_refused(other.m_refused) {}

Ticket& Ticket::operator=(Ticket&& other) noexcept {
    if (this != &other) {
        release();
        m_owner = std::exchange(other.m_owner, nullptr);
        m_slot = other.m_slot;
        m_refused = other.m_refused;
    }
    return *this;
}

void Ticket::release() noexcept {
    if (m_owner != nullptr) {
        m_owner->drop_holder(m_slot);
        m_owner = nullptr;
    }
}

TicketStatus Ticket::status() const noexcept {
    if (m_owner == nullptr) { return TicketStatus::Failed; }
    return m_owner->m_jobs[m_slot].m_status.load(std::memory_order_acquire);
}

TicketStatus Ticket::wait() {
    if (m_owner == nullptr) { return TicketStatus::Failed; }
    auto& status = m_owner->m_jobs[m_slot].m_status;
    status.wait(TicketStatus::Pending, std::memory_order_acquire);
    return status.load(std::memory_order_acquire);
}

TicketStatus Ticket::wait_for(std::chrono::nanoseconds timeout) {
    if (m_owner == nullptr) { return TicketStatus::Failed; }
    const auto until = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const TicketStatus now = status();
        if (now != TicketStatus::Pending || std::chrono::steady_clock::now() >= until) {
            return now;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void* Ticket::user_data() const noexcept {
    return m_owner == nullptr ? nullptr : m_owner->m_jobs[m_slot].m_user_data;
}

anira_status Ticket::error() const noexcept {
    if (m_owner == nullptr) { return m_refused == ANIRA_OK ? ANIRA_ERROR_INVALID_STATE : m_refused; }
    return m_owner->m_jobs[m_slot].m_error;
}

// ---------------------------------------------------------------------------------------------
// AsyncHandler

AsyncHandler::AsyncHandler(anira::Context& context,
                           std::string_view model_json,
                           const ModelSource& weights)
    : m_context(context) {
    auto description = nlohmann::json::parse(model_json);
    std::optional<std::string> anchor;
    for (const char* side : {"inputs", "outputs"}) {
        auto& specs = side == std::string_view("inputs") ? m_inputs : m_outputs;
        for (auto& tensor : description.at(side)) {
            Spec spec;
            spec.m_dtype = dtype_named(tensor.at("dtype").get<std::string>());
            const auto role = tensor.at("role").get<std::string>();
            if (role == "buffer" || role == "streamed") {
                spec.m_role = Role::Streamed;
                spec.m_window = role == "buffer" ? to_streamed(tensor)
                                                 : tensor.at("window").at("max").get<std::int64_t>();
                if (!anchor && side == std::string_view("inputs")) {
                    anchor = tensor.at("name").get<std::string>();
                }
            } else if (role == "state") {
                spec.m_role = Role::State;
                m_has_state = true;
            }
            specs.push_back(spec);
        }
    }
    if (!anchor) { throw Error("AsyncHandler: the model has no Buffer or Streamed input"); }
    if (m_inputs.size() > k_max_tensors || m_outputs.size() > k_max_tensors) {
        throw Error("AsyncHandler: more than 16 tensors on one side");
    }
    description["anchor"] = *anchor;
    m_hard_json = description.dump();
    m_model = load_model(m_hard_json, weights);
    // A lane runs one job at a time: one loaded model per lane, not one per thread of anira's pool
    // (a stateless model's default).
    m_model.max_instances(1);
}

AsyncHandler::~AsyncHandler() { stop(); }

void AsyncHandler::stop() {
    m_stopping.store(true, std::memory_order_release);
    m_queued.release(static_cast<std::ptrdiff_t>(m_lanes.size()));
    for (Lane& lane : m_lanes) {
        if (lane.m_thread.joinable()) { lane.m_thread.join(); }
    }
    m_lanes.clear();
    m_stopping.store(false, std::memory_order_release);
}

anira::PlanReport AsyncHandler::prepare(const anira::Async& contract) {
    stop();
    m_contract = contract;
    const std::uint32_t lanes = std::max(contract.lanes, 1U);
    if (m_has_state && lanes > 1) {
        throw Error("AsyncHandler: a model with State runs on one lane");
    }
    m_capacity = contract.max_in_flight == 0 ? k_default_in_flight : contract.max_in_flight;
    m_jobs = std::make_unique<Job[]>(m_capacity);  // NOLINT(modernize-avoid-c-arrays)
    m_completed = std::make_unique<std::atomic<std::uint32_t>[]>(m_capacity);  // NOLINT
    for (std::uint32_t i = 0; i < m_capacity; ++i) { m_completed[i].store(k_no_slot); }
    m_tail.store(0);
    m_head.store(0);
    m_completed_tail.store(0);
    m_completed_head = 0;
    while (m_queued.try_acquire()) {}

    const auto anchor = std::ranges::find_if(m_inputs, [](const Spec& spec) {
        return spec.m_role == Role::Streamed;
    });
    const auto anchor_window = static_cast<double>(anchor->m_window);
    anira::Hard hard;
    hard.host_block_min = anchor_window;
    hard.host_block_max = anchor_window;
    const auto budget = contract.deadline.value_or(k_default_budget);
    // One job per two budgets: the rate anira sizes its rings and checks its budget against (a
    // job takes its budget at most, within a period of twice that).
    const double jobs_per_second = 0.5e9 / static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(budget).count());
    hard.host_rate = anchor_window * jobs_per_second;
    hard.budget = budget;
    // No warm-up pass: jobs have no real-time budget to measure, and a large model's cold first
    // inference would outlast anira's warm-up limit (10 s); the first job pays it instead.
    hard.warmup = ANIRA_WARMUP_NONE;
    hard.on_miss = ANIRA_MISS_ZEROS;
    const auto description = nlohmann::json::parse(m_hard_json);
    for (std::size_t i = 0; i < m_outputs.size(); ++i) {
        if (m_outputs[i].m_role == Role::Streamed) {
            hard.latencies[description.at("outputs")[i].at("name").get<std::string>()] = 0;
        }
    }

    m_lanes.resize(lanes);
    std::optional<anira::PlanReport> report;
    for (Lane& lane : m_lanes) {
        lane.m_handler = std::make_unique<anira::InferenceHandler>(
            m_context, anira::Pipeline().add_inference(m_model));
        anira::PlanReport lane_report = lane.m_handler->prepare(hard);
        if (!report) { report.emplace(lane_report); }
    }
    for (Lane& lane : m_lanes) {
        lane.m_thread = std::thread([this, &lane] { run(lane); });
    }
    return *report;
}

void AsyncHandler::bind_output(std::uint32_t slot, const anira::Tensor& out) {
    if (slot >= m_outputs.size()) { throw Error("AsyncHandler::bind_output: no such output"); }
    m_bound.at(slot) = out;
}

void AsyncHandler::on_complete(std::function<void(void*, TicketStatus)> callback) {
    m_on_complete = std::move(callback);
}

Ticket AsyncHandler::submit(std::span<const anira::Tensor> inputs,
                            const anira::JobOptions& /*options*/,
                            std::optional<std::chrono::steady_clock::time_point> deadline,
                            void* user_data) noexcept {
    if (m_lanes.empty()) { return Ticket(ANIRA_ERROR_INVALID_STATE); }
    if (inputs.size() != m_inputs.size()) { return Ticket(ANIRA_ERROR_INVALID_ARGUMENT); }
    const std::uint64_t sequence = m_tail.load(std::memory_order_relaxed);
    const auto slot = static_cast<std::uint32_t>(sequence % m_capacity);
    Job& job = m_jobs[slot];
    if (job.m_holders.load(std::memory_order_acquire) != 0) {
        return Ticket(ANIRA_ERROR_CAPACITY);
    }
    std::copy(inputs.begin(), inputs.end(), job.m_inputs.begin());
    if (!deadline && m_contract.deadline) {
        deadline = std::chrono::steady_clock::now() +
                   std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                       *m_contract.deadline);
    }
    job.m_deadline = deadline;
    job.m_user_data = user_data;
    job.m_error = ANIRA_OK;
    job.m_status.store(TicketStatus::Pending, std::memory_order_relaxed);
    job.m_holders.store(m_on_complete ? 3 : 2, std::memory_order_relaxed);
    m_tail.store(sequence + 1, std::memory_order_release);
    m_queued.release();
    return {this, slot};
}

std::uint32_t AsyncHandler::poll() {
    std::uint32_t done = 0;
    while (m_completed_head < m_completed_tail.load(std::memory_order_acquire)) {
        auto& entry = m_completed[m_completed_head % m_capacity];
        std::uint32_t slot = entry.load(std::memory_order_acquire);
        while (slot == k_no_slot) {  // the worker has its place but has not written it yet
            std::this_thread::yield();
            slot = entry.load(std::memory_order_acquire);
        }
        entry.store(k_no_slot, std::memory_order_relaxed);
        ++m_completed_head;
        Job& job = m_jobs[slot];
        if (m_on_complete) { m_on_complete(job.m_user_data, job.m_status.load()); }
        drop_holder(slot);
        ++done;
    }
    return done;
}

void AsyncHandler::reset() noexcept {
    for (Lane& lane : m_lanes) { lane.m_handler->reset(); }
}

anira_status AsyncHandler::set_plan(std::uint32_t plan) noexcept {
    anira_status status = ANIRA_OK;
    for (Lane& lane : m_lanes) {
        const anira_status lane_status = lane.m_handler->set_plan(plan);
        if (ANIRA_FAILED(lane_status)) { status = lane_status; }
    }
    return status;
}

anira::PlanReport AsyncHandler::plan_report() const {
    if (m_lanes.empty()) { return anira::PlanReport(nullptr); }
    return m_lanes.front().m_handler->plan_report();
}

void AsyncHandler::run(Lane& lane) noexcept {
    for (;;) {
        m_queued.acquire();
        if (m_stopping.load(std::memory_order_acquire)) { return; }
        const std::uint64_t sequence = m_head.fetch_add(1, std::memory_order_acq_rel);
        const auto slot = static_cast<std::uint32_t>(sequence % m_capacity);
        execute(*lane.m_handler, m_jobs[slot]);
        finish(slot);
    }
}

void AsyncHandler::execute(anira::InferenceHandler& handler, Job& job) noexcept {
    std::array<anira::Tensor, k_max_tensors> pushed{};
    for (std::size_t i = 0; i < m_inputs.size(); ++i) {
        const Spec& spec = m_inputs[i];
        if (spec.m_role == Role::Streamed) {
            pushed[i] = flat(host_data(job.m_inputs[i]), spec.m_dtype, spec.m_window);
        } else if (spec.m_role == Role::State) {
            pushed[i] = empty(spec.m_dtype);
        } else {
            pushed[i] = job.m_inputs[i];
        }
    }
    // anira frees a job's entry just after its result is popped: a push right behind it may find
    // none free yet. The worker is no real-time thread: it retries for a while.
    const std::span<const anira::Tensor> push(pushed.data(), m_inputs.size());
    anira_status status = handler.push_data(push);
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (status == ANIRA_ERROR_CAPACITY && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        status = handler.push_data(push);
    }
    if (ANIRA_FAILED(status)) {
        job.m_error = status;
        job.m_status.store(TicketStatus::Failed, std::memory_order_release);
        return;
    }

    double timeout_ms = ANIRA_WAIT_FOREVER;
    if (job.m_deadline && m_contract.on_late == ANIRA_LATE_DROP) {
        timeout_ms = std::max(
            0.0, std::chrono::duration<double, std::milli>(*job.m_deadline -
                                                           std::chrono::steady_clock::now())
                     .count());
    }
    bool missed = false;
    bool waited = false;
    for (std::size_t i = 0; i < m_outputs.size(); ++i) {
        const Spec& spec = m_outputs[i];
        const auto out_slot = static_cast<std::uint32_t>(i);
        if (spec.m_role == Role::Streamed) {
            const anira::Tensor out = flat(host_data(m_bound[i]), spec.m_dtype, spec.m_window);
            // The first output waits for the job's inference; the others came with it.
            status = waited ? handler.pop_data(out, out_slot)
                            : handler.pop_data_wait(out, out_slot, timeout_ms);
            waited = true;
            if (status == ANIRA_MISSED) {
                missed = true;
            } else if (ANIRA_FAILED(status)) {
                job.m_error = status;
            }
        } else if (spec.m_role == Role::Static && m_bound[i].ndim > 0) {
            handler.get_static_output(out_slot, m_bound[i]);
        }
    }
    TicketStatus result = TicketStatus::Met;
    if (job.m_error != ANIRA_OK) {
        result = TicketStatus::Failed;
    } else if (missed) {
        result = TicketStatus::Dropped;
    } else if (job.m_deadline && std::chrono::steady_clock::now() > *job.m_deadline) {
        result = TicketStatus::Late;
    }
    job.m_status.store(result, std::memory_order_release);
}

void AsyncHandler::finish(std::uint32_t slot) noexcept {
    m_jobs[slot].m_status.notify_all();
    if (m_on_complete) {
        const std::uint64_t place = m_completed_tail.fetch_add(1, std::memory_order_acq_rel);
        m_completed[place % m_capacity].store(slot, std::memory_order_release);
    }
    drop_holder(slot);
}

void AsyncHandler::drop_holder(std::uint32_t slot) noexcept {
    m_jobs[slot].m_holders.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace anira::models::core
