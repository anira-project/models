// SPDX-License-Identifier: Apache-2.0

// The Core ML exports through core::coreml_engine() (temporary, until anira has a Core ML engine),
// against ONNX Runtime, on every compute unit Core ML picks (the descriptions pin the provider
// "coreml"):
//
//   - every effect wrapper runs the same plucked-string signal on both, as a plugin streams it
//     (resampled, block by block, its state carried), and the outputs agree;
//   - Silero VAD's probabilities agree on the same audio;
//   - every description for anira's Async contract with a Core ML entry (hand tracking, Moonshine,
//     Parakeet, Nemotron diarization) runs the same jobs through core::AsyncHandler on both,
//     several in a row so that a State is carried, and every output agrees: within the Core ML
//     tolerance, within quantization error against a quantized ONNX reference (q4, int8), and
//     printed, not checked, against Parakeet's preprocessor from onnx-asr, which is no export of
//     NeMo's.
//
// A Core ML export that loses state or reads its inputs wrongly diverges. Built with
// ANIRA_MODELS_COREML_ENGINE on macOS; each needs ANIRA_MODELS_TEST_WEIGHTS/<its HF repository>
// with its ONNX, LiteRT and Core ML files, else it skips.

#include <anira/abi/config.h>
#include <anira/anira.h>
#include <anira/models/core/AsyncHandler.h>
#include <anira/models/core/AudioEffect.h>
#include <anira/models/core/CoreMlEngine.h>
#include <anira/models/core/ModelSource.h>
#include <anira/models/core/OnnxEngine.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <anira/anira.hpp>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <numbers>
#include <optional>
#include <random>
#include <ratio>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef ANIRA_MODELS_HAS_SILERO_VAD
#include <anira/models/silero_vad/Vad.h>
#endif
#ifdef ANIRA_MODELS_HAS_GUITAR_LSTM
#include <anira/models/guitar_lstm/GuitarLstm.h>
#endif
#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
#include <anira/models/stateful_lstm/StatefulLstm.h>
#endif
#ifdef ANIRA_MODELS_HAS_STEERABLE_NAFX
#include <anira/models/steerable_nafx/SteerableNafx.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_DJEMBE
#include <anira/models/rave_djembe/RaveDjembe.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_FUNK_DRUM
#include <anira/models/rave_funk_drum/RaveFunkDrum.h>
#endif
#ifdef ANIRA_MODELS_HAS_RAVE_MAGNETIC_RESONATOR_PIANO
#include <anira/models/rave_magnetic_resonator_piano/RaveMagneticResonatorPiano.h>
#endif

namespace core = anira::models::core;
namespace fs = std::filesystem;

namespace {

constexpr double k_rate = 48000.0;
// Relative to the reference's peak, as tools/check_streaming.py's TOLERANCE_COREML: Core ML may
// run in float16 on the GPU and the Neural Engine.
constexpr double k_tolerance = 2e-2;

std::optional<fs::path> weights_dir(const std::string& repo, const std::string& package) {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    fs::path dir = fs::path(root) / repo;
    if (!fs::exists(dir / package)) { return std::nullopt; }
    return dir;
}

// Decaying plucks at 110, 165 and 220 Hz, one every 0.25 s, 1.5 s (test_Effects.cpp's).
std::vector<float> plucks() {
    std::vector<float> out(static_cast<std::size_t>(1.5 * k_rate), 0.0F);
    const std::array<double, 3> notes{110.0, 165.0, 220.0};
    for (std::size_t k = 0; k < 6; ++k) {
        const auto start = static_cast<std::size_t>(0.25 * k_rate * static_cast<double>(k));
        for (std::size_t i = start; i < out.size(); ++i) {
            const double t = static_cast<double>(i - start) / k_rate;
            out[i] += static_cast<float>(0.3 * std::exp(-4.0 * t) *
                                         std::sin(2.0 * std::numbers::pi * notes[k % 3] * t));
        }
    }
    return out;
}

// The plan's output for `input`, and the time of a second run (the first one warms up), in ms.
std::vector<float> run_timed(core::AudioEffect& effect,
                             const std::vector<float>& input,
                             double& ms) {
    effect.run(input);
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> out = effect.run(input);
    ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void compare(core::AudioEffect& effect, const char* name) {
    const anira::PlanReport report = effect.prepare(k_rate, 512);
    const auto reference = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}});
    if (!reference) {
        FAIL() << name << ": no ONNX Runtime plan";
        return;
    }
    const std::vector<float> input = plucks();
    effect.set_plan(*reference);
    double onnx_ms = 0.0;
    const std::vector<float> expected = run_timed(effect, input, onnx_ms);
    float peak = 0.0F;
    for (const float x : expected) { peak = std::max(peak, std::abs(x)); }

    {
        const auto plan = report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
                                                   .provider = ANIRA_PROVIDER_COREML,
                                                   .engine_id = core::k_coreml_engine_id}});
        if (!plan) {
            FAIL() << name << ": no Core ML plan";
            return;
        }
        effect.set_plan(*plan);
        double coreml_ms = 0.0;
        const std::vector<float> got = run_timed(effect, input, coreml_ms);
        ASSERT_EQ(got.size(), expected.size());
        double worst = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            ASSERT_TRUE(std::isfinite(got[i])) << name << " at sample " << i;
            worst = std::max(worst, static_cast<double>(std::abs(got[i] - expected[i])));
        }
        worst /= 1.0 + static_cast<double>(peak);
        std::printf("[ info ] %s on Core ML: max |Core ML - ONNX Runtime| %.2e of peak %.3f\n",
                    name,
                    worst,
                    static_cast<double>(peak));
        std::printf("[ info ] %s: %.1f s of audio in %.0f ms on ONNX Runtime, %.0f ms on Core ML\n",
                    name,
                    static_cast<double>(input.size()) / k_rate,
                    onnx_ms,
                    coreml_ms);
        EXPECT_LT(worst, k_tolerance) << name;
    }
}

// ---- Async descriptions ------------------------------------------------------------------------

constexpr int k_jobs = 4;

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::size_t size_of(anira_dtype dtype) {
    return dtype == ANIRA_DTYPE_I64 ? 8 : 4;  // the descriptions' float32, int32, int64
}

anira_dtype dtype_named(const std::string& name) {
    if (name == "int64") { return ANIRA_DTYPE_I64; }
    if (name == "int32") { return ANIRA_DTYPE_I32; }
    return ANIRA_DTYPE_F32;
}

// One tensor of a description: its dtype, shape and role, with room for its values.
struct Slot {
    std::string m_name;
    anira_dtype m_dtype = ANIRA_DTYPE_F32;
    std::vector<std::int64_t> m_shape;
    bool m_state = false;
    std::vector<std::byte> m_bytes;

    [[nodiscard]] std::size_t count() const {
        std::size_t n = 1;
        for (const std::int64_t extent : m_shape) { n *= static_cast<std::size_t>(extent); }
        return n;
    }
    [[nodiscard]] double at(std::size_t i) const {
        switch (m_dtype) {
            case ANIRA_DTYPE_I64:
                return static_cast<double>(
                    reinterpret_cast<const std::int64_t*>(m_bytes.data())[i]);
            case ANIRA_DTYPE_I32: return reinterpret_cast<const std::int32_t*>(m_bytes.data())[i];
            default: return reinterpret_cast<const float*>(m_bytes.data())[i];
        }
    }
    void set(std::size_t i, double value) {
        switch (m_dtype) {
            case ANIRA_DTYPE_I64:
                reinterpret_cast<std::int64_t*>(m_bytes.data())[i] =
                    static_cast<std::int64_t>(value);
                break;
            case ANIRA_DTYPE_I32:
                reinterpret_cast<std::int32_t*>(m_bytes.data())[i] =
                    static_cast<std::int32_t>(value);
                break;
            default: reinterpret_cast<float*>(m_bytes.data())[i] = static_cast<float>(value);
        }
    }
    anira::Tensor tensor() {
        if (m_state) {
            const std::array<std::int64_t, 1> shape{0};
            anira::Tensor empty{};
            anira_tensor_init_host(&empty, nullptr, m_dtype, 1, shape.data());
            return empty;
        }
        return anira::Tensor::from_host(m_bytes.data(),
                                        m_dtype,
                                        std::span<const std::int64_t>(m_shape));
    }
};

std::vector<Slot> slots_of(const nlohmann::json& tensors) {
    std::vector<Slot> slots;
    for (const auto& tensor : tensors) {
        Slot slot;
        slot.m_name = tensor.at("name").get<std::string>();
        slot.m_dtype = dtype_named(tensor.value("dtype", "float32"));
        for (const auto& axis : tensor.at("axes")) {
            slot.m_shape.push_back(axis[1].get<std::int64_t>());
        }
        slot.m_state = tensor.at("role") == "state";
        if (!slot.m_state) { slot.m_bytes.resize(slot.count() * size_of(slot.m_dtype)); }
        slots.push_back(std::move(slot));
    }
    return slots;
}

// A job's inputs, as tools/check_streaming.py makes them: noise for a float input; for an integer
// one, a length (a name with "len") is the last extent of the first float input, the window it
// measures, else ones.
std::vector<std::vector<std::byte>> job_inputs(std::vector<Slot>& inputs, std::mt19937& random) {
    std::int64_t length = 1;
    for (const Slot& slot : inputs) {
        if (slot.m_dtype == ANIRA_DTYPE_F32 && !slot.m_shape.empty()) {
            length = slot.m_shape.back();
            break;
        }
    }
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<std::vector<std::byte>> values;
    for (Slot& slot : inputs) {
        if (slot.m_state) {
            values.emplace_back();
            continue;
        }
        const bool is_length = slot.m_name.find("len") != std::string::npos;
        for (std::size_t i = 0; i < slot.count(); ++i) {
            slot.set(i,
                     slot.m_dtype == ANIRA_DTYPE_F32 ? noise(random)
                                                     : static_cast<double>(is_length ? length : 1));
        }
        values.push_back(slot.m_bytes);
    }
    return values;
}

// Every output of every job (the State ones aside), as doubles.
using Outputs = std::vector<std::vector<std::vector<double>>>;

Outputs run_jobs(core::AsyncHandler& handler,
                 std::uint32_t plan,
                 std::vector<Slot>& inputs,
                 std::vector<Slot>& outputs,
                 const std::vector<std::vector<std::vector<std::byte>>>& jobs) {
    EXPECT_EQ(handler.set_plan(plan), ANIRA_OK);
    handler.reset();
    Outputs result;
    for (const auto& job : jobs) {
        std::vector<anira::Tensor> tensors;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i].m_state) { inputs[i].m_bytes = job[i]; }
            tensors.push_back(inputs[i].tensor());
        }
        core::Ticket ticket = handler.submit(tensors);
        EXPECT_EQ(ticket.wait(), core::TicketStatus::Met) << ticket.error();
        auto& values = result.emplace_back();
        for (const Slot& slot : outputs) {
            if (slot.m_state) { continue; }
            auto& out = values.emplace_back(slot.count());
            for (std::size_t i = 0; i < out.size(); ++i) { out[i] = slot.at(i); }
        }
    }
    return result;
}

// The worst difference of any output of any job, relative to the reference output's peak; infinite
// for a value that is not finite.
double deviation(const Outputs& reference, const Outputs& got) {
    double worst = 0.0;
    for (std::size_t j = 0; j < reference.size(); ++j) {
        for (std::size_t o = 0; o < reference[j].size(); ++o) {
            double peak = 0.0;
            double diff = 0.0;
            for (std::size_t i = 0; i < reference[j][o].size(); ++i) {
                const double r = reference[j][o][i];
                const double g = got[j][o][i];
                if (!std::isfinite(r) || !std::isfinite(g)) { return INFINITY; }
                peak = std::max(peak, std::abs(r));
                diff = std::max(diff, std::abs(r - g));
            }
            worst = std::max(worst, diff / (1.0 + peak));
        }
    }
    return worst;
}

// A quantized ONNX file (q4, int8) against the float Core ML export: quantization error, measured
// up to 0.14 (tools/check_streaming.py's TOLERANCE_QUANTIZED).
constexpr double k_tolerance_quantized = 0.2;

bool is_quantized(const std::string& onnx) {
    return onnx.find("q4") != std::string::npos || onnx.find("int8") != std::string::npos;
}

// Why an ONNX file is no export of the original model (null: it is one): Parakeet's preprocessor
// from onnx-asr (nemo128.onnx) counts one mel frame more than NeMo (len / 160 + 1) and differs at
// the edge frames; the Core ML export follows NeMo (export/check.py).
const char* not_exact(const std::string& onnx) {
    if (onnx == "nemo128.onnx") { return "onnx-asr's preprocessor, one frame longer than NeMo's"; }
    return nullptr;
}

bool is_async(const fs::path& description) {
    const std::string stem = description.string().substr(
        0,
        description.string().size() - std::string_view(".model.json").size());
    fs::path contract = stem + ".contract.json";
    if (!fs::exists(contract)) {
        std::vector<fs::path> shared;
        for (const auto& entry : fs::directory_iterator(description.parent_path())) {
            if (entry.path().filename().string().ends_with(".contract.json")) {
                shared.push_back(entry.path());
            }
        }
        if (shared.size() != 1) { return false; }
        contract = shared.front();
    }
    const anira::ContractHandle handle = anira::ContractHandle::from_json(read_text(contract));
    return anira_contract_get_kind(handle.native()) == ANIRA_CONTRACT_ASYNC;
}

// The weights' folder of a description's entries: its folder's catalog's HF repository.
std::optional<fs::path> repo_dir(const fs::path& folder) {
    const char* root = std::getenv("ANIRA_MODELS_TEST_WEIGHTS");
    if (root == nullptr) { return std::nullopt; }
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.path().filename().string().ends_with(".catalog.json")) { continue; }
        const auto repo =
            nlohmann::json::parse(read_text(entry.path())).at("repo").get<std::string>();
        return fs::path(root) / repo.substr(repo.find('/') + 1);
    }
    return std::nullopt;
}

}  // namespace

#ifdef ANIRA_MODELS_HAS_GUITAR_LSTM
TEST(CoreMl, GuitarLstm) {
    const auto dir = weights_dir("guitar-lstm", "coreml/GuitarLSTM.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/guitar-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::guitar_lstm::GuitarLstm effect(context, core::ModelSource::directory(*dir));
    compare(effect, "GuitarLSTM");
}
#endif

#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
TEST(CoreMl, StatefulLstm) {
    const auto dir = weights_dir("stateful-lstm", "coreml/stateful_lstm.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/stateful-lstm is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::stateful_lstm::StatefulLstm effect(context, core::ModelSource::directory(*dir));
    compare(effect, "Stateful LSTM");
}
#endif

#ifdef ANIRA_MODELS_HAS_STATEFUL_LSTM
// Compiled-in weights (a bytes source, as the examples embed the small effects): load_model writes
// the package to disk for the engine, and the Core ML plan agrees as from a directory.
TEST(CoreMl, StatefulLstmFromBytes) {
    const auto dir = weights_dir("stateful-lstm", "coreml/stateful_lstm.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/stateful-lstm is missing"; }
    std::vector<std::string> contents;  // the bytes the source borrows
    std::vector<core::ModelSource::File> files;
    for (const auto& entry : fs::recursive_directory_iterator(*dir)) {
        if (!entry.is_regular_file()) { continue; }
        contents.push_back(read_text(entry.path()));
    }
    std::size_t i = 0;
    for (const auto& entry : fs::recursive_directory_iterator(*dir)) {
        if (!entry.is_regular_file()) { continue; }
        files.push_back({.m_path = fs::relative(entry.path(), *dir).generic_string(),
                         .m_bytes = std::as_bytes(std::span(contents[i++]))});
    }
    anira::Context context{anira::ContextConfig{}};
    anira::models::stateful_lstm::StatefulLstm effect(context,
                                                      core::ModelSource::bytes(std::move(files)));
    compare(effect, "Stateful LSTM from bytes");
}
#endif

#ifdef ANIRA_MODELS_HAS_STEERABLE_NAFX
TEST(CoreMl, SteerableNafxInEverySize) {
    const auto dir = weights_dir("steerable-nafx", "coreml/steerable-nafx.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/steerable-nafx is missing"; }
    using anira::models::steerable_nafx::SteerableNafx;
    for (const auto size : {SteerableNafx::Size::Full,
                            SteerableNafx::Size::ThreeBlocks,
                            SteerableNafx::Size::TwoBlocks}) {
        anira::Context context{anira::ContextConfig{}};
        SteerableNafx effect(context, core::ModelSource::directory(*dir), size);
        compare(effect, "Steerable NAFX");
    }
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_DJEMBE
TEST(CoreMl, RaveDjembe) {
    const auto dir = weights_dir("rave-djembe", "coreml/rave_forward.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-djembe is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_djembe::RaveDjembe effect(context, core::ModelSource::directory(*dir));
    compare(effect, "RAVE Djembe");
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_FUNK_DRUM
TEST(CoreMl, RaveFunkDrum) {
    const auto dir = weights_dir("rave-funk-drum", "coreml/rave_funk_drum_forward.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-funk-drum is missing"; }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_funk_drum::RaveFunkDrum effect(context, core::ModelSource::directory(*dir));
    compare(effect, "RAVE Funk Drum");
}
#endif

#ifdef ANIRA_MODELS_HAS_RAVE_MAGNETIC_RESONATOR_PIANO
TEST(CoreMl, RaveMagneticResonatorPiano) {
    const auto dir =
        weights_dir("rave-magnetic-resonator-piano", "coreml/mrp_strengjavera_forward.mlpackage");
    if (!dir) {
        GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/rave-magnetic-resonator-piano is missing";
    }
    anira::Context context{anira::ContextConfig{}};
    anira::models::rave_magnetic_resonator_piano::RaveMagneticResonatorPiano effect(
        context,
        core::ModelSource::directory(*dir));
    compare(effect, "RAVE Magnetic Resonator Piano");
}
#endif

TEST(CoreMl, EveryAsyncDescriptionAgreesWithOnnxRuntime) {
    if (std::getenv("ANIRA_MODELS_TEST_WEIGHTS") == nullptr) {
        GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS is not set";
    }
    std::vector<fs::path> descriptions;
    for (const auto& folder : fs::directory_iterator(ANIRA_MODELS_SOURCE_DIR)) {
        if (!folder.is_directory()) { continue; }
        for (const auto& entry : fs::directory_iterator(folder.path())) {
            if (entry.path().filename().string().ends_with(".model.json")) {
                descriptions.push_back(entry.path());
            }
        }
    }
    std::ranges::sort(descriptions);

    int compared = 0;
    for (const fs::path& path : descriptions) {
        const std::string text = read_text(path);
        const auto description = nlohmann::json::parse(text);
        std::string package;
        std::string onnx;
        for (const auto& entry : description.at("models")) {
            if (entry.at("engine") == core::k_coreml_engine_id) {
                package = entry.at("path").get<std::string>();
            } else if (entry.at("engine") == "onnxruntime" && onnx.empty()) {
                onnx = entry.at("path").get<std::string>();
            }
        }
        if (package.empty() || onnx.empty() || !is_async(path)) { continue; }
        const auto dir = repo_dir(path.parent_path());
        const std::string name = path.filename().string();
        if (!dir || !fs::exists(*dir / package) || !fs::exists(*dir / onnx)) {
            std::printf("[ skip ] %s: weights missing\n", name.c_str());
            continue;
        }

        anira::Context context{anira::ContextConfig{}};
        core::AsyncHandler handler(context, text, core::ModelSource::directory(*dir));
        const anira::PlanReport report = handler.prepare(anira::Async{});
        auto reference = report.find(
            {.backend = {.engine = ANIRA_ENGINE_CUSTOM, .engine_id = core::k_onnx_engine_id}});
        if (!reference) {
            reference = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}});
        }
        const auto coreml = report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
                                                     .provider = ANIRA_PROVIDER_COREML,
                                                     .engine_id = core::k_coreml_engine_id}});
        if (!reference || !coreml) {
            FAIL() << name << ": no ONNX Runtime or no Core ML plan";
            return;
        }

        std::vector<Slot> inputs = slots_of(description.at("inputs"));
        std::vector<Slot> outputs = slots_of(description.at("outputs"));
        for (std::uint32_t o = 0; o < outputs.size(); ++o) {
            if (!outputs[o].m_state) { handler.bind_output(o, outputs[o].tensor()); }
        }
        std::mt19937 random(0);
        std::vector<std::vector<std::vector<std::byte>>> jobs;
        jobs.reserve(k_jobs);
        for (int k = 0; k < k_jobs; ++k) { jobs.push_back(job_inputs(inputs, random)); }

        const auto start = std::chrono::steady_clock::now();
        const Outputs expected = run_jobs(handler, *reference, inputs, outputs, jobs);
        const auto middle = std::chrono::steady_clock::now();
        const Outputs got = run_jobs(handler, *coreml, inputs, outputs, jobs);
        const auto end = std::chrono::steady_clock::now();
        const double worst = deviation(expected, got);
        const char* unchecked = not_exact(onnx);
        const double tolerance = is_quantized(onnx) ? k_tolerance_quantized : k_tolerance;
        std::printf(
            "[ info ] %s: max |Core ML - ONNX Runtime| %.2e%s%s%s%s; %d jobs in %.0f ms on ONNX "
            "Runtime, %.0f ms on Core ML\n",
            name.c_str(),
            worst,
            is_quantized(onnx) ? " (quantized ONNX)" : "",
            unchecked != nullptr ? " (not checked: " : "",
            unchecked != nullptr ? unchecked : "",
            unchecked != nullptr ? ")" : "",
            k_jobs,
            std::chrono::duration<double, std::milli>(middle - start).count(),
            std::chrono::duration<double, std::milli>(end - middle).count());
        if (unchecked != nullptr) {
            EXPECT_TRUE(std::isfinite(worst)) << name;
        } else {
            EXPECT_LT(worst, tolerance) << name;
        }
        ++compared;
    }
    if (compared == 0) { GTEST_SKIP() << "no Async description with its weights here"; }
}

#ifdef ANIRA_MODELS_HAS_SILERO_VAD
TEST(CoreMl, SileroVad) {
    const auto dir = weights_dir("silero-vad", "coreml/silero_vad_16k.mlpackage");
    if (!dir) { GTEST_SKIP() << "ANIRA_MODELS_TEST_WEIGHTS/silero-vad is missing"; }
    // Voiced bursts: a 120 Hz pulse train through two formants, half a second on, half off.
    constexpr double rate = 16000.0;
    std::vector<float> audio(static_cast<std::size_t>(4 * rate));
    double f1 = 0.0;
    double f2 = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;
    std::mt19937 random(1);
    std::normal_distribution<double> noise(0.0, 0.002);
    for (std::size_t i = 0; i < audio.size(); ++i) {
        const bool on = (i / static_cast<std::size_t>(rate / 2)) % 2 == 0;
        const double pulse = on && i % static_cast<std::size_t>(rate / 120) == 0 ? 1.0 : 0.0;
        // Two resonators at 700 and 1200 Hz.
        const double r = 0.97;
        const double c1 = 2.0 * r * std::cos(2.0 * std::numbers::pi * 700.0 / rate);
        const double c2 = 2.0 * r * std::cos(2.0 * std::numbers::pi * 1200.0 / rate);
        const double y1 = pulse + c1 * f1 - r * r * v1;
        v1 = f1;
        f1 = y1;
        const double y2 = y1 + c2 * f2 - r * r * v2;
        v2 = f2;
        f2 = y2;
        audio[i] = static_cast<float>(0.02 * y2 + noise(random));
    }

    anira::Context context{anira::ContextConfig{}};
    anira::models::silero_vad::Vad vad(context, core::ModelSource::directory(*dir));
    const anira::PlanReport report = vad.prepare(rate, 512);
    const auto reference = report.find({.backend = {.engine = ANIRA_ENGINE_ONNXRUNTIME}});
    const auto coreml = report.find({.backend = {.engine = ANIRA_ENGINE_CUSTOM,
                                                 .provider = ANIRA_PROVIDER_COREML,
                                                 .engine_id = core::k_coreml_engine_id}});
    ASSERT_TRUE(reference);
    ASSERT_TRUE(coreml);
    vad.set_non_realtime(true);
    const auto probabilities = [&](std::uint32_t plan) {
        vad.set_plan(plan);
        vad.reset();
        std::vector<float> out(audio.size());
        for (std::size_t start = 0; start < audio.size(); start += 512) {
            const std::size_t n = std::min<std::size_t>(512, audio.size() - start);
            EXPECT_EQ(vad.process(std::span<const float>(audio).subspan(start, n),
                                  std::span<float>(out).subspan(start, n)),
                      ANIRA_OK);
        }
        return out;
    };
    const std::vector<float> expected = probabilities(*reference);
    const std::vector<float> got = probabilities(*coreml);
    float worst = 0.0F;
    float high = 0.0F;
    for (std::size_t i = 0; i < got.size(); ++i) {
        ASSERT_TRUE(std::isfinite(got[i]));
        worst = std::max(worst, std::abs(got[i] - expected[i]));
        high = std::max(high, expected[i]);
    }
    std::printf(
        "[ info ] Silero VAD: max |Core ML - ONNX Runtime| %.2e, probabilities up to %.2f\n",
        static_cast<double>(worst),
        static_cast<double>(high));
    EXPECT_LT(worst, static_cast<float>(k_tolerance));
}
#endif
