// SPDX-License-Identifier: Apache-2.0

#include <anira/models/core/OnnxEngine.h>

#include <anira/anira.hpp>
#include <cstdint>
#include <memory>

#ifdef ANIRA_MODELS_WITH_ONNX_ENGINE

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace anira::models::core {

namespace {

ONNXTensorElementDataType onnx_type_of(anira_dtype dtype) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case ANIRA_DTYPE_F64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case ANIRA_DTYPE_F16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case ANIRA_DTYPE_BF16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
        case ANIRA_DTYPE_I8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case ANIRA_DTYPE_U8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case ANIRA_DTYPE_I16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
        case ANIRA_DTYPE_I32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case ANIRA_DTYPE_I64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case ANIRA_DTYPE_BOOL8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

std::size_t size_of(anira_dtype dtype) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F64:
        case ANIRA_DTYPE_I64: return 8;
        case ANIRA_DTYPE_F32:
        case ANIRA_DTYPE_I32: return 4;
        case ANIRA_DTYPE_F16:
        case ANIRA_DTYPE_BF16:
        case ANIRA_DTYPE_I16: return 2;
        default: return 1;
    }
}

// One environment per process (ONNX Runtime keeps one behind every handle), alive while a session is.
std::shared_ptr<Ort::Env> shared_env() {
    static std::mutex mutex;
    static std::weak_ptr<Ort::Env> current;
    const std::scoped_lock lock(mutex);
    auto env = current.lock();
    if (!env) {
        env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "anira-models");
        current = env;
    }
    return env;
}

// What every session of one loaded model reads: the model and the options.
struct Source {
    std::shared_ptr<Ort::Env> m_env;
    Ort::SessionOptions m_options;
    std::string m_path;               // a path entry
    std::vector<std::byte> m_bytes;   // a bytes entry, copied (the record's die with load)
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

// One session: what one call at a time runs on.
class Session {
public:
    explicit Session(const Source& source) : m_memory(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        m_session = source.m_bytes.empty()
                        ? Ort::Session(*source.m_env, source.m_path.c_str(), source.m_options)
                        : Ort::Session(*source.m_env, source.m_bytes.data(), source.m_bytes.size(),
                                       source.m_options);
        bind(source.m_input_names, true, m_input_names);
        bind(source.m_output_names, false, m_output_names);
        m_inputs.reserve(m_input_names.size());
        m_outputs.reserve(m_output_names.size());
    }

    anira_status run(anira::EngineContext& ctx) noexcept {
        try {
            m_inputs.clear();
            m_outputs.clear();
            for (const anira::Tensor& tensor : ctx.inputs()) { m_inputs.push_back(value_of(tensor)); }
            for (const anira::Tensor& tensor : ctx.outputs()) { m_outputs.push_back(value_of(tensor)); }
            m_session.Run(Ort::RunOptions{nullptr}, m_input_names.data(), m_inputs.data(), m_inputs.size(),
                          m_output_names.data(), m_outputs.data(), m_outputs.size());
            return ANIRA_OK;
        } catch (...) { return ANIRA_ERROR_ENGINE; }
    }

private:
    // Each slot binds to the graph tensor of its name, else to the one at its position (anira's
    // rule for its own adapters).
    void bind(const std::vector<std::string>& names, bool inputs, std::vector<const char*>& out) {
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t count = inputs ? m_session.GetInputCount() : m_session.GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> graph;
        for (std::size_t i = 0; i < count; ++i) {
            graph.push_back(inputs ? m_session.GetInputNameAllocated(i, allocator)
                                   : m_session.GetOutputNameAllocated(i, allocator));
        }
        for (std::size_t slot = 0; slot < names.size(); ++slot) {
            std::size_t index = slot;
            for (std::size_t i = 0; i < count; ++i) {
                if (names[slot] == graph[i].get()) { index = i; }
            }
            if (index >= count) {
                throw anira::Error(ANIRA_ERROR_MODEL_LOAD, "onnx engine: the graph has no " +
                                                               std::string(inputs ? "input" : "output") +
                                                               " '" + names[slot] + "' and too few to bind by position");
            }
            out.push_back(graph[index].get());
        }
        for (auto& name : graph) { m_names.push_back(std::move(name)); }
    }

    // A tensor of this call over its memory (packed host memory, anira's rule for a CPU engine).
    Ort::Value value_of(const anira::Tensor& tensor) const {
        void* data = anira_tensor_data(&tensor, tensor.dtype);
        const std::size_t bytes = anira_tensor_num_elements(&tensor) * size_of(tensor.dtype);
        return Ort::Value::CreateTensor(m_memory, data, bytes, tensor.shape, tensor.ndim,
                                        onnx_type_of(tensor.dtype));
    }

    Ort::MemoryInfo m_memory;
    Ort::Session m_session{nullptr};
    std::vector<Ort::AllocatedStringPtr> m_names;
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
    std::vector<Ort::Value> m_inputs;
    std::vector<Ort::Value> m_outputs;
};

class SharedModel final : public anira::Engine::Loaded {
public:
    SharedModel(std::unique_ptr<Source> source, std::uint32_t instances) : m_source(std::move(source)) {
        for (std::uint32_t i = 0; i < instances; ++i) { m_sessions.push_back(std::make_unique<Session>(*m_source)); }
    }

    std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override;

    std::unique_ptr<Source> m_source;
    std::vector<std::unique_ptr<Session>> m_sessions;  // the shared slots
};

// A stateless handler's calls run on the loaded model's slots; an exclusive one's on its own session.
class Handler final : public anira::Engine::Prepared {
public:
    Handler(SharedModel& loaded, std::unique_ptr<Session> own) : m_loaded(loaded), m_own(std::move(own)) {}

    anira_status process(anira::EngineContext& ctx) noexcept override {
        if (m_own) { return m_own->run(ctx); }
        if (ctx.instance() >= m_loaded.m_sessions.size()) { return ANIRA_ERROR_INVALID_STATE; }
        return m_loaded.m_sessions[ctx.instance()]->run(ctx);
    }

private:
    SharedModel& m_loaded;
    std::unique_ptr<Session> m_own;
};

std::unique_ptr<anira::Engine::Prepared> SharedModel::prepare(const anira::PrepareInfo& info) {
    return std::make_unique<Handler>(*this, info.exclusive() ? std::make_unique<Session>(*m_source) : nullptr);
}

class OnnxEngine final : public anira::Engine {
public:
    explicit OnnxEngine(std::uint32_t threads) : anira::Engine(k_onnx_engine_id), m_threads(threads) {}

    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& info) override {
        auto source = std::make_unique<Source>();
        source->m_env = shared_env();
        source->m_options.SetIntraOpNumThreads(static_cast<int>(m_threads));
        source->m_options.SetInterOpNumThreads(1);
        source->m_options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        const auto bytes = info.model_bytes(info.row());
        if (bytes.empty()) {
            source->m_path = std::string(info.model_path(info.row()));
        } else {
            source->m_bytes.assign(bytes.begin(), bytes.end());
        }
        for (const char* name : info.input_names()) { source->m_input_names.emplace_back(name); }
        for (const char* name : info.output_names()) { source->m_output_names.emplace_back(name); }
        try {
            return std::make_unique<SharedModel>(std::move(source), info.instances());
        } catch (const Ort::Exception& error) {
            throw anira::Error(ANIRA_ERROR_MODEL_LOAD, std::string("onnx engine: ") + error.what());
        }
    }

private:
    std::uint32_t m_threads;
};

}  // namespace

std::shared_ptr<anira::Engine> make_onnx_engine(std::uint32_t threads) {
    return std::make_shared<OnnxEngine>(threads);
}

}  // namespace anira::models::core

#else

namespace anira::models::core {

std::shared_ptr<anira::Engine> make_onnx_engine(std::uint32_t /*threads*/) { return nullptr; }

}  // namespace anira::models::core

#endif
