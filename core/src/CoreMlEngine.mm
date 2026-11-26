// SPDX-License-Identifier: Apache-2.0

#include <anira/anira.h>
#include <anira/models/core/CoreMlEngine.h>

#include <anira/anira.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

namespace anira::models::core {

namespace {

std::string text_of(NSError* error) {
    return error != nil ? std::string(error.localizedDescription.UTF8String) : "unknown error";
}

// An element of a buffer as double, and back: the conversions between anira's data types and
// Core ML's (int64 <-> int32, float32 <-> float16).
double load(const void* data, std::size_t i, anira_dtype dtype) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F32: return static_cast<const float*>(data)[i];
        case ANIRA_DTYPE_F64: return static_cast<const double*>(data)[i];
        case ANIRA_DTYPE_F16: return static_cast<double>(static_cast<const _Float16*>(data)[i]);
        case ANIRA_DTYPE_I64: return static_cast<double>(static_cast<const std::int64_t*>(data)[i]);
        case ANIRA_DTYPE_I32: return static_cast<const std::int32_t*>(data)[i];
        case ANIRA_DTYPE_I16: return static_cast<const std::int16_t*>(data)[i];
        case ANIRA_DTYPE_I8: return static_cast<const std::int8_t*>(data)[i];
        case ANIRA_DTYPE_U8:
        case ANIRA_DTYPE_BOOL8: return static_cast<const std::uint8_t*>(data)[i];
        default: return 0.0;
    }
}

void store(void* data, std::size_t i, anira_dtype dtype, double value) noexcept {
    switch (dtype) {
        case ANIRA_DTYPE_F32: static_cast<float*>(data)[i] = static_cast<float>(value); break;
        case ANIRA_DTYPE_F64: static_cast<double*>(data)[i] = value; break;
        case ANIRA_DTYPE_F16: static_cast<_Float16*>(data)[i] = static_cast<_Float16>(value); break;
        case ANIRA_DTYPE_I64:
            static_cast<std::int64_t*>(data)[i] = static_cast<std::int64_t>(value);
            break;
        case ANIRA_DTYPE_I32:
            static_cast<std::int32_t*>(data)[i] = static_cast<std::int32_t>(value);
            break;
        case ANIRA_DTYPE_I16:
            static_cast<std::int16_t*>(data)[i] = static_cast<std::int16_t>(value);
            break;
        case ANIRA_DTYPE_I8:
            static_cast<std::int8_t*>(data)[i] = static_cast<std::int8_t>(value);
            break;
        case ANIRA_DTYPE_U8:
        case ANIRA_DTYPE_BOOL8:
            static_cast<std::uint8_t*>(data)[i] = static_cast<std::uint8_t>(value);
            break;
        default: break;
    }
}

anira_dtype dtype_of(MLMultiArrayDataType type) noexcept {
    switch (type) {
        case MLMultiArrayDataTypeFloat32: return ANIRA_DTYPE_F32;
        case MLMultiArrayDataTypeDouble: return ANIRA_DTYPE_F64;
        case MLMultiArrayDataTypeFloat16: return ANIRA_DTYPE_F16;
        case MLMultiArrayDataTypeInt32: return ANIRA_DTYPE_I32;
        default: return static_cast<anira_dtype>(0);
    }
}

NSArray<NSNumber*>* shape_of(const anira::Tensor& tensor) {
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:tensor.ndim];
    for (std::uint32_t d = 0; d < tensor.ndim; ++d) { [shape addObject:@(tensor.shape[d])]; }
    if (tensor.ndim == 0) { [shape addObject:@1]; }
    return shape;
}

// What every instance of one loaded model reads: the compiled model and the binding.
struct Source {
    NSURL* m_compiled = nil;  // the .mlmodelc of the package (compiled(), kept for the process)
    MLComputeUnits m_units = MLComputeUnitsAll;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

// What identifies a package's content: its files' relative paths and sizes, the small files whole
// and of a large one (the weights) its first, middle and last MiB, and the OS (Core ML's compiled
// format follows it). FNV-1a, as hex. Path-independent: a package written per process from
// compiled-in weights hits the same compiled model.
std::string content_key(NSString* package) {
    constexpr std::uint64_t k_prime = 1099511628211ULL;
    std::uint64_t hash = 14695981039346656037ULL;
    const auto mix = [&](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i]) * k_prime; }
    };
    NSString* os = NSProcessInfo.processInfo.operatingSystemVersionString;
    mix(os.UTF8String, std::strlen(os.UTF8String));
    NSFileManager* files = NSFileManager.defaultManager;
    NSArray<NSString*>* paths = [[files subpathsOfDirectoryAtPath:package error:nil]
        sortedArrayUsingSelector:@selector(compare:)];
    constexpr unsigned long long k_whole = 4ULL << 20U;
    constexpr NSUInteger k_slice = 1U << 20U;
    for (NSString* relative in paths) {
        NSString* full = [package stringByAppendingPathComponent:relative];
        NSDictionary* attributes = [files attributesOfItemAtPath:full error:nil];
        if (![attributes.fileType isEqualToString:NSFileTypeRegular]) { continue; }
        const unsigned long long size = attributes.fileSize;
        mix(relative.UTF8String, std::strlen(relative.UTF8String));
        mix(&size, sizeof(size));
        NSFileHandle* handle = [NSFileHandle fileHandleForReadingAtPath:full];
        if (handle == nil) { continue; }
        const auto take = [&](unsigned long long offset, NSUInteger length) {
            [handle seekToFileOffset:offset];
            NSData* data = [handle readDataOfLength:length];
            mix(data.bytes, data.length);
        };
        if (size <= k_whole) {
            take(0, static_cast<NSUInteger>(size));
        } else {
            take(0, k_slice);
            take(size / 2, k_slice);
            take(size - k_slice, k_slice);
        }
        [handle closeFile];
    }
    std::array<char, 17> hex{};
    std::snprintf(hex.data(), hex.size(), "%016llx", static_cast<unsigned long long>(hash));
    return hex.data();
}

// The package at `path` compiled (.mlmodelc): anira loads a model again at every prepare (a
// plugin's every activate) and a host may run each plugin in a process of its own, while compiling
// a large package takes seconds. So the compiled model is kept in the user's caches,
// <Caches>/anira-models/coreml/<content key>.mlmodelc, shared by every process, and remembered
// in this one.
NSURL* compiled(const std::string& path) {
    static std::mutex mutex;
    static std::map<std::string, NSURL*> done;
    const std::scoped_lock lock(mutex);
    if (const auto found = done.find(path); found != done.end()) { return found->second; }
    NSString* package = @(path.c_str());
    NSFileManager* files = NSFileManager.defaultManager;
    NSURL* caches =
        [files URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask].firstObject;
    NSURL* folder = [caches URLByAppendingPathComponent:@"anira-models/coreml" isDirectory:YES];
    NSURL* cached =
        [folder URLByAppendingPathComponent:[NSString stringWithFormat:@"%s.mlmodelc",
                                                                       content_key(package).c_str()]
                                isDirectory:YES];
    if (![files fileExistsAtPath:cached.path]) {
        NSError* error = nil;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        NSURL* url = [MLModel compileModelAtURL:[NSURL fileURLWithPath:package] error:&error];
#pragma clang diagnostic pop
        if (url == nil) {
            throw anira::Error(ANIRA_ERROR_MODEL_LOAD,
                               "coreml engine: compiling " + path + ": " + text_of(error));
        }
        [files createDirectoryAtURL:folder
            withIntermediateDirectories:YES
                             attributes:nil
                                  error:nil];
        // Another process may have put it there meanwhile: then its copy serves, ours goes.
        if (![files moveItemAtURL:url toURL:cached error:nil]) {
            [files removeItemAtURL:url error:nil];
            if (![files fileExistsAtPath:cached.path]) {
                throw anira::Error(ANIRA_ERROR_MODEL_LOAD,
                                   "coreml engine: cannot keep the compiled model in " +
                                       std::string(folder.path.UTF8String));
            }
        }
    }
    done.emplace(path, cached);
    return cached;
}

// Loaded models no session holds, per compiled model and compute units, for the next session to
// take: anira loads a model again at every prepare (a plugin's every activate), and loading a
// large Core ML model takes a second or more. A model serves one session at a time, never two.
class ModelPool {
public:
    using Key = std::pair<std::string, MLComputeUnits>;

    static ModelPool& instance() {
        static ModelPool pool;
        return pool;
    }

    MLModel* take(NSURL* compiled, MLComputeUnits units) {
        {
            const std::scoped_lock lock(m_mutex);
            auto& free = m_free[Key(compiled.path.UTF8String, units)];
            if (!free.empty()) {
                MLModel* model = free.back();
                free.pop_back();
                return model;
            }
        }
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = units;
        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (model == nil) {
            throw anira::Error(ANIRA_ERROR_MODEL_LOAD, "coreml engine: " + text_of(error));
        }
        return model;
    }

    void give_back(NSURL* compiled, MLComputeUnits units, MLModel* model) {
        const std::scoped_lock lock(m_mutex);
        m_free[Key(compiled.path.UTF8String, units)].push_back(model);
    }

private:
    std::mutex m_mutex;
    std::map<Key, std::vector<MLModel*>> m_free;
};

// One MLModel: what one call at a time runs on, from the pool and back into it.
class Session {
public:
    explicit Session(const Source& source)
        : m_compiled(source.m_compiled)
        , m_units(source.m_units)
        , m_model(ModelPool::instance().take(source.m_compiled, source.m_units)) {
        NSDictionary<NSString*, MLFeatureDescription*>* inputs =
            m_model.modelDescription.inputDescriptionsByName;
        NSDictionary<NSString*, MLFeatureDescription*>* outputs =
            m_model.modelDescription.outputDescriptionsByName;
        m_bound_inputs = bind(source.m_input_names, inputs);
        m_bound_outputs = bind(source.m_output_names, outputs);
        for (std::string& name : m_bound_inputs) {
            MLFeatureDescription* feature = inputs[@(name.c_str())];
            if (feature == nil || feature.multiArrayConstraint == nil) {
                throw anira::Error(ANIRA_ERROR_MODEL_LOAD,
                                   "coreml engine: the model has no array input '" + name + "'");
            }
            m_input_types.push_back(feature.multiArrayConstraint.dataType);
        }
        for (const std::string& name : m_bound_outputs) {
            if (outputs[@(name.c_str())] == nil) {
                throw anira::Error(ANIRA_ERROR_MODEL_LOAD,
                                   "coreml engine: the model has no output '" + name + "'");
            }
        }
    }

    ~Session() { ModelPool::instance().give_back(m_compiled, m_units, m_model); }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;

    anira_status run(anira::EngineContext& ctx) noexcept {
        @autoreleasepool {
            @try {
                NSMutableDictionary<NSString*, MLFeatureValue*>* features =
                    [NSMutableDictionary dictionaryWithCapacity:ctx.inputs().size()];
                for (std::size_t i = 0; i < ctx.inputs().size(); ++i) {
                    MLMultiArray* array = input_array(ctx.inputs()[i], m_input_types[i]);
                    if (array == nil) { return ANIRA_ERROR_ENGINE; }
                    features[@(m_bound_inputs[i].c_str())] =
                        [MLFeatureValue featureValueWithMultiArray:array];
                }
                NSError* error = nil;
                MLDictionaryFeatureProvider* provider =
                    [[MLDictionaryFeatureProvider alloc] initWithDictionary:features error:&error];
                if (provider == nil) { return ANIRA_ERROR_ENGINE; }
                id<MLFeatureProvider> result = [m_model predictionFromFeatures:provider
                                                                         error:&error];
                if (result == nil) { return ANIRA_ERROR_ENGINE; }
                for (std::size_t i = 0; i < ctx.outputs().size(); ++i) {
                    MLMultiArray* array =
                        [result featureValueForName:@(m_bound_outputs[i].c_str())].multiArrayValue;
                    if (array == nil || !copy_out(array, ctx.outputs()[i])) {
                        return ANIRA_ERROR_ENGINE;
                    }
                }
                return ANIRA_OK;
            } @catch (NSException*) { return ANIRA_ERROR_ENGINE; }
        }
    }

private:
    // The model's feature each slot binds to: the one of its name, else, where the model has a
    // single feature on that side and the description a single slot, that one (anira's rule for
    // its own adapters: by name, else by position, which a dictionary of features only fixes for
    // one).
    static std::vector<std::string> bind(const std::vector<std::string>& names,
                                         NSDictionary<NSString*, MLFeatureDescription*>* features) {
        std::vector<std::string> bound;
        for (const std::string& name : names) {
            if (features[@(name.c_str())] != nil) {
                bound.push_back(name);
            } else if (names.size() == 1 && features.count == 1) {
                bound.emplace_back(features.allKeys.firstObject.UTF8String);
            } else {
                throw anira::Error(ANIRA_ERROR_MODEL_LOAD,
                                   "coreml engine: the model has no feature '" + name + "'");
            }
        }
        return bound;
    }

    // anira's packed host tensor as the array Core ML takes: over its memory where the types
    // match, else a converted copy.
    static MLMultiArray* input_array(const anira::Tensor& tensor, MLMultiArrayDataType type) {
        NSArray<NSNumber*>* shape = shape_of(tensor);
        void* data = anira_tensor_data(&tensor, tensor.dtype);
        const std::size_t count = anira_tensor_num_elements(&tensor);
        NSError* error = nil;
        if (dtype_of(type) == tensor.dtype) {
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:shape.count];
            NSInteger stride = 1;
            for (NSInteger d = static_cast<NSInteger>(shape.count) - 1; d >= 0; --d) {
                [strides insertObject:@(stride) atIndex:0];
                stride *= shape[static_cast<NSUInteger>(d)].integerValue;
            }
            return [[MLMultiArray alloc] initWithDataPointer:data
                                                       shape:shape
                                                    dataType:type
                                                     strides:strides
                                                 deallocator:nil
                                                       error:&error];
        }
        MLMultiArray* array = [[MLMultiArray alloc] initWithShape:shape dataType:type error:&error];
        if (array == nil) { return nil; }
        const anira_dtype to = dtype_of(type);
        // A fresh array is packed row-major.
        void* out = array.dataPointer;
        for (std::size_t i = 0; i < count; ++i) { store(out, i, to, load(data, i, tensor.dtype)); }
        return array;
    }

    // Core ML's output (any strides, its own data type) into anira's packed host tensor.
    static bool copy_out(MLMultiArray* array, const anira::Tensor& tensor) {
        const std::size_t count = anira_tensor_num_elements(&tensor);
        if (static_cast<std::size_t>(array.count) != count) { return false; }
        const anira_dtype from = dtype_of(array.dataType);
        if (from == 0) { return false; }
        void* out = anira_tensor_data(&tensor, tensor.dtype);
        const NSUInteger rank = array.shape.count;
        std::vector<NSInteger> extents(rank);
        std::vector<NSInteger> strides(rank);
        for (NSUInteger d = 0; d < rank; ++d) {
            extents[d] = array.shape[d].integerValue;
            strides[d] = array.strides[d].integerValue;
        }
        const auto copy = [&](const void* bytes) {
            std::vector<NSInteger> index(rank, 0);
            for (std::size_t i = 0; i < count; ++i) {
                NSInteger offset = 0;
                for (NSUInteger d = 0; d < rank; ++d) { offset += index[d] * strides[d]; }
                store(out, i, tensor.dtype, load(bytes, static_cast<std::size_t>(offset), from));
                for (auto d = static_cast<NSInteger>(rank) - 1; d >= 0; --d) {
                    const auto at = static_cast<std::size_t>(d);
                    if (++index[at] < extents[at]) { break; }
                    index[at] = 0;
                }
            }
        };
        if (@available(macOS 12.3, iOS 15.4, *)) {
            [array getBytesWithHandler:^(const void* bytes, NSInteger /*size*/) { copy(bytes); }];
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            copy(array.dataPointer);
#pragma clang diagnostic pop
        }
        return true;
    }

    NSURL* m_compiled = nil;
    MLComputeUnits m_units = MLComputeUnitsAll;
    MLModel* m_model = nil;
    std::vector<std::string> m_bound_inputs;
    std::vector<std::string> m_bound_outputs;
    std::vector<MLMultiArrayDataType> m_input_types;
};

class SharedModel final : public anira::Engine::Loaded {
public:
    SharedModel(std::unique_ptr<Source> source, std::uint32_t instances)
        : m_source(std::move(source)) {
        for (std::uint32_t i = 0; i < instances; ++i) {
            m_sessions.push_back(std::make_unique<Session>(*m_source));
        }
    }

    std::unique_ptr<anira::Engine::Prepared> prepare(const anira::PrepareInfo& info) override;

    std::unique_ptr<Source> m_source;
    std::vector<std::unique_ptr<Session>> m_sessions;  // the shared slots
};

// A stateless handler's calls run on the loaded model's slots; an exclusive one's on its own
// session.
class Handler final : public anira::Engine::Prepared {
public:
    Handler(SharedModel& loaded, std::unique_ptr<Session> own)
        : m_loaded(loaded), m_own(std::move(own)) {}

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
    return std::make_unique<Handler>(
        *this,
        info.exclusive() ? std::make_unique<Session>(*m_source) : nullptr);
}

constexpr std::array<const char*, 2> k_providers{"cpu", "coreml"};

class CoreMlEngine final : public anira::Engine {
public:
    CoreMlEngine() : anira::Engine(k_coreml_engine_id) {}

    std::span<const char* const> providers() const noexcept override { return k_providers; }

    std::unique_ptr<anira::Engine::Loaded> load(const anira::EngineLoadInfo& info) override {
        const std::string path(info.model_path(info.row()));
        if (path.empty()) {
            throw anira::Error(ANIRA_ERROR_NOT_SUPPORTED,
                               "coreml engine: a .mlpackage is a directory, it loads from a path");
        }
        auto source = std::make_unique<Source>();
        source->m_units =
            info.provider().kind == ANIRA_PROVIDER_CPU ? MLComputeUnitsCPUOnly : MLComputeUnitsAll;
        for (const char* name : info.input_names()) { source->m_input_names.emplace_back(name); }
        for (const char* name : info.output_names()) { source->m_output_names.emplace_back(name); }
        @autoreleasepool {
            source->m_compiled = compiled(path);
            return std::make_unique<SharedModel>(std::move(source), info.instances());
        }
    }
};

}  // namespace

std::shared_ptr<anira::Engine> coreml_engine() {
    static std::mutex mutex;
    static std::weak_ptr<anira::Engine> current;
    const std::scoped_lock lock(mutex);
    auto engine = current.lock();
    if (!engine) {
        engine = std::make_shared<CoreMlEngine>();
        current = engine;
    }
    return engine;
}

}  // namespace anira::models::core
