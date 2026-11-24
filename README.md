# anira models

Ready-to-use model libraries on [anira](https://github.com/anira-project/anira): add hand tracking
or speech to text to an application with one CMake option and one target. The weights come from
the [anira-project](https://huggingface.co/anira-project) organisation on Hugging Face.

**Status: scaffold.** The libraries (core, hand tracking, Silero VAD, Parakeet, Nemotron diarization, Nemotron ASR, Moonshine) follow. The dummy networks for tests are in `test-models/`.

The effects (GuitarLSTM, the stateful LSTM, steerable-nafx and the three RAVE models) are one header
each on `core::AudioEffect`: prepare, process in place, reset, get_latency, any channel count and
host rate (resampled to the 44.1 kHz they were trained at), for example
`anira::models::guitar_lstm::GuitarLstm` from `<anira/models/guitar_lstm/GuitarLstm.h>` with
`ANIRA_MODELS_WITH_GUITAR_LSTM`. A RAVE header also has the model split: `<Name>Encoder`
(`core::LatentEncoder`: audio in, latent frames out) and `<Name>Decoder` (`core::LatentDecoder`:
latents in from a control thread, audio out).

## A model

One folder per model under `models/`, JSON first:

```
models/<name>/
  <name>.model.json                anira's description: engines, tensors, state
  <name>.contract.json             the contract
  <hf-repo>.catalog.json           the weights on Hugging Face: revision, SHA256 per file
  <Name>Stage.h                    pre/post-processing, only if needed
  <Name>.h                         a tiny wrapper shaped like an anira handler
```

Tensor in, tensor out: every model runs as an anira handler over its JSON; the wrapper turns output
tensors into results (text, hands) with plain functions. The descriptions live here, versioned with
the code, and are compiled into the library; Hugging Face holds only the weights and their licence
texts.

## Use

```cmake
set(ANIRA_MODELS_WITH_HANDTRACKING ON)
FetchContent_Declare(anira-models
    GIT_REPOSITORY https://github.com/anira-project/models.git
    GIT_TAG v0.1.0)
FetchContent_MakeAvailable(anira-models)
target_link_libraries(app PRIVATE anira::models::handtracking)
```

### anira

A process holds one copy of anira. If your project already adds anira (`add_subdirectory`,
`FetchContent`, `find_package`), anira-models uses it; otherwise it fetches the anira commit
it is tested with (`ANIRA_MODELS_ANIRA_GIT_TAG`), or finds an installed anira 3 first. When
your anira differs from the tested one, configure prints a warning; anira 2.x is an error. The
inference engines are yours to choose (`ANIRA_WITH_ONNXRUNTIME`, `ANIRA_WITH_LITERT`, …).

Every model also comes as a Core ML ML program (`coreml/*.mlpackage`, engine `coreml` in the
catalogs) for the native Core ML engine anira 3 adds, and as an ExecuTorch program lowered to Vulkan
(`*_vulkan.pte`, `"provider": "vulkan"`, not yet run on a GPU); the descriptions name these once
anira and a test device do, and the catalogs tie each to its description (`"model"`). ExecuTorch
programs lowered to Core ML (`*_coreml.pte`, `"provider": "coreml"`) are fetched on macOS and iOS
only, Vulkan programs only when `PROVIDERS` names `vulkan`.

## Build

```bash
git clone https://github.com/anira-project/models.git && cd anira-models
just setup
just test
```

Requirements: CMake ≥ 3.25, Ninja, clang, `just`. Dependencies, all fetched by CMake: anira,
tanh-lib (Core and DspBasics, both Apache-2.0), nlohmann_json (the release anira uses; anira's
copy when it brings one) and, for the tests, googletest.

### Offline

Every download can come from a local copy. In `CMakeUserPresets.json` (not committed):

```json
{
  "version": 6,
  "configurePresets": [{
    "name": "offline", "inherits": "debug",
    "cacheVariables": {
      "FETCHCONTENT_SOURCE_DIR_ANIRA": "/path/to/anira",
      "FETCHCONTENT_SOURCE_DIR_TANH-LIB": "/path/to/tanh-lib",
      "ANIRA_ONNXRUNTIME_ROOTDIR": "/path/to/onnxruntime-1.26.0-macOS-arm64-shared",
      "ANIRA_LITERT_ROOTDIR": "/path/to/LiteRt-2.1.5-macOS-arm64-shared",
      "ANIRA_BACKENDS_SKIP_REMOTE_CHECK": "ON"
    }
  }]
}
```

The anira and tanh-lib checkouts must be at the commits anira pins (anira: this repo's
`ANIRA_MODELS_ANIRA_GIT_TAG`; tanh-lib: the `GIT_TAG` in `cmake/dependencies.cmake`).

### The weights

Two checks run on the weights themselves, from a folder with one checkout per Hugging Face
repository (`<weights>/<repo name>/onnx/...`):

```bash
ANIRA_MODELS_TEST_WEIGHTS=<weights> just test   # every description prepares in anira on every engine
just check-streaming <weights>                       # every format, many consecutive calls
```

`tools/check_streaming.py` runs every format of every description (ONNX Runtime, LiteRT, LibTorch,
ExecuTorch on XNNPACK and Core ML, native Core ML) over twelve consecutive calls with the state fed
back, and fails when a format disagrees on any call, when a stateful model does not depend on its past
(its state is lost between calls: it needs state tensors in and out), or when a model lacks an engine
or backend. It prints each format's time per call. A model's export scripts live with its weights
(`export/` in each repository) and check each export the same way.

## Licence

[Apache-2.0](LICENSE), like anira and tanh-lib's Core and DspBasics. One licence per library
target: configure fails if an Apache-2.0 target links AGPL code (`cmake/licenses.cmake`). Model
weights keep the licence of their upstream project, listed in each model's `catalog.json` and on its
Hugging Face card.

For commercial support, fine-tuning a model on your own data, or a custom model, contact tanh-lab,
the company behind anira: **contact@tanh-lab.com**.
