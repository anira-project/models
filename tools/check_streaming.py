# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["executorch==1.3.1", "torch==2.12.0", "coremltools==9.0", "onnxruntime", "ai-edge-litert", "numpy"]
# ///
"""Streams every description's weights over many consecutive calls, on every format it names that
runs in Python (ONNX Runtime, LiteRT / TensorFlow Lite, ExecuTorch with XNNPACK or Core ML, LibTorch)
and its native Core ML ML program (the entries of core::coreml_engine(), until anira has a Core
ML engine), with the same inputs and each format's state fed back as the description's state pairs say,
and checks:

  coverage   every model comes for every engine and backend: ONNX Runtime, LiteRT and TensorFlow
             Lite, LibTorch, ExecuTorch on XNNPACK, Core ML and Vulkan, and native Core ML (the
             -libtorch descriptions, the streaming TorchScript for LibTorch alone, excepted);
  agreement  every format matches the reference (LibTorch where the description has it, else ONNX
             Runtime) on every call, not only the first: a format that loses state between calls
             diverges from the second call on;
  history    a "stateful" description depends on the past: its outputs, state fed back, differ from
             the same calls each started from a fresh state. One without state tensors keeps its
             state inside the model: then a continuous stream must differ from each call made on a
             freshly loaded model, and every format must keep it (the agreement above); a format
             that loses it needs the state as tensors in and out instead.

    uv run tools/check_streaming.py <weights root: one folder per HF repository> [models dir]

Exits non-zero when a check fails.
"""

import glob
import json
import os
import sys
import time

import coremltools as ct
import numpy as np
import onnxruntime
import torch
from ai_edge_litert.interpreter import Interpreter
from executorch.runtime import Runtime

CALLS = 12
TOLERANCE = 1e-3  # relative to the reference output's peak
TOLERANCE_COREML = 2e-2  # Core ML programs, hand tracking's in float16
REFERENCES = ("libtorch", "onnxruntime")
COREML_ENGINE_ID = "org.anira-project.models.coreml"  # core/include/anira/models/core/CoreMlEngine.h
REQUIRED = ("onnxruntime", "litert", "tflite", "libtorch", "executorch", "executorch/coreml", "executorch/vulkan",
            "coreml")
# Formats a model cannot have yet, and why: reported, not failed.
KNOWN_GAPS = {}


def shape(tensor):
    return [n for _, n in tensor["axes"]]


class Format:
    """One weight file of a description, called positionally with the description's tensors."""

    def __init__(self, engine, path, entry, description):
        self.engine = engine
        if engine == "onnxruntime":
            self.session = onnxruntime.InferenceSession(path)
            self.names = [i.name for i in self.session.get_inputs()]
        elif engine in ("litert", "tflite"):
            # As anira's LiteRT engine binds: the first signature (for a file without one, LiteRT's
            # synthesized one, keyed by the tensor names), a description tensor to the key of its
            # name, the others to the remaining keys in sorted order.
            self.interpreter = Interpreter(model_path=path)
            signatures = self.interpreter.get_signature_list()
            if signatures:
                self.runner = self.interpreter.get_signature_runner(next(iter(signatures)))
                details, out_keys = self.runner.get_input_details(), list(self.runner.get_output_details())
            else:
                self.runner = None
                self.interpreter.allocate_tensors()
                details = {d["name"]: d for d in self.interpreter.get_input_details()}
                self.out_index = {d["name"]: d["index"] for d in self.interpreter.get_output_details()}
                out_keys = list(self.out_index)
            self.inputs = bind([t["name"] for t in description["inputs"]], list(details))
            self.shapes = {k: details[k].get("shape_signature", details[k]["shape"]) for k in details}
            self.in_index = {k: details[k]["index"] for k in details}
            self.outputs = bind([t["name"] for t in description["outputs"]], out_keys)
        elif engine == "executorch":
            self.method = Runtime.get().load_program(path).load_method("forward")
        elif engine == "libtorch":
            module = torch.jit.load(path).eval()
            self.method = getattr(module, entry.get("name", "forward")) if entry else module
        elif engine == "coreml":
            self.model = ct.models.MLModel(path)
            spec = self.model.get_spec()
            self.names = [i.name for i in spec.description.input]
            self.shapes = [list(i.type.multiArrayType.shape) for i in spec.description.input]
            self.out_names = [o.name for o in spec.description.output]

    def __call__(self, inputs):
        if self.engine == "coreml":
            # Core ML takes integers as int32.
            inputs = [v.astype(np.int32) if v.dtype == np.int64 else v for v in inputs]
            feed = {n: v.reshape(s) if int(np.prod(s)) == v.size else v for n, s, v in zip(self.names, self.shapes, inputs)}
            out = self.model.predict(feed)
            return [out[n] for n in self.out_names]
        if self.engine == "onnxruntime":
            return self.session.run(None, dict(zip(self.names, inputs)))
        if self.engine in ("litert", "tflite"):
            feed = {}
            for key, value in zip(self.inputs, inputs):
                # The same elements in the file's layout (a size-1 axis elsewhere, a dynamic axis
                # resolved from the element count), as anira passes them.
                expected = [int(d) for d in self.shapes[key]]
                if expected.count(-1) == 1:
                    known = int(np.prod([d for d in expected if d != -1]))
                    expected[expected.index(-1)] = value.size // known
                feed[key] = value.reshape(expected) if int(np.prod(expected)) == value.size else value
            if self.runner is not None:
                out = self.runner(**feed)
                return [np.array(out[key]) for key in self.outputs]
            for key, value in feed.items():
                if tuple(value.shape) != tuple(self.interpreter.get_tensor(self.in_index[key]).shape):
                    self.interpreter.resize_tensor_input(self.in_index[key], value.shape)
                    self.interpreter.allocate_tensors()
                self.interpreter.set_tensor(self.in_index[key], value)
            self.interpreter.invoke()
            return [self.interpreter.get_tensor(self.out_index[key]).copy() for key in self.outputs]
        with torch.no_grad():
            values = [torch.from_numpy(v) for v in inputs]
            out = self.method.execute(values) if self.engine == "executorch" else self.method(*values)
        out = out if isinstance(out, (list, tuple)) else [out]
        return [np.asarray(o.detach() if isinstance(o, torch.Tensor) else o) for o in out]


def input_values(rng, tensor, length):
    """A call's values for an input: noise for a float tensor; for an integer or boolean one, a
    length (a name with "len") is `length`, the whole window, else ones (every position valid, as a
    mask wants); in the description's dtype."""
    dtype = np.dtype(tensor.get("dtype", "float32"))
    if np.issubdtype(dtype, np.floating):
        return (rng.standard_normal(shape(tensor)) * 0.1).astype(dtype)
    return np.full(shape(tensor), length if "len" in tensor["name"] else 1, dtype)


def bind(names, keys):
    """The signature key of each name: its own where one matches, else the next unused sorted key."""
    rest = iter(sorted(k for k in keys if k not in names))
    return [name if name in keys else next(rest) for name in names]


def stream(run, description, calls, fresh_each_call=False):
    """The outputs of consecutive calls, each output list without the state outputs."""
    inputs, outputs = description["inputs"], description["outputs"]
    out_index = {t["name"]: i for i, t in enumerate(outputs)}
    dtypes = [np.dtype(t.get("dtype", "float32")) for t in inputs]
    states = {i: np.zeros(shape(t), dtypes[i]) for i, t in enumerate(inputs) if t.get("state_source")}
    results = []
    for values in calls:
        feed = [states[i] if i in states else values[i] for i in range(len(inputs))]
        if fresh_each_call:
            feed = [np.zeros_like(states[i]) if i in states else values[i] for i in range(len(inputs))]
        out = run(feed)
        for i in states:
            states[i] = np.asarray(out[out_index[inputs[i]["state_source"]]], dtypes[i])
        state_outputs = {out_index[inputs[i]["state_source"]] for i in states}
        results.append([np.asarray(o, np.float32) for k, o in enumerate(out) if k not in state_outputs])
    return results


def timed(run):
    """run, with the time of its calls in run.times (reported: the median after the first call, which
    may compile or warm up)."""
    def call(values):
        start = time.perf_counter()
        out = run(values)
        call.times.append(time.perf_counter() - start)
        return out
    call.times = []
    return call


def deviation(reference, other):
    worst = []
    for ref, got in zip(reference, other):
        # nan anywhere (in the reference or the format) is a failure, not a pass of every comparison.
        worst.append(max(float(np.max(np.abs(r - g.reshape(r.shape)))) / (1.0 + float(np.max(np.abs(r))))
                         if np.all(np.isfinite(r)) and np.all(np.isfinite(g)) else float("inf")
                         for r, g in zip(ref, got)))
    return worst


def check(path, weights_root):
    description = json.load(open(path))
    folder = os.path.dirname(path)
    name = os.path.basename(path)
    repos = {}
    for catalog_path in glob.glob(f"{folder}/*.catalog.json"):
        catalog = json.load(open(catalog_path))
        for f in catalog["files"]:
            repos[f["path"]] = catalog["repo"].split("/")[1]
            if ".mlpackage/" in f["path"]:  # a Core ML entry names the package, a directory
                repos[f["path"][:f["path"].index(".mlpackage") + len(".mlpackage")]] = repos[f["path"]]
    # core::coreml_engine()'s entries, until anira has a Core ML engine: "coreml" here.
    entries = [dict(e, engine="coreml", provider=None) if e["engine"] == COREML_ENGINE_ID else e
               for e in description["models"]]
    present = {e["engine"] + ("/" + e["provider"] if e.get("provider") else "") for e in entries}
    gaps = KNOWN_GAPS.get(name, {})
    missing = [r for r in REQUIRED if r not in present and r not in gaps] if not name.endswith("-libtorch.model.json") else []
    formats, loaders = {}, {}
    for entry in entries:
        engine = entry["engine"]
        if engine == "tflite" or engine not in ("onnxruntime", "litert", "executorch", "libtorch", "coreml") \
                or entry.get("provider") == "vulkan":
            continue  # tflite runs the litert file; Python's ExecuTorch has no Vulkan delegate
        weights = os.path.join(weights_root, repos.get(entry["path"], ""), entry["path"])
        if not os.path.exists(weights):
            continue
        label = engine + ("/" + entry["provider"] if entry.get("provider") else "")
        loaders[label] = lambda engine=engine, weights=weights, e=entry.get("entry"): Format(engine, weights, e, description)
        formats[label] = loaders[label]()
    if not formats:
        return True, f"{os.path.basename(path)}: no weights here"

    rng = np.random.default_rng(0)
    # A length input's value: the last extent of the first float input, the window it measures.
    length = next((shape(t)[-1] for t in description["inputs"]
                   if np.issubdtype(np.dtype(t.get("dtype", "float32")), np.floating)), 1)
    calls = [[input_values(rng, t, length) for t in description["inputs"]] for _ in range(CALLS)]
    reference_label = next((r for r in REFERENCES if r in formats), next(iter(formats)))
    reference_run = timed(formats[reference_label])
    reference = stream(reference_run, description, calls)
    lines, ok = [], not missing
    if missing:
        lines.append(f"    coverage: FAIL, missing {', '.join(missing)}")
    for gap, reason in gaps.items():
        if gap not in present:
            lines.append(f"    coverage: no {gap} ({reason})")
    lines.append(f"    {reference_label:22} reference, {1000 * np.median(reference_run.times[1:]):.2f} ms per call")
    for label, run in formats.items():
        if label == reference_label:
            continue
        run = timed(run)
        try:
            worst = deviation(reference, stream(run, description, calls))
        except Exception as error:  # pylint: disable=broad-except
            ok = False
            lines.append(f"    {label:22} FAIL: {str(error).splitlines()[0][:120]}")
            continue
        first, later = worst[0], max(worst[1:])
        tolerance = TOLERANCE_COREML if "coreml" in label else TOLERANCE
        bad = max(worst) > tolerance
        ok &= not bad
        lost = bad and first <= tolerance and description.get("state") == "stateful"
        lines.append(f"    {label:22} vs {reference_label}: call 1 {first:.1e}, calls 2..{CALLS} {later:.1e}, "
                     f"{1000 * np.median(run.times[1:]):.2f} ms per call"
                     + ("  FAIL" if bad else "") + ("  (diverges after the first call: state lost)" if lost else ""))
    has_pairs = any(t.get("state_source") for t in description["inputs"])
    if description.get("state") == "stateful" and not has_pairs:
        # State inside the model: a stream on one load against each call on a fresh load.
        carried = stream(loaders[reference_label](), description, calls[:4])
        fresh = [stream(loaders[reference_label](), description, [c])[0] for c in calls[:4]]
        history = max(deviation(carried, fresh)[1:])
        if history <= 1e-6:
            ok = False
            lines.append(f"    history: FAIL, stateful but calls 2..4 do not depend on the past ({history:.1e})")
        else:
            lines.append(f"    history: the state is inside the model ({history:.1e}); one stream per loaded model")
    elif description.get("state") == "stateful":
        # Fresh loads: a model's own state must not carry from one stream into the next.
        carried = stream(loaders[reference_label](), description, calls)
        fresh = stream(loaders[reference_label](), description, calls, fresh_each_call=True)
        history = max(deviation(carried, fresh)[1:])
        if history <= 1e-6:
            ok = False
            lines.append(f"    history: calls 2..{CALLS} do not depend on the past ({history:.1e}): FAIL, "
                         "the state is lost between calls; it needs state tensors in and out")
        else:
            lines.append(f"    history: depends on the past ({history:.1e})")
    status = "ok  " if ok else "FAIL"
    return ok, f"{status} {os.path.relpath(path, os.path.dirname(folder))} ({', '.join(formats)})\n" + "\n".join(lines)


def main():
    weights_root = sys.argv[1]
    models = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "models")
    all_ok = True
    for path in sorted(glob.glob(f"{models}/**/*.model.json", recursive=True)):
        ok, report = check(path, weights_root)
        all_ok &= ok
        print(report, flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
