# SPDX-License-Identifier: Apache-2.0
"""Writes a model folder's catalog from a checkout of its Hugging Face repository: every file under
<checkout>/<prefix> (and the repository's LICENSE) with its size and SHA256, its engine by its format
folder, and the provider an ExecuTorch program was lowered to by its suffix. A file that only a
provider runs (Core ML ML programs, Vulkan programs) names the description of the model it is an
export of ("model"), found by the ONNX file of the same stem in the folder's descriptions.

    python3 tools/make_catalog.py <checkout> <org/repo> <revision> <license> <model folder> [<prefix>]

The catalog is written to <model folder>/<repo>.catalog.json.
"""

import hashlib
import json
import os
import sys

ENGINES = {"onnx": "onnxruntime", "libtorch": "libtorch", "executorch": "executorch", "coreml": "coreml"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stems_to_descriptions(folder):
    """The ONNX file stem of each description in the folder -> the description's file name."""
    stems = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".model.json"):
            continue
        for entry in json.load(open(os.path.join(folder, name)))["models"]:
            if entry["engine"] == "onnxruntime":
                stems[os.path.splitext(os.path.basename(entry["path"]))[0]] = name
    return stems


def entry(checkout, rel, stems):
    parts = rel.split("/")
    fmt = next((p for p in parts if p in ENGINES or p == "litert"), None)
    item = {"path": rel}
    name = os.path.basename(rel)
    if fmt == "litert":
        item["engines"] = ["litert", "tflite"]
    elif fmt:
        item["engine"] = ENGINES[fmt]
    stem = None
    if fmt == "executorch":
        for provider in ("coreml", "vulkan"):
            if name.endswith(f"_{provider}.pte"):
                item["provider"] = provider
                stem = name[: -len(f"_{provider}.pte")]
        if item.get("provider") == "coreml":
            stem = None  # an ExecuTorch Core ML program is listed in the description itself
    elif fmt == "coreml":
        stem = next(p[: -len(".mlpackage")] for p in parts if p.endswith(".mlpackage"))
    if stem is not None and stem in stems:
        item["model"] = stems[stem]
    path = os.path.join(checkout, rel)
    item["size"] = os.path.getsize(path)
    item["sha256"] = sha256(path)
    return item


def main():
    checkout, repo, revision, license_id, folder = sys.argv[1:6]
    prefix = sys.argv[6] if len(sys.argv) > 6 else ""
    stems = stems_to_descriptions(folder)
    files = []
    if os.path.exists(os.path.join(checkout, "LICENSE")):
        files.append("LICENSE")
    for root, dirs, names in os.walk(os.path.join(checkout, prefix)):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            rel = os.path.relpath(os.path.join(root, name), checkout)
            if name.startswith(".") or rel == "LICENSE" or rel.endswith(".md") or "/export/" in f"/{rel}":
                continue
            files.append(rel)
    items = [entry(checkout, rel, stems) for rel in sorted(set(files))]
    lines = ",\n".join("    " + json.dumps(i) for i in items)
    text = (f'{{\n  "repo": "{repo}",\n  "revision": "{revision}",\n  "license": "{license_id}",\n'
            f'  "files": [\n{lines}\n  ]\n}}\n')
    out = os.path.join(folder, repo.split("/")[1] + ".catalog.json")
    with open(out, "w") as f:
        f.write(text)
    print(f"{out}: {len(items)} files")


if __name__ == "__main__":
    main()
