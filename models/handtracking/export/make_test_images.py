# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["mediapipe==0.10.*", "pillow"]
# ///
"""The test images of test/test_HandTracker.cpp: MediaPipe's own (storage.googleapis.com/mediapipe-assets,
Apache-2.0) as binary PPM, for anira-project/mediapipe-gesture-recognizer test/, and MediaPipe's
GestureRecognizer's result on each (gesture, handedness, wrist): the test's reference values.

    uv run make_test_images.py <out dir>
"""

import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

IMAGES = ["thumb_up", "victory", "fist", "pointing_up", "right_hands"]
ASSETS = "https://storage.googleapis.com/mediapipe-assets"
TASK = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/"
        "float16/1/gesture_recognizer.task")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    task = os.path.join(out, "gesture_recognizer.task")
    urllib.request.urlretrieve(TASK, task)
    recognizer = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=task), num_hands=2))
    for name in IMAGES:
        jpg = os.path.join(out, name + ".jpg")
        urllib.request.urlretrieve(f"{ASSETS}/{name}.jpg", jpg)
        Image.open(jpg).convert("RGB").save(os.path.join(out, name + ".ppm"))
        result = recognizer.recognize(mp.Image.create_from_file(jpg))
        print(name, len(result.hand_landmarks),
              [(g[0].category_name, round(g[0].score, 3)) for g in result.gestures],
              [(h[0].category_name, round(h[0].score, 3)) for h in result.handedness],
              [(round(l[0].x, 3), round(l[0].y, 3)) for l in result.hand_landmarks])
        os.remove(jpg)
    os.remove(task)


if __name__ == "__main__":
    main()
