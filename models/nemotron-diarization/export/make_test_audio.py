# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "soundfile"]
# ///
# SPDX-License-Identifier: Apache-2.0
"""The diarization test recording: three LibriSpeech dev-clean utterances (CC BY 4.0, read LibriVox
audiobooks), a woman, a man, the woman again, 0.5 s of silence between them, as 16 kHz 16-bit mono.

    uv run make_test_audio.py <LibriSpeech/dev-clean> <two_speakers_16k.wav>

Prints each utterance's speaker and first and end sample (test_Diarizer.cpp's k_voices).
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

UTTERANCES = [
    (0, "84-121123-0003"),     # speaker 84, Christie Nowak (F)
    (1, "1272-128104-0003"),   # speaker 1272, John Rose (M)
    (0, "84-121123-0008"),
]
GAP = 8000  # 0.5 s at 16 kHz


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    pieces = []
    position = 0
    for i, (speaker, utterance) in enumerate(UTTERANCES):
        reader, chapter, _ = utterance.split("-")
        audio, rate = sf.read(root / reader / chapter / f"{utterance}.flac", dtype="int16")
        assert rate == 16000 and audio.ndim == 1
        if i > 0:
            pieces.append(np.zeros(GAP, np.int16))
            position += GAP
        print(f"speaker {speaker}: {position} .. {position + len(audio)}  ({utterance})")
        pieces.append(audio)
        position += len(audio)
    sf.write(out, np.concatenate(pieces), 16000, subtype="PCM_16")


if __name__ == "__main__":
    main()
