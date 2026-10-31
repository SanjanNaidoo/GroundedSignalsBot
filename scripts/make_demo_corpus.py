#!/usr/bin/env python3
"""Write a small placeholder corpus so C2-rag can be demonstrated.

This is NOT course material — six passages written to exercise retrieval before
the real EEE4114F corpus exists. corpus/processed/ is gitignored, so this script
is how the demo reproduces after a fresh clone. Delete the output once real
material is ingested.
"""

import json
import pathlib

CHUNKS = [
    ("demo-001", "sampling", 1,
     "Ideal sampling is modelled as multiplication of x(t) by an impulse train of period "
     "Ts. Because multiplication in time corresponds to convolution in frequency, the "
     "sampled spectrum is the original spectrum replicated at every integer multiple of "
     "the sampling rate fs. For a signal bandlimited to B, the replicas do not overlap "
     "provided fs > 2B, the Nyquist condition."),
    ("demo-002", "sampling", 2,
     "When fs < 2B the spectral replicas overlap and high-frequency content folds down "
     "into the baseband. A component at frequency f0 above fs/2 appears at |f0 - k*fs| "
     "for the integer k that brings it into the range 0 to fs/2. This folding is "
     "irreversible: once the replicas overlap there is no filter that can separate the "
     "original content from the aliased content."),
    ("demo-003", "dtft-dft", 7,
     "Frequency resolution is determined by the duration of the observed record, "
     "T = N*Ts, and is approximately 1/T. Zero-padding a record before the DFT increases "
     "the number of frequency samples and produces a smoother plot, but it interpolates "
     "the same underlying DTFT and adds no information. Two tones separated by less than "
     "1/T cannot be resolved by any amount of zero-padding."),
    ("demo-004", "dtft-dft", 9,
     "The DFT treats the observed record as one period of a periodic sequence. If the "
     "record does not contain a whole number of cycles, the implied periodic extension is "
     "discontinuous and energy leaks across all bins. Equivalently, the DFT samples the "
     "DTFT of the rectangularly windowed signal, whose kernel has sidelobes about 13 dB "
     "below the main lobe. Tapered windows such as Hann reduce sidelobes to roughly 31 dB "
     "at the cost of doubling the main-lobe width."),
    ("demo-005", "ztransform", 14,
     "The region of convergence of a causal sequence is the region outside the outermost "
     "pole. A discrete-time LTI system is BIBO stable if and only if its ROC contains the "
     "unit circle. The transfer function alone does not determine stability: the same "
     "H(z) with a different ROC describes a different system, and only one of them may be "
     "stable."),
    ("demo-006", "ztransform", 15,
     "For a first-order causal recursion y[n] = x[n] + a*y[n-1], taking the z-transform "
     "gives H(z) = 1/(1 - a*z^-1), a single pole at z = a and a zero at the origin. The "
     "impulse response is h[n] = a^n u[n], which is absolutely summable when |a| < 1, so "
     "the system is stable exactly when the pole lies inside the unit circle."),
]

OUT = pathlib.Path("corpus/processed/DEMO-chunks.jsonl")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for chunk_id, topic, page, text in CHUNKS:
            fh.write(json.dumps({
                "chunk_id": chunk_id,
                "topic": topic,
                "source": "DEMO-placeholder.pdf",
                "page": page,
                "text": text,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {len(CHUNKS)} placeholder chunks to {OUT}")
    print("These are NOT course material. Replace with ingested EEE4114F notes.")


if __name__ == "__main__":
    main()
