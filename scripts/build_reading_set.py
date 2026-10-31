"""The images the reading test set is measured on, rebuilt from the course's own papers.

The first version of this set was cut by hand at 140 dpi, which produced crops of about
250 thousand pixels. The screenshot that exposed the reader's worst failure - `u[n+5]`
read as `u[n-5]`, which inverts the question - was 699x116, about 81 thousand. So the
number that set produced, 92% of mathematical expressions read intact, belongs to images
roughly three times larger than the ones students actually send.

This rebuilds the same crops at any resolution, so the set can be measured at the size a
student screenshots rather than the size that happened to be convenient. Two are built:

    screen  ~90 dpi   what a PDF viewer gives at ordinary zoom on a laptop
    print   140 dpi   the original set, kept so the first measurement stays comparable

The crops are the same regions of the same pages either way, so the only variable
between the two is resolution - which makes the pair a measurement of how much
resolution matters, rather than only a correction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus" / "raw"
BANK = ROOT / "stress" / "bank" / "reading"

# pdftoppm ships with MiKTeX here; a Linux box has it on PATH.
CANDIDATE_RENDERERS = (
    Path(r"C:\Users\User\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdftoppm.exe"),
    Path("/usr/bin/pdftoppm"),
    Path("/usr/local/bin/pdftoppm"),
)

RESOLUTIONS = {"screen": 90, "print": 140}

# (id, source pdf, page, x0, y0, x1, y1) with the box as fractions of the page, so the
# same region is cut whatever the resolution.
CROPS = [
    ("ex26-mcq1",   "EEE4114F_Final_Exam_2026.pdf",          2,  0.08, 0.150, 0.98, 0.320),
    ("ex26-mcq2",   "EEE4114F_Final_Exam_2026.pdf",          2,  0.08, 0.320, 0.98, 0.455),
    ("ex26-mcq4",   "EEE4114F_Final_Exam_2026.pdf",          2,  0.08, 0.565, 0.98, 0.675),
    ("ex26-mcq5",   "EEE4114F_Final_Exam_2026.pdf",          2,  0.08, 0.675, 0.98, 0.905),
    ("ex25-q1stem", "EEE4114F_Final_Exam_2025.pdf",          2,  0.07, 0.130, 0.95, 0.265),
    ("ex25-q5fig",  "EEE4114F_Final_Exam_2025.pdf",          8,  0.07, 0.070, 0.95, 0.440),
    ("ex26-ml1",    "EEE4114F_Final_Exam_2026.pdf",         12,  0.08, 0.140, 0.98, 0.265),
    ("ex26-ml5",    "EEE4114F_Final_Exam_2026.pdf",         12,  0.08, 0.610, 0.98, 0.760),
    ("info-trans",  "EEE4114F 2025 Class Test 1 of 2.pdf",   6,  0.10, 0.145, 0.96, 0.330),
    ("info-props",  "EEE4114F 2025 Class Test 1 of 2.pdf",   6,  0.24, 0.370, 0.80, 0.530),
    ("info-pairs",  "EEE4114F 2025 Class Test 1 of 2.pdf",   6,  0.24, 0.560, 0.80, 0.800),
    ("ct25-q1stem", "EEE4114F_Class_Test_1_2025.pdf",        2,  0.08, 0.080, 0.95, 0.240),
    ("ct25-q1ans",  "EEE4114F_Class_Test_1_2025.pdf",        2,  0.12, 0.580, 0.95, 0.650),
    ("ex25-ml5",    "EEE4114F_Final_Exam_2025.pdf",         12,  0.10, 0.450, 0.95, 0.520),
]


def renderer() -> Path:
    for path in CANDIDATE_RENDERERS:
        if path.exists():
            return path
    raise SystemExit("pdftoppm not found; it ships with MiKTeX or poppler-utils")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", default="screen", choices=[*RESOLUTIONS, "all"])
    args = parser.parse_args()

    from PIL import Image

    tool = renderer()
    wanted = RESOLUTIONS if args.which == "all" else {args.which: RESOLUTIONS[args.which]}
    scratch = BANK / "_pages"
    scratch.mkdir(parents=True, exist_ok=True)

    for name, dpi in wanted.items():
        out = BANK / ("images" if name == "print" else f"images-{name}")
        out.mkdir(parents=True, exist_ok=True)
        sizes = []
        for crop_id, pdf, page, x0, y0, x1, y1 in CROPS:
            stem = scratch / f"{dpi}-{pdf.replace(' ', '_')[:-4]}-p{page}"
            rendered = sorted(scratch.glob(f"{stem.name}*.png"))
            if not rendered:
                subprocess.run([str(tool), "-png", "-r", str(dpi), "-f", str(page),
                                "-l", str(page), str(RAW / pdf), str(stem)],
                               capture_output=True, check=False)
                rendered = sorted(scratch.glob(f"{stem.name}*.png"))
            if not rendered:
                print(f"  could not render {pdf} page {page}")
                continue
            page_image = Image.open(rendered[0])
            w, h = page_image.size
            crop = page_image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            crop.convert("RGB").save(out / f"{crop_id}.png")
            sizes.append(crop.size[0] * crop.size[1])
        median = sorted(sizes)[len(sizes) // 2] if sizes else 0
        print(f"{name:7} {dpi:3} dpi -> {len(sizes):2} crops in "
              f"{out.relative_to(ROOT)}, median {median / 1000:.0f}k pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
