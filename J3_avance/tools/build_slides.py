"""Compile every slides/T*.tex with pdflatex (twice) and report errors and overfull boxes.

Usage:  python build_slides.py [T1 T2 ...]     (no argument = all decks)
"""
import re
import subprocess
import sys
from pathlib import Path

SLIDES = Path(__file__).resolve().parent.parent / "slides"


def build(tex):
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name], cwd=SLIDES,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = (SLIDES / tex.with_suffix(".log").name).read_text(encoding="latin-1")
    errors = [l for l in log.splitlines() if l.startswith("!")]
    vboxes = re.findall(r"Overfull \\vbox \(([\d.]+)pt too high\) detected at line (\d+)", log)
    out = re.search(r"Output written on .*\((\d+) pages", log)
    print(f"{tex.stem}: {out.group(1) if out else 'NO'} pages, {len(errors)} errors")
    for e in errors[:5]:
        print("   ", e)
    for h, line in vboxes:
        if float(h) > 3 and line != "9":  # line 9 = title page, harmless
            print(f"    overfull vbox {h}pt at line {line}")


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for tex in sorted(SLIDES.glob("T*.tex")):
        if not wanted or any(tex.stem.startswith(w) for w in wanted):
            build(tex)
    for ext in ["aux", "log", "nav", "out", "snm", "toc", "vrb"]:
        for f in SLIDES.glob(f"*.{ext}"):
            f.unlink()
