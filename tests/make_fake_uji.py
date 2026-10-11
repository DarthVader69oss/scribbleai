"""
tests/make_fake_uji.py

Writes a STAND-IN file in exact UJIpenchars2 format, built from font glyph outlines
(matplotlib TextPath) with per-"writer" distortions. It exists only so the whole
pipeline (build_dataset -> train -> evaluate -> demo) can be smoke-tested before or
without the real download. NEVER report numbers from it.

    python tests/make_fake_uji.py data/raw/fake_uji/ujipenchars2.txt [--writers 60]
"""
import argparse
import random
from pathlib import Path

import numpy as np

CHARS = (list("abcdefghijklmnopqrstuvwxyz") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
         list("0123456789") + list("áéíóúàèòüïçñÁÉÍÓÚÀÈÒÜÏÇÑ") +
         list("!?¿¡.,;:'\"()-+*/%&@#€=<>_[]"))


def glyph_strokes(ch, font):
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), ch, size=40, prop=FontProperties(family=font))
    polys = tp.to_polygons(closed_only=False)
    return [np.asarray(p, float) for p in polys if len(p) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--writers", type=int, default=60)
    ap.add_argument("--chars", type=int, default=len(CHARS))
    a = ap.parse_args()
    fonts = ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]
    chars = CHARS[:a.chars]
    cache = {(c, f): glyph_strokes(c, f) for c in chars for f in fonts}
    lines = ["// FAKE UJIpenchars2-format file generated from font outlines (smoke tests only)"]
    n_test_writers = max(1, a.writers // 3)
    for w in range(1, a.writers + 1):
        rng = random.Random(w)
        split = "tst" if w > a.writers - n_test_writers else "trn"
        font = fonts[w % len(fonts)]
        slant, sx, sy = rng.uniform(-0.3, 0.3), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)
        for c in chars:
            for rep in (1, 2):
                strokes = cache[(c, font)] or [np.array([[0.0, 0.0], [1.0, 1.0]])]
                out = []
                for s in strokes:
                    q = s.copy()
                    q[:, 0] = (q[:, 0] + q[:, 1] * slant) * sx
                    q[:, 1] = -q[:, 1] * sy                       # y down like a tablet
                    q += np.random.default_rng(w * 1000 + rep).normal(0, 0.3, q.shape)
                    q = q * 40 + 5000
                    q = np.round(q).astype(int)
                    q = np.repeat(q, 2, axis=0)[: max(1, len(q) * 2 - 1)]   # duplicate points like the real data
                    out.append(q)
                lines.append(f"WORD {c} {split}_UJI_W{w:02d}-{rep:02d}")
                lines.append(f"  NUMSTROKES {len(out)}")
                for q in out:
                    lines.append(f"  POINTS {len(q)} # " + " ".join(f"{x} {y}" for x, y in q))
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    print(f"wrote {p} ({a.writers} writers x {len(chars)} chars x 2)")


if __name__ == "__main__":
    main()
