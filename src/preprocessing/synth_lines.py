"""
src/preprocessing/synth_lines.py  (guide 4.2)

Builds synthetic handwritten LINES out of isolated UJIpenchars2 characters, so the
CTC model has real alignment work to do.

Rules (from the guide):
  * text comes from real English (a corpus file if present, else frequent English words)
  * every character of a line comes from ONE writer (consistent style)
  * characters are placed left-to-right with random gaps, wider gaps for spaces,
    pen-lift between characters
  * characters keep their relative sizes via a typographic layout table
    (x-height letters, ascenders, descenders, capitals, punctuation)
  * light augmentation: slant, rotation, scale, per-char size jitter, baseline wobble, point jitter
  * lines are built ONLY from writers of the requested split (no leakage)

Output strokes are raw (un-normalised); they then go through the SAME
pipeline.preprocess() as real handwriting.
"""
import gzip
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from .clean import to_ink, remove_duplicates

# ---------------------------------------------------------------- layout table
# (top, bottom, max_width) in x-height units, y DOWN, baseline at y = 0, x-height top at y = -1
X = (-1.0, 0.0, 1.3)
ASC = (-1.7, 0.0, 1.1)
DESC = (-1.0, 0.7, 1.1)
CAP = (-1.6, 0.0, 1.4)
_LOWER_SPECIAL = {"t": (-1.4, 0.0, 0.9), "i": (-1.45, 0.0, 0.5), "j": (-1.45, 0.7, 0.7),
                  "f": (-1.7, 0.4, 0.9)}
_ASC = set("bdhkl")
_DESC = set("gpqy")
_PUNCT = {
    ".": (-0.15, 0.0, 0.15), ",": (-0.15, 0.3, 0.2), ":": (-0.95, 0.0, 0.15), ";": (-0.95, 0.3, 0.2),
    "!": (-1.6, 0.0, 0.3), "?": (-1.6, 0.0, 0.8), "¡": (-1.0, 0.6, 0.3), "¿": (-1.0, 0.6, 0.8),
    "'": (-1.65, -1.1, 0.15), '"': (-1.65, -1.1, 0.4), "`": (-1.65, -1.2, 0.2),
    "-": (-0.6, -0.4, 0.6), "_": (0.1, 0.25, 1.0), "=": (-0.75, -0.25, 0.8), "+": (-1.0, -0.1, 0.9),
    "*": (-1.6, -0.8, 0.6), "/": (-1.7, 0.3, 0.8), "\\": (-1.7, 0.3, 0.8), "|": (-1.7, 0.3, 0.2),
    "(": (-1.7, 0.4, 0.5), ")": (-1.7, 0.4, 0.5), "[": (-1.7, 0.4, 0.5), "]": (-1.7, 0.4, 0.5),
    "{": (-1.7, 0.4, 0.6), "}": (-1.7, 0.4, 0.6), "<": (-1.1, -0.1, 0.8), ">": (-1.1, -0.1, 0.8),
    "%": (-1.6, 0.0, 1.1), "&": (-1.6, 0.0, 1.1), "@": (-1.4, 0.3, 1.4), "#": (-1.6, 0.0, 1.1),
    "$": (-1.8, 0.2, 0.9), "€": (-1.6, 0.0, 1.1), "£": (-1.6, 0.0, 1.0), "^": (-1.6, -1.0, 0.7),
    "~": (-0.7, -0.35, 0.8), "º": (-1.6, -0.9, 0.5), "ª": (-1.6, -0.9, 0.5), "·": (-0.6, -0.4, 0.15),
}


def char_layout(c):
    """(top, bottom, max_width) for a character in x-height units."""
    if c in _PUNCT:
        return _PUNCT[c]
    base = unicodedata.normalize("NFD", c)[0]
    accented = len(unicodedata.normalize("NFD", c)) > 1
    if c.isdigit():
        return CAP
    if base.isalpha() and base.isupper():
        t, b, w = CAP
        if accented and base.lower() != "c":
            t -= 0.45
        if base == "C" and accented:              # Ç: cedilla below
            b = 0.35
        return (t, b, w)
    if base.isalpha():
        if base in _LOWER_SPECIAL:
            t, b, w = _LOWER_SPECIAL[base]
        elif base in _ASC:
            t, b, w = ASC
        elif base in _DESC:
            t, b, w = DESC
        else:
            t, b, w = X
        if accented:
            if base == "c":                        # ç
                b = 0.35
            else:
                t = min(t, -1.5)
        return (t, b, w)
    return (-1.2, 0.0, 1.0)


# ---------------------------------------------------------------- character bank
class CharBank:
    """writer -> label -> list of Ink (exact duplicates removed, raw units)."""

    def __init__(self, uji_samples, flip_y=False):
        self.bank = defaultdict(lambda: defaultdict(list))
        for s in uji_samples:
            ink = remove_duplicates(to_ink(s.strokes, flip_y=flip_y), 0.0)
            self.bank[s.writer][unicodedata.normalize("NFC", s.label)].append(ink)
        self.writers = sorted(self.bank)
        if not self.writers:
            raise ValueError("CharBank needs at least one sample")
        self.labels = sorted({l for w in self.bank.values() for l in w})

    def get(self, writer, c, rng):
        options = self.bank[writer].get(c)
        if not options:                            # fall back to another writer of the SAME split
            ws = [w for w in self.writers if c in self.bank[w]]
            if not ws:
                raise KeyError(f"no samples of {c!r}")
            options = self.bank[rng.choice(ws)][c]
        return options[rng.randrange(len(options))]


# ---------------------------------------------------------------- text sampling
def _english_wordfreq(top_n=20000):
    import spellchecker
    path = Path(spellchecker.__file__).parent / "resources" / "en.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        freq = json.load(f)
    items = sorted(freq.items(), key=lambda kv: -kv[1])
    words = [w for w, _ in items if re.fullmatch(r"[a-z]+", w) and (len(w) > 1 or w in ("a", "i"))]
    words = words[:top_n]
    counts = np.array([freq[w] for w in words], dtype=np.float64)
    return words, counts


class TextSampler:
    """
    Produces label strings made of real English.
    If `corpus_file` exists, random spans of consecutive words are taken from it;
    otherwise words are drawn from the English frequency list bundled with pyspellchecker.
    Characters outside the vocabulary are dropped.
    """

    def __init__(self, allowed_chars, corpus_file=None, min_words=1, max_words=6,
                 max_chars=48, single_char_fraction=0.05):
        self.allowed = set(allowed_chars) | {" "}
        self.min_words, self.max_words = min_words, max_words
        self.max_chars = max_chars
        self.single_char_fraction = single_char_fraction
        self.symbols = sorted(set(allowed_chars) - {" "})
        self.corpus_words = None
        if corpus_file and Path(corpus_file).exists():
            text = Path(corpus_file).read_text(encoding="utf-8", errors="ignore")
            words = [self._clean(w) for w in text.split()]
            self.corpus_words = [w for w in words if w]
            if len(self.corpus_words) < 1000:
                self.corpus_words = None
        if self.corpus_words is None:
            self.words, counts = _english_wordfreq()
            p = np.sqrt(counts)
            self.p = p / p.sum()
            self.cum = np.cumsum(self.p)

    def _clean(self, s):
        s = unicodedata.normalize("NFC", s)
        return "".join(c for c in s if c in self.allowed and c != " ")

    def _freq_words(self, n, rng):
        idx = np.searchsorted(self.cum, [rng.random() for _ in range(n)])
        ws = [self.words[min(i, len(self.words) - 1)] for i in idx]
        if rng.random() < 0.4:
            ws[0] = ws[0].capitalize()
        if rng.random() < 0.05:
            ws[rng.randrange(n)] = str(rng.randrange(0, 10 ** rng.randint(1, 4)))
        if rng.random() < 0.08:
            ws[rng.randrange(n)] = ws[rng.randrange(n)].upper()
        if rng.random() < 0.25:
            ws[-1] += rng.choice(".,.?!:;")
        if n > 2 and rng.random() < 0.1:
            k = rng.randrange(n - 1)
            ws[k] += ","
        return ws

    def sample(self, rng):
        if rng.random() < self.single_char_fraction:
            return "".join(rng.choice(self.symbols) for _ in range(rng.randint(1, 3)))
        n = rng.randint(self.min_words, self.max_words)
        if self.corpus_words:
            i = rng.randrange(max(1, len(self.corpus_words) - n))
            ws = self.corpus_words[i:i + n]
        else:
            ws = self._freq_words(n, rng)
        text = " ".join(w for w in ws if w)
        text = "".join(c for c in text if c in self.allowed)
        if len(text) > self.max_chars:
            text = text[:self.max_chars].rsplit(" ", 1)[0] or text[:self.max_chars]
        text = " ".join(text.split())
        return text or rng.choice(self.symbols)


# ---------------------------------------------------------------- line builder
def _u(rng, lo_hi):
    return rng.uniform(lo_hi[0], lo_hi[1])


class LineBuilder:
    def __init__(self, bank: CharBank, scfg: dict):
        self.bank = bank
        self.c = scfg

    def build(self, text, writer, rng: random.Random):
        """Return raw strokes (list of (N,2) arrays) for `text` written by `writer`."""
        c = self.c
        strokes = []
        cursor = 0.0
        for ch in text:
            if ch == " ":
                cursor += _u(rng, c["space_gap"])
                continue
            glyph = self.bank.get(writer, ch, rng)
            pts = np.concatenate(glyph)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            w, h = x1 - x0, y1 - y0
            top, bottom, maxw = char_layout(ch)
            th = bottom - top
            cands = [th / h if h > 1e-9 else np.inf, maxw / w if w > 1e-9 else np.inf]
            s = min(cands)
            if not np.isfinite(s):
                s = 0.05 / max(w, h, 1e-9) if max(w, h) > 0 else 1.0
            s *= 1.0 + rng.uniform(-c["char_scale_jitter"], c["char_scale_jitter"])
            yc = (top + bottom) / 2 + rng.uniform(-c["baseline_wobble"], c["baseline_wobble"])
            for st in glyph:
                q = np.empty_like(st)
                q[:, 0] = (st[:, 0] - x0) * s + cursor
                q[:, 1] = (st[:, 1] - (y0 + y1) / 2) * s + yc
                strokes.append(q)
            cursor += w * s + _u(rng, c["char_gap"])
        if not strokes:
            raise ValueError(f"nothing to draw for {text!r}")
        return self._augment(strokes, rng)

    def _augment(self, strokes, rng):
        c = self.c
        k = np.tan(np.radians(_u(rng, c["slant_deg"])))
        t = np.radians(_u(rng, c["rotate_deg"]))
        sc = _u(rng, c["scale"])
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        nrng = np.random.default_rng(rng.getrandbits(32))
        out = []
        for s in strokes:
            q = s.copy()
            q[:, 0] = q[:, 0] - q[:, 1] * k            # positive k leans the line right
            q = (q @ R.T) * sc
            q = q + nrng.normal(0, c["point_jitter"], size=q.shape)
            out.append(q * 100.0)                    # arbitrary "device units"
        return out


def ctc_feasible(n_frames, label_ids):
    """CTC needs T >= L + number of adjacent repeated labels."""
    repeats = sum(1 for a, b in zip(label_ids, label_ids[1:]) if a == b)
    return n_frames >= len(label_ids) + repeats
