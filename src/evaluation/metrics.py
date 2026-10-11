"""
src/evaluation/metrics.py

CER = (S + D + I) / N over characters, WER the same over whitespace tokens.
Corpus-level numbers are total edits / total reference length (not a mean of
per-line rates), which is the convention in the literature.
"""
import time
from contextlib import contextmanager

try:
    import Levenshtein as _lev
except ImportError:          # pure-Python fallback
    _lev = None


def edit_distance(a, b):
    """Levenshtein distance between two sequences (strings or lists)."""
    if _lev is not None and isinstance(a, str) and isinstance(b, str):
        return _lev.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(len(r), 1)


def corpus_scores(refs, hyps):
    ce = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    cn = sum(len(r) for r in refs)
    we = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    wn = sum(len(r.split()) for r in refs)
    exact = sum(r == h for r, h in zip(refs, hyps))
    return {"cer": ce / max(cn, 1), "wer": we / max(wn, 1),
            "line_acc": exact / max(len(refs), 1), "n": len(refs)}


class Latency:
    """Collects per-call wall-clock times in milliseconds."""

    def __init__(self):
        self.ms = []

    @contextmanager
    def time(self):
        t0 = time.perf_counter()
        yield
        self.ms.append((time.perf_counter() - t0) * 1000)

    def summary(self):
        if not self.ms:
            return {}
        s = sorted(self.ms)
        return {"mean_ms": sum(s) / len(s), "p50_ms": s[len(s) // 2],
                "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))], "max_ms": s[-1]}
