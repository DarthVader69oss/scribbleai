"""
src/evaluation/confusion.py -- per-character confusion from an edit-distance backtrace.

align(ref, hyp) returns the operations that turn ref into hyp:
  ("=", r, h)  match      ("S", r, h)  substitution
  ("D", r, "") deletion   ("I", "", h) insertion
Insertions/deletions are counted against the special symbol EPS ("∅").
"""
from collections import Counter

import numpy as np

EPS = "∅"


def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j - 1])); j -= 1
    return ops[::-1]


def confusion_counts(refs, hyps):
    """Counter over (true_char, predicted_char) incl. matches; EPS for ins/del."""
    c = Counter()
    for r, h in zip(refs, hyps):
        for op, a, b in align(r, h):
            c[(a or EPS, b or EPS)] += 1
    return c


def top_confusions(counts, k=20):
    errs = [((a, b), n) for (a, b), n in counts.items() if a != b]
    return sorted(errs, key=lambda x: -x[1])[:k]


def per_char_error(counts):
    """For each reference character: occurrences and error rate."""
    tot, err = Counter(), Counter()
    for (a, b), n in counts.items():
        if a == EPS:
            continue
        tot[a] += n
        if a != b:
            err[a] += n
    return {c: {"n": tot[c], "err_rate": err[c] / tot[c]} for c in tot}


def plot_confusion(counts, path, max_chars=40, title="Character confusion"):
    """Heat-map of the most frequent reference characters (row-normalised)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freq = Counter()
    for (a, b), n in counts.items():
        if a != EPS:
            freq[a] += n
    chars = [c for c, _ in freq.most_common(max_chars)]
    labels = chars + [EPS]
    idx = {c: i for i, c in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for (a, b), n in counts.items():
        if a in idx and b in idx:
            M[idx[a], idx[b]] += n
    M = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    show = [repr(c)[1:-1] if c != " " else "␣" for c in labels]
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(show, fontsize=7)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(show, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
