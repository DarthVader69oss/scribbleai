"""
src/evaluation/analysis.py -- error analysis (Objective 7).

  cer_by_bins()      CER as a function of a per-line quantity (length, strokes, speed...)
  worst_samples()    the N lines with the highest CER
  plot_bins()        bar chart for the report
"""
import numpy as np

from src.evaluation.metrics import edit_distance


def cer_by_bins(rows, key, bins):
    """
    rows: dicts with 'ref', 'hyp' and `key`. bins: list of edges, e.g. [0, 5, 10, 20, 40, 1e9].
    Returns [{"bin": "5-10", "n": .., "cer": ..}, ...] (corpus CER inside each bin).
    """
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if r.get(key) not in (None, "") and lo <= float(r[key]) < hi]
        if not sel:
            continue
        e = sum(edit_distance(r["ref"], r["hyp"]) for r in sel)
        n = sum(len(r["ref"]) for r in sel)
        label = f"{lo:g}-{hi:g}" if hi < 1e8 else f"{lo:g}+"
        out.append({"bin": label, "n": len(sel), "cer": e / max(n, 1)})
    return out


def quantile_bins(values, k=5):
    v = np.asarray([float(x) for x in values if x not in (None, "")])
    if len(v) == 0:
        return [0, 1e9]
    edges = sorted(set(np.round(np.quantile(v, np.linspace(0, 1, k + 1)), 3).tolist()))
    edges[-1] = edges[-1] + 1e-6
    return edges


def worst_samples(rows, n=10):
    scored = [dict(r, cer=edit_distance(r["ref"], r["hyp"]) / max(len(r["ref"]), 1)) for r in rows]
    return sorted(scored, key=lambda r: -r["cer"])[:n]


def plot_bins(table, path, xlabel, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not table:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar([t["bin"] for t in table], [100 * t["cer"] for t in table], color="#2b5fa8")
    for i, t in enumerate(table):
        ax.text(i, 100 * t["cer"], f"n={t['n']}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("CER (%)"); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
