"""
src/preprocessing/visualize.py -- look at the data before training on it.

    python -m src.preprocessing.visualize uji      --n 12   # raw UJI characters
    python -m src.preprocessing.visualize synth    --n 6    # synthetic lines: raw vs preprocessed
    python -m src.preprocessing.visualize inhouse  --n 6    # collected samples: raw vs preprocessed

Images are written to results/figures/. If UJI characters look upside down,
set data.uji_flip_y: true in the config.
"""
import argparse
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, resolve


def draw(ax, strokes, title="", dots=False):
    for s in strokes:
        s = [(p[0], p[1]) for p in s]
        xs, ys = [p[0] for p in s], [p[1] for p in s]
        ax.plot(xs, ys, "-", lw=1.4, color="#1d3557")
        if dots:
            ax.plot(xs, ys, ".", ms=2.5, color="#e63946")
    ax.set_aspect("equal")
    ax.invert_yaxis()                     # y points down
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["uji", "synth", "inhouse"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    d, pcfg = cfg["data"], cfg["preprocessing"]
    rng = random.Random(a.seed)
    out = resolve("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    from src.preprocessing.pipeline import normalize_ink

    if a.what in ("uji", "synth"):
        from src.preprocessing.parse_uji import parse_uji_file, make_splits
        samples = parse_uji_file(resolve(d["uji_file"]))
    if a.what == "uji":
        pick = rng.sample(samples, a.n)
        fig, axes = plt.subplots(2, a.n, figsize=(1.6 * a.n, 3.6))
        for i, s in enumerate(pick):
            raw = [[(x, -y if d.get("uji_flip_y") else y) for x, y in st] for st in s.strokes]
            draw(axes[0, i], raw, f"{s.label!r} W{s.writer}")
            draw(axes[1, i], normalize_ink(s.strokes, pcfg, d.get("uji_flip_y", False)), "processed", dots=True)
    elif a.what == "synth":
        from src.preprocessing.vocab import Vocab
        from src.preprocessing.synth_lines import CharBank, TextSampler, LineBuilder
        train, _, _ = make_splits(samples, tuple(d["uji_val_writers"]))
        vocab = Vocab.from_labels(s.label for s in samples)
        bank = CharBank(train, d.get("uji_flip_y", False))
        sampler = TextSampler(vocab.chars, resolve(d["corpus_file"]))
        lb = LineBuilder(bank, cfg["synthetic"])
        fig, axes = plt.subplots(a.n, 2, figsize=(14, 1.5 * a.n))
        for i in range(a.n):
            text, w = sampler.sample(rng), rng.choice(bank.writers)
            st = lb.build(text, w, rng)
            draw(axes[i, 0], st, f"raw: {text!r} (writer {w})")
            draw(axes[i, 1], normalize_ink(st, pcfg), "after preprocess()", dots=True)
    else:
        from src.preprocessing.parse_inhouse import load_dir
        samples = load_dir(resolve(d["inhouse_dir"]))
        pick = rng.sample(samples, min(a.n, len(samples)))
        fig, axes = plt.subplots(len(pick), 2, figsize=(14, 1.6 * len(pick)), squeeze=False)
        for i, s in enumerate(pick):
            draw(axes[i, 0], s.strokes, f"{s.writer}: {s.text!r} ({s.device}, {s.duration:.1f}s)")
            draw(axes[i, 1], normalize_ink(s.strokes, pcfg), "after preprocess()", dots=True)
    fig.tight_layout()
    path = out / f"{a.what}_samples.png"
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    main()
