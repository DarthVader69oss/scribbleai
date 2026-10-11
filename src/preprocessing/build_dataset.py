"""
src/preprocessing/build_dataset.py  (C1.10)

Run from the repo root:
    python -m src.preprocessing.build_dataset                 # full build
    python -m src.preprocessing.build_dataset --quick         # tiny build for smoke tests

What it does
  1. Parses UJIpenchars2 and makes writer-disjoint train/val/test splits.
  2. Builds the vocabulary from the data (chars + space + blank-last), saves
     data/processed/vocab.json and writes vocab_size / blank_index into the config.
  3. Builds synthetic lines for val and test (and train if training.train_source == "saved"),
     each line from ONE writer of the right split.
  4. Encodes isolated UJI test characters (to compare with the baseline classifier).
  5. Loads in-house .jsonl exports (if any) and splits them by writer.
  6. Fits feature mean/std on TRAINING lines only and saves them.

Output per set NAME in data/processed/:
    NAME_feats.npy   all frames of all samples concatenated (float32, un-standardised)
    NAME.csv         manifest: id, text, writer, source, offset, n_frames, n_strokes, ...
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

from src.utils import load_config, resolve, set_seed, update_config_values
from src.preprocessing.parse_uji import parse_uji_file, make_splits
from src.preprocessing.parse_inhouse import load_dir, writer_split
from src.preprocessing.pipeline import preprocess
from src.preprocessing.features import FeatureNormalizer
from src.preprocessing.vocab import Vocab
from src.preprocessing.synth_lines import CharBank, TextSampler, LineBuilder, ctc_feasible
from src.preprocessing.resample import ink_length

MANIFEST_FIELDS = ["id", "text", "writer", "source", "offset", "n_frames", "n_strokes",
                   "n_chars", "duration", "ink_length"]


def save_set(name, rows, feats, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    for r, f in zip(rows, feats):
        r["offset"], r["n_frames"] = offset, len(f)
        offset += len(f)
    allf = np.concatenate(feats).astype(np.float32) if feats else np.zeros((0, 8), np.float32)
    np.save(out_dir / f"{name}_feats.npy", allf)
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    print(f"  saved {name}: {len(rows)} samples, {len(allf)} frames")


def synth_set(name, n, bank, sampler, builder, vocab, pcfg, seed):
    rng = random.Random(seed)
    rows, feats, skipped = [], [], 0
    t0 = time.time()
    while len(rows) < n:
        writer = rng.choice(bank.writers)
        text = sampler.sample(rng)
        strokes = builder.build(text, writer, rng)
        F, _ = preprocess(strokes, pcfg)
        ids = vocab.encode(text)
        if not ctc_feasible(len(F), ids):
            skipped += 1
            continue
        rows.append({"id": f"{name}_{len(rows):06d}", "text": text, "writer": writer,
                     "source": "synthetic", "n_strokes": len(strokes), "n_chars": len(text)})
        feats.append(F)
        if len(rows) % 2000 == 0:
            print(f"    {name}: {len(rows)}/{n}  ({time.time() - t0:.0f}s)")
    if skipped:
        print(f"  {name}: skipped {skipped} CTC-infeasible lines (increase density if this is large)")
    return rows, feats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--quick", action="store_true", help="tiny sets for smoke tests")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d, pcfg, scfg = cfg["data"], cfg["preprocessing"], cfg["synthetic"]
    out = resolve(d["processed_dir"])

    # 1. UJI + splits
    uji_path = resolve(d["uji_file"])
    if not uji_path.exists():
        sys.exit(f"UJIpenchars2 not found at {uji_path}. Download it first (see docs/GUIDE.md).")
    samples = parse_uji_file(uji_path)
    train, val, test = make_splits(samples, val_writers=tuple(d["uji_val_writers"]))
    print(f"UJI: total={len(samples)} train={len(train)} val={len(val)} test={len(test)}")

    # 2. vocab (from data) -> file + config
    vocab = Vocab.from_labels(s.label for s in samples)
    vocab.save(resolve(d["vocab_file"]))
    update_config_values(args.config, {"vocab_size": vocab.size, "blank_index": vocab.blank})
    print(f"vocab: {len(vocab.chars)} chars + blank -> vocab_size={vocab.size}, blank_index={vocab.blank}")

    # 3. synthetic lines
    flip = d.get("uji_flip_y", False)
    banks = {k: CharBank(v, flip_y=flip) for k, v in (("train", train), ("val", val), ("test", test))}
    sampler = TextSampler(vocab.chars, resolve(d["corpus_file"]), scfg["min_words"], scfg["max_words"],
                          scfg["max_chars"], scfg["single_char_fraction"])
    print("text source:", "corpus file" if sampler.corpus_words else "English word-frequency list")
    q = args.quick
    n_val = 64 if q else scfg["val_lines"]
    n_test = 64 if q else scfg["test_lines"]
    saved_train = cfg["training"]["train_source"] == "saved"
    n_train = (256 if q else scfg["train_lines"]) if saved_train else (256 if q else 3000)
    train_name = "train" if saved_train else "train_stats_sample"

    sets = {}
    for name, bank, n, seed in ((train_name, banks["train"], n_train, 1),
                                ("val", banks["val"], n_val, 2), ("test", banks["test"], n_test, 3)):
        builder = LineBuilder(bank, scfg)
        sets[name] = synth_set(name, n, bank, sampler, builder, vocab, pcfg, cfg["seed"] * 1000 + seed)

    # 6. feature stats on TRAIN only
    norm = FeatureNormalizer.fit(sets[train_name][1])
    norm.save(resolve(d["stats_file"]))
    print("feature mean:", np.round(norm.mean, 3).tolist())
    print("feature std :", np.round(norm.std, 3).tolist())

    for name, (rows, feats) in sets.items():
        save_set(name, rows, feats, out)

    # 4. isolated UJI test characters
    rows, feats = [], []
    for s in (test[:300] if q else test):
        F, _ = preprocess(s.strokes, pcfg, flip_y=flip)
        rows.append({"id": s.sample_id + "_" + s.label, "text": s.label, "writer": s.writer,
                     "source": "uji_char", "n_strokes": s.num_strokes, "n_chars": 1})
        feats.append(F)
    save_set("uji_chars_test", rows, feats, out)

    # 5. in-house
    inh = resolve(d["inhouse_dir"])
    if any(inh.glob("*.jsonl")):
        ih = load_dir(inh)
        tw = d.get("inhouse_test_writers") or []
        if tw:
            ih_train, ih_test = writer_split(ih, tw)
        else:
            print("  WARNING: inhouse_test_writers is empty -> ALL in-house data is test-only")
            ih_train, ih_test = [], ih
        for name, group in (("inhouse_train", ih_train), ("inhouse_test", ih_test)):
            rows, feats = [], []
            for i, s in enumerate(group):
                bad = vocab.unknown_chars(s.text)
                if bad:
                    print(f"  {name}: {s.writer} {s.text!r} has chars outside vocab {bad} (dropped from the label)")
                F, ink = preprocess(s.strokes, pcfg)
                label = " ".join("".join(c for c in vocab.normalize_text(s.text) if c in vocab.index).split())
                if not label:
                    continue
                rows.append({"id": f"{name}_{i:05d}", "text": label, "writer": s.writer,
                             "source": f"inhouse:{s.device}", "n_strokes": len(s.strokes),
                             "n_chars": len(s.text), "duration": round(s.duration, 3),
                             "ink_length": round(ink_length(ink), 3)})
                feats.append(F)
            if rows:
                save_set(name, rows, feats, out)
    else:
        print(f"  no in-house .jsonl files in {inh} yet (skipping)")
    print("done.")


if __name__ == "__main__":
    main()
