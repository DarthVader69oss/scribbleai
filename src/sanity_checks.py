"""
src/sanity_checks.py -- the five checks to pass BEFORE any real training run (guide 5.5).

    python -m src.sanity_checks [--steps 600]

  1. Overfit one tiny batch (4 lines): loss -> ~0 and greedy output == reference.
  2. Label ids <-> characters round-trip; blank is the last index and never a label.
  3. Feature scale: standardised features have mean ~0 / std ~1 on the continuous dims.
  4. Loss is finite at every step (target longer than input makes CTC return inf).
  5. Every layer receives a non-zero gradient.
Exits with status 1 if any check fails.
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.utils import load_config, resolve, set_seed
from src.preprocessing.vocab import Vocab
from src.preprocessing.features import FeatureNormalizer, FEATURE_NAMES, CONTINUOUS
from src.models.bilstm_ctc import BiLSTMCTC
from src.models.dataset import SavedDataset, collate_fn
from src.decoding.greedy import greedy_decode_batch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    d = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocab.load(resolve(d["vocab_file"]))
    norm = FeatureNormalizer.load(resolve(d["stats_file"]))
    ds = SavedDataset(resolve(d["processed_dir"]), "val", vocab, norm)
    ok = True

    # ---- 2. labels
    print("[2] label ids <-> characters")
    x0, ids0, text0, _ = ds[0]
    print(f"    text   : {text0!r}\n    ids    : {ids0.tolist()}\n    decoded: {vocab.decode(ids0.tolist())!r}")
    bad = [ds[i][2] for i in range(min(200, len(ds))) if vocab.decode(ds[i][1].tolist()) != vocab.normalize_text(ds[i][2])]
    blank_used = any((ds[i][1] == vocab.blank).any() for i in range(min(200, len(ds))))
    c2 = not bad and not blank_used and vocab.blank == vocab.size - 1
    print(f"    blank={vocab.blank} (last index: {vocab.blank == vocab.size - 1}), blank used as label: {blank_used}, "
          f"round-trip failures: {len(bad)}  -> {'PASS' if c2 else 'FAIL'}")
    ok &= c2

    # ---- 3. feature scale
    print("[3] feature scale (standardised)")
    X = np.concatenate([ds[i][0].numpy() for i in range(min(200, len(ds)))])
    for j, nm in enumerate(FEATURE_NAMES):
        print(f"    {nm:10s} mean {X[:, j].mean():+.3f}  std {X[:, j].std():.3f}  min {X[:, j].min():+.2f}  max {X[:, j].max():+.2f}")
    c3 = all(abs(X[:, j].mean()) < 0.5 and 0.3 < X[:, j].std() < 3 for j in CONTINUOUS) and np.isfinite(X).all()
    print(f"    -> {'PASS' if c3 else 'FAIL'}")
    ok &= c3

    # ---- 1/4/5. overfit a tiny batch
    print(f"[1/4/5] overfitting {a.n} lines for {a.steps} steps on {device}")
    batch = next(iter(DataLoader(Subset(ds, range(a.n)), batch_size=a.n, collate_fn=collate_fn)))
    model = BiLSTMCTC.from_config(cfg, vocab.size).to(device)
    model.dropout.p = 0.0
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=False)   # zero_infinity off: we WANT to see inf
    x, L = batch["x"].to(device), batch["lengths"].to(device)
    tg, tl = batch["targets"].to(device), batch["target_lengths"].to(device)
    finite, c5 = True, True
    for step in range(a.steps):
        model.train()
        opt.zero_grad()
        lp, ol = model(x, L)
        loss = ctc(lp, tg, ol, tl)
        if not torch.isfinite(loss):
            finite = False
            print(f"    non-finite loss at step {step}")
            break
        loss.backward()
        if step == 0:
            dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
            c5 = not dead
            print(f"[5] layers with zero gradient: {dead or 'none'}  -> {'PASS' if c5 else 'FAIL'}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            print(f"    step {step:4d} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        lp, ol = model(x, L)
    hyps = [vocab.decode(h) for h in greedy_decode_batch(lp, ol, vocab.blank)]
    match = [h == vocab.normalize_text(r) for h, r in zip(hyps, batch["texts"])]
    for r, h in zip(batch["texts"], hyps):
        print(f"    REF {r!r}\n    HYP {h!r}")
    c1 = all(match) and loss.item() < 0.5
    print(f"[1] overfit one batch -> {'PASS' if c1 else 'FAIL (if loss is still falling, rerun with --steps 1200)'}")
    print(f"[4] loss finite every step -> {'PASS' if finite else 'FAIL'}")
    ok &= c1 and finite and c5
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED -- fix before training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
