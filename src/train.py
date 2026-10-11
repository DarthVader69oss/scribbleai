"""
src/train.py -- training loop for the BiLSTM + CTC recogniser.

    # normal training (Colab: put checkpoints on Drive!)
    python -m src.train --ckpt-dir /content/drive/MyDrive/scribbleai_checkpoints --wandb

    # resume after a Colab disconnect: just run the same command again
    #   (it picks up <ckpt-dir>/last.pt automatically; --no-resume to start over)

    # in-house fine-tuning experiment (Phase 3)
    python -m src.train --finetune --init <ckpt-dir>/best.pt --ckpt-dir <other-dir>

Each epoch: zero_grad -> forward -> CTC loss -> backward -> clip_grad_norm_ -> step,
then validation loss + greedy CER/WER, checkpoint last.pt (always) and best.pt
(lowest val CER). Checkpoints are self-contained (config, vocab, feature stats),
so the demo and evaluation only need the .pt file.
"""
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

from src.utils import load_config, resolve, set_seed
from src.preprocessing.vocab import Vocab
from src.preprocessing.features import FeatureNormalizer
from src.models.bilstm_ctc import BiLSTMCTC
from src.models.dataset import SavedDataset, OnlineSynthDataset, collate_fn
from src.decoding.greedy import greedy_decode_batch
from src.evaluation.metrics import corpus_scores


def build_online_train(cfg, vocab, norm, n, seed):
    from src.preprocessing.parse_uji import parse_uji_file, make_splits
    from src.preprocessing.synth_lines import CharBank, TextSampler, LineBuilder
    d, s = cfg["data"], cfg["synthetic"]
    samples = parse_uji_file(resolve(d["uji_file"]))
    train, _, _ = make_splits(samples, tuple(d["uji_val_writers"]))
    bank = CharBank(train, d.get("uji_flip_y", False))
    sampler = TextSampler(vocab.chars, resolve(d["corpus_file"]), s["min_words"], s["max_words"],
                          s["max_chars"], s["single_char_fraction"])
    return OnlineSynthDataset(bank, sampler, LineBuilder(bank, s), vocab, norm,
                              cfg["preprocessing"], n, seed)


@torch.no_grad()
def evaluate(model, loader, ctc, vocab, device, max_batches=None):
    model.eval()
    tot_loss, nb, refs, hyps = 0.0, 0, [], []
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        lp, out_len = model(b["x"].to(device), b["lengths"].to(device))
        loss = ctc(lp, b["targets"].to(device), out_len, b["target_lengths"].to(device))
        tot_loss += loss.item(); nb += 1
        refs += b["texts"]
        hyps += [vocab.decode(h) for h in greedy_decode_batch(lp, out_len, vocab.blank)]
    s = corpus_scores(refs, hyps)
    s["loss"] = tot_loss / max(nb, 1)
    return s, list(zip(refs, hyps))


def save_ckpt(path, model, opt, sched, epoch, best, cfg, vocab, norm, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "epoch": epoch, "best_cer": best,
                "config": cfg, "vocab": vocab.chars,
                "norm_mean": norm.mean.tolist(), "norm_std": norm.std.tolist(),
                "history": history}, tmp)
    tmp.replace(path)                      # atomic: a disconnect never leaves a half-written file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (fine-tuning)")
    ap.add_argument("--finetune", action="store_true", help="train on inhouse_train (+ synthetic replay)")
    ap.add_argument("--max-val-batches", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    d, t = cfg["data"], cfg["training"]
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pdir = resolve(d["processed_dir"])
    vocab = Vocab.load(resolve(d["vocab_file"]))
    if cfg["features"]["vocab_size"] not in (None, vocab.size):
        raise ValueError(f"config vocab_size={cfg['features']['vocab_size']} but vocab.json has {vocab.size}")
    norm = FeatureNormalizer.load(resolve(d["stats_file"]))
    ckpt_dir = Path(a.ckpt_dir or resolve(t["checkpoint_dir"]))

    epochs = a.epochs or (cfg["finetune"]["num_epochs"] if a.finetune else t["num_epochs"])
    lr = cfg["finetune"]["learning_rate"] if a.finetune else t["learning_rate"]
    bs = cfg["finetune"]["batch_size"] if a.finetune else t["batch_size"]

    # ---------------- data
    if a.finetune:
        ih = SavedDataset(pdir, "inhouse_train", vocab, norm)
        replay = build_online_train(cfg, vocab, norm, len(ih), cfg["seed"])   # 50/50 replay vs forgetting
        train_ds = ConcatDataset([ih, replay])
        print(f"fine-tuning on {len(ih)} in-house lines + {len(replay)} synthetic replay lines")
    elif t["train_source"] == "online":
        train_ds = build_online_train(cfg, vocab, norm, t["lines_per_epoch"], cfg["seed"])
    else:
        train_ds = SavedDataset(pdir, "train", vocab, norm)
    val_ds = SavedDataset(pdir, "val", vocab, norm)
    nw = t["num_workers"]
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                          num_workers=nw, pin_memory=device.type == "cuda",
                          persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ---------------- model / optim
    model = BiLSTMCTC.from_config(cfg, vocab.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=t["lr_patience"])
    ctc = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)
    print(f"device={device}  params={model.num_parameters():,}  vocab={vocab.size} (blank={vocab.blank})")

    start, best, history = 0, math.inf, []
    last = ckpt_dir / "last.pt"
    if a.init:
        st = torch.load(a.init, map_location=device, weights_only=False)
        if st["vocab"] != vocab.chars:
            raise ValueError("--init checkpoint was trained with a different vocabulary")
        model.load_state_dict(st["model"])
        print(f"initialised weights from {a.init}")
    if last.exists() and not a.no_resume:
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start, best, history = st["epoch"] + 1, st["best_cer"], st.get("history", [])
        print(f"resumed from {last} at epoch {start} (best val CER {best:.4f})")

    run = None
    if a.wandb:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"],
                         name=cfg["experiment_name"] + ("_ft" if a.finetune else ""), config=cfg,
                         resume="allow", id=(cfg["experiment_name"] + ("_ft" if a.finetune else "")))

    bad_epochs, best_loss = 0, min([h["val_loss"] for h in history] or [math.inf])
    for epoch in range(start, epochs):
        if isinstance(train_ds, OnlineSynthDataset):
            train_ds.set_epoch(epoch)
        elif isinstance(train_ds, ConcatDataset):
            for ds in train_ds.datasets:
                if hasattr(ds, "set_epoch"):
                    ds.set_epoch(epoch)
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, b in enumerate(train_dl):
            x, L = b["x"].to(device), b["lengths"].to(device)
            opt.zero_grad(set_to_none=True)
            lp, out_len = model(x, L)
            loss = ctc(lp, b["targets"].to(device), out_len, b["target_lengths"].to(device))
            if not torch.isfinite(loss):
                print(f"  non-finite loss at step {step}, skipping batch")
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip_norm"])
            opt.step()
            run_loss += loss.item(); n += 1
            if step % 100 == 0:
                print(f"  ep {epoch} step {step}/{len(train_dl)} loss {loss.item():.3f} gnorm {gn:.2f}")
        train_loss = run_loss / max(n, 1)
        vs, examples = evaluate(model, val_dl, ctc, vocab, device, a.max_val_batches)
        sched.step(vs["loss"])        # loss, not CER: CER sits at 1.0 during the early CTC plateau
        lr_now = opt.param_groups[0]["lr"]
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": vs["loss"], "val_cer": vs["cer"],
               "val_wer": vs["wer"], "lr": lr_now, "secs": time.time() - t0}
        history.append(rec)
        print(f"epoch {epoch}: train_loss {train_loss:.3f}  val_loss {vs['loss']:.3f}  "
              f"val CER {vs['cer']:.4f}  WER {vs['wer']:.4f}  lr {lr_now:.2e}  ({rec['secs']:.0f}s)")
        for r, h in examples[:t["log_examples"]]:
            print(f"    REF: {r}\n    HYP: {h}")
        if run:
            run.log({**rec, "examples": "\n".join(f"{r} -> {h}" for r, h in examples[:10])})

        improved = vs["cer"] < best
        if improved:
            best = vs["cer"]
        if improved or vs["loss"] < best_loss - 1e-3:   # patience counts epochs with no progress in either
            bad_epochs = 0
        else:
            bad_epochs += 1
        best_loss = min(best_loss, vs["loss"])
        save_ckpt(last, model, opt, sched, epoch, best, cfg, vocab, norm, history)
        if improved:
            save_ckpt(ckpt_dir / "best.pt", model, opt, sched, epoch, best, cfg, vocab, norm, history)
            print(f"  new best val CER {best:.4f} -> {ckpt_dir / 'best.pt'}")
        if bad_epochs >= t["early_stop_patience"]:
            print(f"early stopping: no improvement for {bad_epochs} epochs")
            break
    if run:
        run.finish()
    print(f"done. best val CER {best:.4f}. checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
