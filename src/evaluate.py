"""
src/evaluate.py -- full evaluation + error analysis for one checkpoint (Objectives 3-7).

    python -m src.evaluate --ckpt checkpoints/best.pt
    python -m src.evaluate --ckpt checkpoints/best.pt --sets test inhouse_test --max-items 500

For every set (synthetic test lines, isolated UJI test characters, in-house test writers):
  * CER / WER / exact-line accuracy for each decoding stage:
        greedy -> beam -> beam + char LM -> beam + char LM + spell
    (each stage switchable, so improvements are attributed to the right component)
  * CPU latency per line (model + decoder) for each stage
  * beam-width sweep (config decoding.beam_width_options) on the synthetic test set
  * confusion matrix, CER vs line length / stroke count / writing speed (in-house only),
    worst samples
Writes everything to results/<run-name>/ (summary.json, summary.md, *.csv, *.png).
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils import resolve, save_json
from src.inference import Recognizer, MODES
from src.models.dataset import SavedDataset, collate_fn
from src.evaluation.metrics import corpus_scores, Latency
from src.evaluation.confusion import confusion_counts, top_confusions, plot_confusion, per_char_error
from src.evaluation.analysis import cer_by_bins, quantile_bins, worst_samples, plot_bins


@torch.no_grad()
def all_log_probs(rec, ds, device, bs=32):
    """(log_probs (T,C) numpy, ref text, manifest row) for every sample, batched."""
    model = rec.model.to(device)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    out, k = [], 0
    for b in dl:
        lp, ol = model(b["x"].to(device), b["lengths"].to(device))
        lp = lp.cpu().numpy()
        for j in range(lp.shape[1]):
            out.append((lp[:int(ol[j]), j], b["texts"][j], ds.df.iloc[k].to_dict()))
            k += 1
    rec.model.to(rec.device)
    return out


def cpu_latency(rec, ds, n=100):
    """Batch-size-1 model latency on CPU (what the demo experiences)."""
    lat = Latency()
    m = rec.model.cpu()
    torch.set_grad_enabled(False)
    for i in range(min(n, len(ds))):
        x, _, _, _ = ds[i]
        with lat.time():
            m(x.unsqueeze(0), torch.tensor([x.shape[0]]))
    torch.set_grad_enabled(True)
    return lat.summary()


def md_table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="*", default=["test", "uji_chars_test", "inhouse_test"])
    ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--max-items", type=int, default=None, help="cap per set (beam+LM is slow on long sets)")
    ap.add_argument("--sweep-items", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--alpha", type=float, default=None, help="override LM weight (tune on --sets val!)")
    ap.add_argument("--beta", type=float, default=None, help="override length bonus")
    ap.add_argument("--beam", type=int, default=None, help="override default beam width")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = Recognizer(a.ckpt, device="cpu")
    if a.alpha is not None:
        rec.alpha = a.alpha
    if a.beta is not None:
        rec.beta = a.beta
    if a.beam is not None:
        rec.default_beam = a.beam
    modes = [m for m in a.modes if m in rec.available_modes()]
    skipped = sorted(set(a.modes) - set(modes))
    if skipped:
        print(f"skipping modes {skipped} (build the LM with: python -m src.decoding.lm)")
    pdir = resolve(rec.cfg["data"]["processed_dir"])
    out = Path(a.out) if a.out else resolve(f"results/{Path(a.ckpt).parent.name}_{Path(a.ckpt).stem}")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"checkpoint": str(a.ckpt), "epoch": rec.epoch, "alpha": rec.alpha, "beta": rec.beta,
               "beam": rec.default_beam, "val_cer_at_save": rec.best_cer, "sets": {}}
    md = [f"# Evaluation of `{a.ckpt}` (epoch {rec.epoch})\n"]

    for name in a.sets:
        if not (pdir / f"{name}.csv").exists():
            print(f"[{name}] not built, skipping")
            continue
        ds = SavedDataset(pdir, name, rec.vocab, rec.norm, max_items=a.max_items)
        print(f"[{name}] {len(ds)} samples")
        items = all_log_probs(rec, ds, device)
        model_lat = cpu_latency(rec, ds)
        res = {"n": len(items), "model_cpu_latency": model_lat, "modes": {}}
        preds = {}
        set_modes = modes
        if name == "uji_chars_test":
            # a line-level language model is meaningless for single characters (it prefers whatever
            # symbol is most likely to be a whole line on its own), so only greedy/beam here
            set_modes = [m for m in modes if not m.startswith("beam_lm")]
        for mode in set_modes:
            lat = Latency()
            hyps = []
            for lp, _, _ in items:
                with lat.time():
                    hyps.append(rec.decode(lp, mode))
            refs = [rec.vocab.normalize_text(r) for _, r, _ in items]
            sc = corpus_scores(refs, hyps)
            sc["decode_ms"] = lat.summary().get("mean_ms", 0.0)
            sc["total_ms_per_line"] = sc["decode_ms"] + model_lat.get("mean_ms", 0.0)
            res["modes"][mode] = sc
            preds[mode] = hyps
            print(f"   {mode:14s} CER {sc['cer']:.4f}  WER {sc['wer']:.4f}  line-acc {sc['line_acc']:.3f}  "
                  f"{sc['total_ms_per_line']:.1f} ms/line")

        refs = [rec.vocab.normalize_text(r) for _, r, _ in items]
        rows = []
        for i, (_, r, meta) in enumerate(items):
            row = {"id": meta["id"], "writer": meta["writer"], "ref": refs[i],
                   "n_chars": len(refs[i]), "n_strokes": meta.get("n_strokes", ""), "n_frames": meta["n_frames"]}
            dur = meta.get("duration", "")
            if dur not in ("", None) and float(dur) > 0:
                row["chars_per_sec"] = round(len(refs[i]) / float(dur), 3)
                if meta.get("ink_length", "") not in ("", None):
                    row["xheights_per_sec"] = round(float(meta["ink_length"]) / float(dur), 3)
            for m in set_modes:
                row[f"hyp_{m}"] = preds[m][i]
            rows.append(row)
        with open(out / f"{name}_predictions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        # error analysis on greedy output (the raw model)
        g = [dict(r, hyp=r["hyp_greedy"]) for r in rows]
        counts = confusion_counts([r["ref"] for r in g], [r["hyp"] for r in g])
        res["top_confusions"] = [{"true": t, "pred": p, "count": n} for (t, p), n in top_confusions(counts, 25)]
        pce = per_char_error(counts)
        res["hardest_chars"] = sorted(({"char": c, **v} for c, v in pce.items() if v["n"] >= 5),
                                      key=lambda d: -d["err_rate"])[:15]
        plot_confusion(counts, out / f"{name}_confusion.png", title=f"{name}: greedy confusion (row-normalised)")
        analyses = {"n_chars": "line length (characters)", "n_strokes": "number of strokes"}
        if name.startswith("inhouse"):
            analyses.update({"chars_per_sec": "writing speed (characters / s)",
                             "xheights_per_sec": "pen speed (x-heights / s)"})
        res["analysis"] = {}
        for key, label in analyses.items():
            vals = [r.get(key) for r in g]
            if not any(v not in (None, "") for v in vals):
                continue
            bins = [0, 2, 5, 10, 20, 30, 50, 1e9] if key == "n_chars" else quantile_bins(vals)
            tab = cer_by_bins(g, key, bins)
            res["analysis"][key] = tab
            plot_bins(tab, out / f"{name}_cer_vs_{key}.png", label, f"{name}: CER vs {label}")
        res["worst"] = [{"id": r["id"], "ref": r["ref"], "hyp": r["hyp"], "cer": r["cer"]} for r in worst_samples(g, 10)]
        summary["sets"][name] = res

        md.append(f"\n## {name} ({len(items)} samples)\n")
        md.append(md_table([{"decoder": m, **res["modes"][m]} for m in set_modes],
                           ["decoder", "cer", "wer", "line_acc", "total_ms_per_line"]))
        md.append("\nTop confusions (true → predicted): " + ", ".join(
            f"`{c['true']}`→`{c['pred']}` ({c['count']})" for c in res["top_confusions"][:12]) + "\n")
        md.append("\nWorst lines:\n" + "\n".join(f"- `{w['ref']}` → `{w['hyp']}` (CER {w['cer']:.2f})" for w in res["worst"][:5]) + "\n")

    # beam-width sweep on the synthetic test set
    if (pdir / "test.csv").exists():
        ds = SavedDataset(pdir, "test", rec.vocab, rec.norm, max_items=a.sweep_items)
        items = all_log_probs(rec, ds, device)
        refs = [rec.vocab.normalize_text(r) for _, r, _ in items]
        sweep = []
        for mode in [m for m in ("beam", "beam_lm") if m in modes]:
            for bw in rec.cfg["decoding"]["beam_width_options"]:
                lat = Latency()
                hyps = []
                for lp, _, _ in items:
                    with lat.time():
                        hyps.append(rec.decode(lp, mode, bw))
                sc = corpus_scores(refs, hyps)
                sweep.append({"mode": mode, "beam": bw, "cer": sc["cer"], "wer": sc["wer"],
                              "decode_ms": lat.summary()["mean_ms"]})
                print(f"   sweep {mode} beam={bw}: CER {sc['cer']:.4f} WER {sc['wer']:.4f} {sweep[-1]['decode_ms']:.1f} ms")
        summary["beam_sweep"] = sweep
        if sweep:
            md.append("\n## Beam-width sweep (synthetic test subset)\n")
            md.append(md_table(sweep, ["mode", "beam", "cer", "wer", "decode_ms"]))

    sets = summary["sets"]
    if "test" in sets and "inhouse_test" in sets:
        gap = {m: sets["inhouse_test"]["modes"][m]["cer"] - sets["test"]["modes"][m]["cer"]
               for m in modes if m in sets["test"]["modes"]}
        summary["generalisation_gap_cer"] = gap
        md.append("\n## Generalisation gap (in-house CER − synthetic test CER)\n")
        md.append("\n".join(f"- {m}: {100 * v:+.1f} points" for m, v in gap.items()) + "\n")
    base = resolve("results/baseline.json")
    if base.exists() and "uji_chars_test" in sets:
        import json
        b = json.loads(base.read_text())
        md.append("\n## Isolated characters: baseline vs sequence model\n")
        md.append(f"- k-NN baseline error: {b['knn']['test_cer']:.4f}\n- MLP baseline error: {b['mlp']['test_cer']:.4f}\n"
                  f"- BiLSTM-CTC greedy CER: {sets['uji_chars_test']['modes']['greedy']['cer']:.4f}\n")

    save_json(summary, out / "summary.json")
    (out / "summary.md").write_text("".join(md), encoding="utf-8")
    print(f"\nwrote {out}/summary.md, summary.json, predictions, plots")


if __name__ == "__main__":
    main()
