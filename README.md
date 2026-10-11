# ScribbleAI — Stroke-Level (Online) Handwriting Recognition

A BiLSTM + CTC sequence model that reads the **pen trajectory** (not an image) and outputs text.
Trained on **UJIpenchars2** isolated characters assembled into **synthetic lines**, and tested on an
**in-house** dataset of real handwriting collected from unseen writers. (The original synopsis used
IAM-OnDB; access never arrived, so the plan changed — see `docs/GUIDE.md` §0.)

**Full step-by-step instructions: [`docs/GUIDE.md`](docs/GUIDE.md).**
Colab runbook: [`notebooks/scribbleai_colab.ipynb`](notebooks/scribbleai_colab.ipynb).

## Pipeline

```
pen strokes (x, y[, t, p])
  -> clean (duplicates, hooks) -> normalise (baseline, slant, height — per LINE)
  -> arc-length resampling -> 8-dim features per point
  -> 3 x BiLSTM(256) -> linear -> CTC
  -> greedy | beam | beam + char n-gram LM | + spell correction
  -> text
```

## Repo structure

```
scribbleai/
├── configs/base.yaml              all hyperparameters (vocab_size / blank_index are written by build_dataset)
├── data/raw/ujipenchars2/         ujipenchars2.txt, uji2.names (download; not in git)
├── data/inhouse/                  collector .jsonl exports (not in git)
├── data/processed/                features, manifests, vocab, stats, LMs (generated)
├── src/
│   ├── preprocessing/  parse_uji, parse_inhouse, parse_xml, clean, normalize, resample, features,
│   │                   vocab, synth_lines, pipeline (THE preprocess function), build_dataset, visualize
│   ├── models/         baseline (kNN/MLP), dataset (+collate), bilstm_ctc
│   ├── decoding/       greedy, beam (prefix beam search + char LM; pyctcdecode wrapper), lm, spell
│   ├── evaluation/     metrics (CER/WER/latency), confusion, analysis
│   ├── train.py        training loop (resumable, Drive checkpoints, W&B)
│   ├── sanity_checks.py the five pre-training checks
│   ├── evaluate.py     full evaluation + error analysis report
│   └── inference.py    Recognizer: strokes -> text (used by evaluate and the demo)
├── demo/
│   ├── collector.html  data-collection page (browser-tested)
│   ├── app.py          live demo server: canvas at /, Gradio batch scorer at /gradio
│   └── static/index.html
├── notebooks/scribbleai_colab.ipynb
├── tests/              unit + integration tests (pytest)
└── docs/GUIDE.md
```

## Quick commands (run from the repo root)

```bash
pip install -r requirements.txt && pip install --no-deps pyctcdecode
python setup_env.py
python -m pytest tests -q

python -m src.preprocessing.build_dataset          # vocab, synthetic val/test, stats, in-house sets
python -m src.models.baseline                      # Objective 2
python -m src.sanity_checks                        # must print ALL CHECKS PASSED
python -m src.train --ckpt-dir /content/drive/MyDrive/scribbleai/checkpoints --wandb
python -m src.decoding.lm                          # character LM (+ word ARPA)
python -m src.evaluate --ckpt checkpoints/best.pt  # results/<run>/summary.md
python demo/app.py --ckpt checkpoints/best.pt      # http://localhost:7860
```

## Notes
* `kenlm` is **not required**: beam search and the character n-gram LM are implemented in pure Python.
* Never `pip install pyctcdecode` normally on Colab: it downgrades numpy and breaks scipy. Use `--no-deps`.
* `tests/make_fake_uji.py` writes a font-based stand-in file in UJI format for smoke tests only;
  never report numbers from it.
