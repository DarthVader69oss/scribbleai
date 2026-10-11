# ScribbleAI — Implementation Guide, Start to Finish (v3)

This guide takes the project from the current repo to a live demo in front of the faculty.
It follows the synopsis, with the changes in the v2 plan: **UJIpenchars2 + synthetic lines + in-house data
instead of IAM-OnDB**.

Legend: **[CODE ✔]** = already written and tested. **[YOU]** = a person has to do it (data, accounts,
collection, running on Colab, the presentation). Commands run from the repo root (`scribbleai/`)
in Git Bash (local) or in a Colab cell (prefix `!`).

---

## 0. What has changed and what is already done

### 0.1 Scope change from the synopsis (tell the faculty first) [YOU]

| Synopsis | Now | Why |
|---|---|---|
| Train/test on IAM-OnDB (221 writers) | Train on **UJIpenchars2** characters assembled into **synthetic lines**; test on synthetic lines from held-out UJI writers **and** on **in-house** real handwriting | IAM-OnDB registration never got approved |
| Objective 1 parses IAM-OnDB XML | Parses UJIpenchars2 + collector JSONL (the IAM parser still exists for later) | as above |
| Objective 3: CER ≤ 12% on the IAM test split | CER on the synthetic test split (greedy), **target set after the first full run** | numbers aren't comparable with IAM literature |
| Objective 6: in-house set is test-only | In-house **test writers** stay test-only; optional **train writers** are used for a fine-tuning experiment | measure transfer + adaptation |
| Objective 7: error vs writing speed | Speed analysis on **in-house data only** | UJI has no timestamps |
| KenLM n-gram | **Character n-gram LM in pure Python** (KenLM optional) | kenlm fails to build on Colab |

Get written (email) approval from your guide before Week 2.

### 0.2 What is already implemented [CODE ✔]

Everything marked ○ in the old structure is now written, plus a few files the plan needed:

| Area | Files | What it does |
|---|---|---|
| Preprocessing | `clean.py`, `normalize.py`, `resample.py`, `features.py`, `pipeline.py` | duplicate and hook removal; baseline, slant and height normalisation **per line**; arc-length resampling; 8-dim features; `preprocess()` is the one function used by training, evaluation and the demo |
| Vocabulary | `vocab.py` | vocabulary built **from the data** (UJI labels + space), blank = last index; `build_dataset` writes `vocab_size`/`blank_index` into the config |
| Synthetic lines | `synth_lines.py` | one writer per line, real English text, typographic layout table (x-height, ascenders, descenders, capitals, punctuation) so a/A, o/O keep their size difference, wider space gaps, augmentation, writer-split safe, CTC feasibility check |
| Dataset build | `build_dataset.py`, `visualize.py` | val/test synthetic lines, isolated UJI test chars, in-house sets, feature stats from train only, `.npy` + manifest CSVs; plots to check the data by eye |
| Baseline | `models/baseline.py` | k-NN + MLP isolated-character classifiers, evaluated on the 20 test writers |
| Model + training | `models/dataset.py`, `models/bilstm_ctc.py`, `train.py`, `sanity_checks.py` | online synthetic data (new lines every epoch), packed 3×BiLSTM-256 + CTC, grad clipping, ReduceLROnPlateau, early stopping, resumable atomic checkpoints, W&B, fine-tuning mode, the five sanity checks |
| Decoding | `decoding/greedy.py`, `beam.py`, `lm.py`, `spell.py` | greedy (collapse **then** remove blanks), CTC prefix beam search with character n-gram LM fusion, ARPA word LM writer, pyctcdecode wrapper, spell correction |
| Evaluation | `evaluation/metrics.py`, `confusion.py`, `analysis.py`, `evaluate.py`, `inference.py` | CER/WER/line accuracy, CPU latency, per-stage comparison, beam-width sweep, confusion matrix, CER vs length / strokes / writing speed, worst samples, generalisation gap, markdown report |
| Demo | `demo/app.py`, `demo/static/index.html` | live canvas that re-recognises after every stroke, decoder switch, side-by-side decoder comparison, "what the model sees" view, latency read-out; Gradio page at `/gradio` scores a whole collector file |
| Collector | `demo/collector.html` | now **browser-tested**: drawing, save, export and re-loading with `parse_inhouse` all work |
| Notebook | `notebooks/scribbleai_colab.ipynb` | the whole Colab run, top to bottom |
| Tests | `tests/` | 120 tests (+7 that run once the real UJI file is present), including a tiny end-to-end train → evaluate → demo-API test |

What was verified here: all tests pass, and the full pipeline (build → baseline → sanity checks → training
→ LM → evaluation → demo) ran end-to-end on a **font-based stand-in** of the UJI file
(`tests/make_fake_uji.py`). **No real accuracy numbers exist yet** — those come from your Colab run on the
real data.

### 0.3 What only you can do [YOU]

1. Faculty approval of the scope change (§0.1).
2. Download UJIpenchars2 (§2).
3. Collect in-house handwriting (§3) — the longest step, so **start in Week 1**.
4. Run the pipeline on Colab with a GPU (§5–§9).
5. Write the report, rehearse and give the demo (§10–§12).

---

## 1. Local setup and committing the new code [YOU]

1. Open Git Bash in `F:\STUDY_MATERIALS\Sem V\ML\LAB\project\scribbleai`.
2. Create a virtual environment and install (Python 3.11+):
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate          # Git Bash on Windows
   pip install -r requirements.txt
   pip install --no-deps pyctcdecode      # optional; never plain "pip install pyctcdecode"
   python setup_env.py                    # GPU warning is fine locally
   python -m pytest tests -q              # expect: 120 passed, 7 skipped (skips = real UJI not yet present)
   ```
   If PyTorch's default wheel is too big for your laptop, install the CPU build first:
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
3. Commit and push (check `git status` first — `data/`, `checkpoints/*.pt` and `results/` are ignored):
   ```bash
   git add -A
   git commit -m "Implement preprocessing, synthetic lines, BiLSTM-CTC, decoding, evaluation, demo"
   git push
   ```
4. Each teammate: `git pull`, same install, same test run.

**Exit check:** tests green on every laptop; code on GitHub.

---

## 2. UJIpenchars2 (the character data) [YOU]

1. Download from the UCI page (dataset 177, "UJI Pen Characters (Version 2)"), unzip.
2. Put `ujipenchars2.txt` and `uji2.names` in `data/raw/ujipenchars2/`. Upload the same two files to
   Google Drive at `MyDrive/scribbleai/raw/` (the notebook copies them from there).
3. Check:
   ```bash
   python src/preprocessing/parse_uji.py            # expect total=11640 train=6596 val=1164 test=3880
   python -m pytest tests/test_parse_uji.py -q      # the 7 skipped tests now run
   python -m src.preprocessing.visualize uji --n 12
   python -m src.preprocessing.visualize synth --n 6
   ```
4. Open `results/figures/uji_samples.png` and `synth_samples.png`:
   * characters must be **upright**. If they're upside down, set `data.uji_flip_y: true` in
     `configs/base.yaml` and re-plot;
   * synthetic lines should read as words, with sensible letter sizes (ascenders taller, descenders
     below the line, capitals taller than lower case);
   * the "after preprocess()" column should be flat and upright, with evenly spaced dots.
5. Optional but recommended: put an English plain-text file at `data/raw/corpus.txt` (and on Drive),
   e.g. one or two public-domain books from Project Gutenberg with the licence header removed. Synthetic
   line text and the language model then come from real sentences instead of the built-in word-frequency
   list. Pick it **before** the first build and keep it fixed.

---

## 3. In-house data collection [YOU] — start in Week 1

### 3.1 Decide the protocol before anyone writes
* **Writer codes and roles, fixed in advance.** Example with 20 writers: W01–W12 = train (used only for
  the fine-tuning experiment), W13–W20 = test (never trained on, never used for tuning).
  Put the test codes in `configs/base.yaml`:
  ```yaml
  data:
    inhouse_test_writers: ["W13", "W14", "W15", "W16", "W17", "W18", "W19", "W20"]
  ```
  Commit this before collection. Don't change it afterwards.
* **Prompts:** keep the collector's default 37 prompts (words, short and longer lines). Don't change
  them mid-collection. The demo vocabulary is UJI's 97 characters + space, so avoid symbols outside it.
* **Device:** use tablets/phones with stylus or finger where possible (closest to the demo); a mouse
  works but is much less natural. The device is saved with every sample, so it can be analysed later.
* **Consent:** every writer ticks the box; only anonymous codes are stored.

### 3.2 Run the collector
1. Open `demo/collector.html` directly in a browser (Chrome/Edge/Safari on the tablet). To open it on a
   tablet, send the file to it, or serve the folder from a laptop: `python -m http.server 8000` in
   `demo/` and open `http://<laptop-ip>:8000/collector.html` on the tablet (same Wi-Fi).
2. Enter the writer code, hand, device → tick consent → **Start writing**.
3. Write each prompt naturally (not unusually neat) → **Save & next**. Use **Undo stroke** / **Clear** for
   mistakes, **Skip** if a prompt can't be written.
4. At the end press **Export data (.jsonl)**. The file is named `scribbleai_<writer>_<date>.jsonl`.
   Samples are also auto-saved in that browser; **Restore saved samples** recovers them after an
   accidental reload (then continue and export).
5. Collect every file into `data/inhouse/` and upload them to Drive `MyDrive/scribbleai/inhouse/`.

### 3.3 Check quality after the first 2–3 writers (not after 500 samples)
```bash
python -c "from src.preprocessing.parse_inhouse import load_dir; s=load_dir('data/inhouse'); print(len(s), s[0].text, s[0].num_points, round(s[0].duration,2))"
python -m src.preprocessing.visualize inhouse --n 6
```
Look at `results/figures/inhouse_samples.png`: the processed column should be upright and flat.
Target ≈ 20 writers × ≈ 25–37 prompts ≈ 500–700 samples.

---

## 4. Colab and Drive setup [YOU]

1. Colab → *File → Open notebook → GitHub* (or upload) → `notebooks/scribbleai_colab.ipynb`.
2. *Runtime → Change runtime type → T4 GPU*.
3. In cell 1 set `REPO_USER`, run it, paste a GitHub token when asked (it is not stored).
   Expect `setup_env.py` to show the T4 and `[OK]` for every required library.
4. Cell 2 mounts Drive and copies the raw UJI file, the optional corpus and the in-house files.
5. Cell 3 runs the tests. All must pass.

Everything persistent goes to `MyDrive/scribbleai/` (`processed/`, `checkpoints/`, `results/`).

---

## 5. Build the dataset (Phase 1 exit) [YOU run, CODE ✔]

```bash
python -m src.preprocessing.build_dataset
```
Expect:
```
UJI: total=11640 train=6596 val=1164 test=3880
vocab: 97 chars + blank -> vocab_size=99 ...        (97 UJI characters + space + blank)
text source: corpus file | English word-frequency list
saved val: 3000 ... saved test: 5000 ... saved uji_chars_test: 3880 ...
saved inhouse_train / inhouse_test (if files are present)
```
* `configs/base.yaml` now contains the real `vocab_size` and `blank_index` — commit that change.
* Lines that would be CTC-infeasible are skipped and counted; if that number is large, lower
  `preprocessing.resample_step`.
* Any in-house prompt characters outside the vocabulary are reported and dropped from the label.
* Re-run this step whenever preprocessing settings, the corpus, or the in-house files change.

**Phase 1 exit check:** tests pass; features and manifests saved; splits writer-disjoint (checked in code);
baseline number recorded (next step).

---

## 6. Baseline (Objective 2)

```bash
python -m src.models.baseline
```
Prints validation accuracy for k = 1, 3, 5, 7, then test accuracy for k-NN and the MLP on the 20 held-out
writers plus their most common confusions; saves `results/baseline.json`. Record **test accuracy** and
**isolated-character error**: on the same isolated test characters (`uji_chars_test`) the BiLSTM-CTC
model must do clearly better (the evaluation report prints both side by side).

---

## 7. Sanity checks (guide §5.5) — before any real training

```bash
python -m src.sanity_checks --steps 600
```
Checks: (1) overfitting 4 lines (loss → near 0, greedy output == reference); (2) label ids ↔ characters,
blank last and never a label; (3) feature scale; (4) finite loss every step; (5) non-zero gradients in every
layer. Must end with **ALL CHECKS PASSED**.

If (1) fails but the loss is still going down, rerun with `--steps 1200`. CTC usually sits on a plateau
(it outputs only blanks, loss ≈ 3–4) for a while before it drops sharply. If the loss stays flat, the bug
is in the code or the labels, not in the amount of data.

---

## 8. Training (Phase 2) [YOU run]

```bash
python -m src.train --ckpt-dir /content/drive/MyDrive/scribbleai/checkpoints --wandb
```
* Each epoch makes **20,000 new synthetic lines** from the 34 training writers
  (`training.train_source: online`), so the model never sees the same line twice.
  For a fixed training set use `train_source: saved` and rebuild.
* Every epoch prints train loss, val loss, val CER/WER and five REF/HYP examples; W&B gets the same.
  `last.pt` is saved every epoch, `best.pt` whenever val CER improves (atomic writes, so a disconnect never
  leaves a broken file).
* **Disconnected?** Reconnect, re-run notebook cells 1–2, restore `data/processed` from Drive, re-run the
  training cell: it resumes from `last.pt`. `--no-resume` starts over.
* Expect several epochs of "only blanks" (val CER ≈ 1.0) before CER starts falling.
  The LR halves when val CER stops improving (patience 3); training stops early after 8 epochs without
  improvement.
* Rough cost on a T4: a few minutes per epoch; 40 epochs ≈ 2–3 hours. Colab free sessions can end earlier
  — just resume.

**Experiments worth running** (copy `configs/base.yaml` to `configs/expNN_<name>.yaml`, change one thing,
pass `--config`, use a different `--ckpt-dir`, and set `experiment_name`):
1. `learning_rate`: 1e-3 vs 5e-4.
2. `preprocessing.resample_step`: 0.08 / 0.12 / 0.18 (frames per character vs speed). Needs a rebuild.
3. `model.hidden_size`: 128 vs 256 (accuracy vs demo latency).
4. Augmentation off vs on (set the `synthetic` ranges to zero) — shows why augmentation matters.

After the first full run, **set your own targets** (e.g. "greedy CER on the synthetic test set ≤ X%,
in-house CER reported with the gap analysed") and put them in the report.

---

## 9. Decoding, language model and evaluation (Phase 3–4)

### 9.1 Build the language model
```bash
python -m src.decoding.lm
```
Trains a character 6-gram (Witten-Bell smoothing) on `data/raw/corpus.txt` (or the word-frequency sampler),
prints its held-out perplexity, and writes `data/processed/char_lm.json` and `word_3gram.arpa` (the latter
is only used if kenlm happens to be installed).

### 9.2 Tune the LM weight on VALIDATION, never on test
```bash
for a in 0.2 0.5 0.8 1.2; do
  python -m src.evaluate --ckpt CKPT/best.pt --sets val --modes greedy beam_lm --alpha $a --beta 1.0 \
         --max-items 500 --out results/tune_a$a
done
```
Then try β ∈ {0, 0.5, 1, 2} with the best α. Write the winners into `decoding.lm_alpha` /
`decoding.lm_beta` in `configs/base.yaml` (the evaluation and demo read them via `--alpha/--beta` or
`--config`).

### 9.3 Full evaluation
```bash
python -m src.evaluate --ckpt CKPT/best.pt --alpha <best> --beta <best>
```
`results/<run>/summary.md` contains, for the synthetic test set, isolated UJI test characters and in-house
test writers:
* CER / WER / exact-line accuracy for **greedy → beam → beam+LM → beam+LM+spell** (Objectives 4, 5);
* CPU latency per line for each stage (the demo target is < 200 ms per line, greedy);
* beam-width sweep 1/5/10/25 (CER vs ms);
* the **generalisation gap** (in-house CER − synthetic CER) (Objective 6);
* top confusions, hardest characters, worst lines; plots `*_confusion.png`, `*_cer_vs_n_chars.png`,
  `*_cer_vs_n_strokes.png`, and for in-house data `*_cer_vs_chars_per_sec.png` (writing speed) (Objective 7);
* baseline vs sequence model on isolated characters (Objective 2).

Per-line predictions for every decoder are in `*_predictions.csv` — use them for the qualitative
"where and why it fails" section.

### 9.4 In-house fine-tuning experiment (optional, after collection)
```bash
python -m src.train --finetune --init CKPT/best.pt --ckpt-dir CKPT_finetune
python -m src.evaluate --ckpt CKPT_finetune/best.pt --sets inhouse_test test --out results/finetune
```
Fine-tunes on the in-house **train** writers mixed 50/50 with fresh synthetic lines (so it doesn't forget),
then re-evaluates on the **same** held-out in-house test writers. Report both numbers, and check that the
synthetic test CER didn't get much worse.

---

## 10. Live demo [YOU set up, CODE ✔]

### 10.1 Install on the demo laptop
1. Download from Drive into the repo: `checkpoints/best.pt` (or the fine-tuned one) and
   `data/processed/char_lm.json`.
2. Run:
   ```bash
   python demo/app.py --ckpt checkpoints/best.pt --config configs/base.yaml
   ```
   Open **http://localhost:7860**. `--config` makes the demo use your tuned α/β.
3. To write on a **tablet or phone** (much better than a mouse): run with `--host 0.0.0.0`, find the
   laptop's IP (`ipconfig`), and open `http://<laptop-ip>:7860` on the tablet on the same Wi-Fi.
   Allow Python through the Windows firewall when asked. Hotspot from a phone works if the lab Wi-Fi blocks
   device-to-device traffic.
4. `/gradio` (link in the page footer) scores a whole collector `.jsonl` file and shows CER per line —
   useful as a backup if live writing goes badly.

### 10.2 What the demo shows
* Transcription updates after every stroke (debounced ~0.2 s), with preprocess/model/decode/total ms.
* Decoder buttons: Greedy, Beam, Beam + LM, Beam + LM + spell (greyed out until the LM file exists).
* **Compare all decoders** runs all four on the same input, side by side, with decode times.
* **What the model sees**: the normalised, resampled trajectory with pen-up points highlighted —
  good for explaining preprocessing.

### 10.3 Demo-day checklist
- [ ] Run the demo once on the laptop the morning of the presentation, offline (no internet needed).
- [ ] Charger, tablet/stylus, phone hotspot as backup network.
- [ ] Practise 5–6 phrases that work well (short real words; write each letter clearly separated; the
      model was trained on separately written characters, so joined cursive is its weak spot — say so).
- [ ] Have `results/<run>/summary.md` and the plots open in another tab.
- [ ] Backup: a short screen recording of a successful demo run, and the `/gradio` page with a
      pre-collected file.

---

## 11. Suggested presentation flow (10–12 minutes)

1. **Problem** (1 min): online vs offline handwriting; we read the pen path, not pixels.
2. **Scope change** (1 min): IAM-OnDB didn't come through → UJIpenchars2 + synthetic lines + our own
   in-house data. Be upfront; it is an honest finding about data access.
3. **Pipeline** (2 min): preprocessing steps with the `visualize` figures; the 8 features; BiLSTM + CTC
   (why bidirectional: delayed strokes like i-dots and t-crosses; why CTC: no segmentation).
4. **Synthetic lines** (1 min): how characters become lines; why writer-disjoint splits.
5. **Live demo** (3 min): write → text; switch decoders; compare decoders; "what the model sees".
6. **Results** (2 min): baseline vs model; greedy vs beam vs LM vs spell table with latency; the
   generalisation gap synthetic → in-house; confusion matrix highlights.
7. **Limitations and next steps** (1 min): no real cursive in training data, separate-character writing,
   small in-house set; IAM-OnDB if access arrives (parser ready), Bézier features, transformer encoder.

---

## 12. Report outline

1. Introduction and motivation (from the synopsis).
2. Related work (synopsis §3).
3. **Data**: UJIpenchars2 facts (60 writers, 97 chars × 2, (x, y) only, 19.4% duplicate points, writer-disjoint
   34/6/20 split); synthetic line construction (layout table, augmentation); in-house protocol (writers,
   devices, prompts, consent, split). State the IAM-OnDB situation.
4. **Method**: preprocessing (each step + why), features, model, CTC, decoders, LM, spell.
5. **Experiments**: setup (config, hardware, epochs), baseline, main results table, decoder ablation,
   beam sweep, latency, hyperparameter experiments.
6. **Error analysis**: confusion matrix, CER vs length/strokes/speed, worst samples with pictures.
7. **Generalisation**: synthetic vs in-house gap, fine-tuning experiment; explain the gap honestly
   (isolated-character training, no coarticulation or real joins, different devices).
8. Conclusion, limitations, future work. Appendix: config file, test list, dataset card for the released
   in-house data.

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `numpy has no attribute 'long'` / scipy import errors on Colab | something installed numpy < 2 (usually plain `pip install pyctcdecode`). `pip install -U numpy scipy`, restart runtime, use `pip install --no-deps pyctcdecode`. |
| `UJIpenchars2 not found` | file must be at `data/raw/ujipenchars2/ujipenchars2.txt` (notebook cell 2 copies it from Drive). |
| Characters upside down in `uji_samples.png` | `data.uji_flip_y: true`, rebuild. |
| Val CER stuck at 1.0 for many epochs | normal for the first few epochs. If it persists past ~10 epochs: rerun the sanity checks, try `learning_rate: 0.001`, check that `feature_stats.json` came from this build. |
| Loss `nan`/`inf` | `zero_infinity=True` already skips impossible alignments; if many batches are skipped, lower `resample_step`. |
| Training very slow on CPU | Expected: packed LSTMs are slow on CPU. Train on the Colab GPU. |
| `config vocab_size=… but vocab.json has …` | the config and processed data come from different builds: rebuild, or restore the matching `data/processed` from Drive. |
| Demo: "model not reachable" | check the terminal running `demo/app.py`; the checkpoint path must exist. |
| Demo on tablet doesn't load | `--host 0.0.0.0`, same network, Windows firewall allowed, use the laptop's IPv4 address. |
| Demo outputs nonsense for joined-up writing | expected limitation (trained on separate characters). Write letters separately; mention it as a finding. |
| `beam_lm` greyed out | run `python -m src.decoding.lm` (or copy `char_lm.json` into `data/processed/`). |

---

## 14. Four-week plan (revised)

| Week | Do |
|---|---|
| 1 | Faculty approval · §1 local setup and push · §2 UJI download + visual check · §3 decide writer split, start collection · §4 Colab · §5 build · §6 baseline |
| 2 | §7 sanity checks · §8 first full training run · set targets · 1–2 hyperparameter experiments · keep collecting |
| 3 | §9.1–9.2 LM + tuning on val · collection finished → rebuild with in-house data · §9.3 evaluation · §9.4 fine-tuning |
| 4 | Error analysis write-up · §10 demo on the presentation laptop + tablet · report · rehearse §11 twice |
