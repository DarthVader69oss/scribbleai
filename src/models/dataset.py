"""
src/models/dataset.py

Datasets return (features T x 8 float32 tensor, label ids LongTensor, text, sample_id).

  SavedDataset         -- a set written by build_dataset.py (val, test, inhouse_*, uji_chars_test, train)
  OnlineSynthDataset   -- fresh synthetic lines every epoch from the TRAIN writers (infinite augmentation)
  collate_fn           -- pads features, keeps true lengths, CONCATENATES targets (what nn.CTCLoss wants)
"""
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.pipeline import preprocess
from src.preprocessing.synth_lines import ctc_feasible


class SavedDataset(Dataset):
    def __init__(self, processed_dir, name, vocab, normalizer, max_items=None):
        self.df = pd.read_csv(f"{processed_dir}/{name}.csv", keep_default_na=False,
                              dtype={"text": str, "writer": str})
        if max_items:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)
        self.feats = np.load(f"{processed_dir}/{name}_feats.npy", mmap_mode="r")
        self.vocab, self.norm = vocab, normalizer
        self.name = name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        F = np.asarray(self.feats[r.offset:r.offset + r.n_frames])
        ids = self.vocab.encode(r.text, on_unknown="skip")
        return torch.from_numpy(self.norm(F)), torch.tensor(ids, dtype=torch.long), r.text, r.id


class OnlineSynthDataset(Dataset):
    """Each (epoch, index) pair deterministically produces one new synthetic line."""

    def __init__(self, bank, sampler, builder, vocab, normalizer, pcfg, n_per_epoch, seed=0):
        self.bank, self.sampler, self.builder = bank, sampler, builder
        self.vocab, self.norm, self.pcfg = vocab, normalizer, pcfg
        self.n, self.seed, self.epoch = n_per_epoch, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        for _ in range(20):
            text = self.sampler.sample(rng)
            strokes = self.builder.build(text, rng.choice(self.bank.writers), rng)
            F, _ = preprocess(strokes, self.pcfg)
            ids = self.vocab.encode(text)
            if ctc_feasible(len(F), ids):
                return torch.from_numpy(self.norm(F)), torch.tensor(ids, dtype=torch.long), text, f"online_{i}"
        raise RuntimeError("could not make a CTC-feasible line in 20 tries; lower resample_step")


def collate_fn(batch):
    feats, labels, texts, ids = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    x = torch.zeros(len(feats), int(lengths.max()), feats[0].shape[1])
    for i, f in enumerate(feats):
        x[i, :f.shape[0]] = f
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels) if len(labels) else torch.zeros(0, dtype=torch.long)
    return {"x": x, "lengths": lengths, "targets": targets, "target_lengths": target_lengths,
            "texts": list(texts), "ids": list(ids)}
