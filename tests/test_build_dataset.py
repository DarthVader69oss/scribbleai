import json
import numpy as np
import pandas as pd
from src.utils import load_config


def test_build_outputs(built):
    cfg = load_config(built)
    d = cfg["data"]
    v = json.loads(open(d["vocab_file"], encoding="utf-8").read())
    assert cfg["features"]["vocab_size"] == len(v["chars"]) + 1
    assert cfg["decoding"]["blank_index"] == len(v["chars"]) == v["blank"]
    assert " " in v["chars"]
    for name in ("val", "test", "uji_chars_test"):
        df = pd.read_csv(f"{d['processed_dir']}/{name}.csv", keep_default_na=False)
        F = np.load(f"{d['processed_dir']}/{name}_feats.npy")
        assert len(df) > 0 and F.shape[1] == 8
        assert df.n_frames.sum() == len(F) and (df.offset.diff().dropna() > 0).all()


def test_splits_writer_disjoint(built):
    cfg = load_config(built)
    p = cfg["data"]["processed_dir"]
    w = {n: set(pd.read_csv(f"{p}/{n}.csv").writer) for n in ("train_stats_sample", "val", "test")}
    assert not (w["train_stats_sample"] & w["val"]) and not (w["train_stats_sample"] & w["test"])
    assert not (w["val"] & w["test"])
    assert w["val"] <= set(cfg["data"]["uji_val_writers"])


def test_stats_file(built):
    cfg = load_config(built)
    s = json.loads(open(cfg["data"]["stats_file"]).read())
    assert len(s["mean"]) == 8 and all(x > 0 for x in s["std"])
