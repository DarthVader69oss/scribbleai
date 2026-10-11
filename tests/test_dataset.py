import torch
from src.models.dataset import collate_fn, SavedDataset
from src.preprocessing.vocab import Vocab
from src.preprocessing.features import FeatureNormalizer
from src.utils import load_config


def test_collate_pads_and_concatenates():
    b = [(torch.ones(5, 8), torch.tensor([1, 2]), "ab", "x"), (torch.ones(3, 8), torch.tensor([3]), "c", "y")]
    out = collate_fn(b)
    assert out["x"].shape == (2, 5, 8) and out["x"][1, 3:].abs().sum() == 0
    assert out["lengths"].tolist() == [5, 3]
    assert out["targets"].tolist() == [1, 2, 3] and out["target_lengths"].tolist() == [2, 1]


def test_saved_dataset_items(built):
    cfg = load_config(built)
    d = cfg["data"]
    ds = SavedDataset(d["processed_dir"], "val", Vocab.load(d["vocab_file"]), FeatureNormalizer.load(d["stats_file"]))
    x, ids, text, _ = ds[0]
    assert x.shape[1] == 8 and x.dtype == torch.float32 and len(ids) == len(text)
    assert x.shape[0] >= len(ids)


def test_online_dataset_deterministic(built):
    from src.train import build_online_train
    cfg = load_config(built)
    d = cfg["data"]
    v, n = Vocab.load(d["vocab_file"]), FeatureNormalizer.load(d["stats_file"])
    ds = build_online_train(cfg, v, n, 4, seed=1)
    a, b = ds[2], ds[2]
    assert a[2] == b[2] and torch.equal(a[0], b[0])
    ds.set_epoch(1)
    assert len(ds) == 4 and isinstance(ds[2][2], str)
