"""Integration: tiny train run -> checkpoint -> resume -> LM -> evaluate -> demo API."""
import json
import sys
from pathlib import Path

import pytest
import torch

from src.utils import load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ckpt(built):
    from src import train
    cfg = load_config(built)
    train.main(["--config", str(built), "--epochs", "1", "--device", "cpu"])
    p = Path(cfg["training"]["checkpoint_dir"]) / "last.pt"
    assert p.exists() and (p.parent / "best.pt").exists()
    return p


def test_checkpoint_self_contained(ckpt):
    st = torch.load(ckpt, weights_only=False)
    assert {"model", "vocab", "norm_mean", "norm_std", "config", "epoch"} <= set(st)


def test_resume_continues_epochs(built, ckpt):
    from src import train
    train.main(["--config", str(built), "--epochs", "2", "--device", "cpu"])
    st = torch.load(ckpt, weights_only=False)
    assert st["epoch"] == 1 and len(st["history"]) == 2


def test_lm_and_evaluate(built, ckpt, tmp_path):
    from src.decoding import lm
    from src import evaluate
    lm.main(["--config", str(built), "--lines", "3000"])
    # evaluation reads the LM path from the checkpoint's config, which is the same tiny config
    evaluate.main(["--ckpt", str(ckpt), "--sets", "test", "uji_chars_test", "inhouse_test", "--max-items", "8",
                   "--sweep-items", "4", "--out", str(tmp_path), "--alpha", "0.3"])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert set(s["sets"]["test"]["modes"]) == {"greedy", "beam", "beam_lm", "beam_lm_spell"}
    assert s["alpha"] == 0.3 and "generalisation_gap_cer" in s
    assert "chars_per_sec" in s["sets"]["inhouse_test"]["analysis"]
    assert (tmp_path / "summary.md").exists() and (tmp_path / "test_confusion.png").exists()


def test_inhouse_split_and_finetune(built, ckpt, tmp_path):
    import pandas as pd
    from src import train
    cfg = load_config(built)
    p = cfg["data"]["processed_dir"]
    tr, te = pd.read_csv(f"{p}/inhouse_train.csv"), pd.read_csv(f"{p}/inhouse_test.csv")
    assert set(tr.writer) == {"W01"} and set(te.writer) == {"W02"}
    assert (te.duration > 0).all()
    train.main(["--config", str(built), "--finetune", "--init", str(ckpt), "--ckpt-dir", str(tmp_path),
                "--epochs", "1", "--device", "cpu"])
    assert (tmp_path / "best.pt").exists()


def test_recognizer_and_demo_api(ckpt):
    from fastapi.testclient import TestClient
    sys.path.insert(0, str(ROOT / "demo"))
    import app as demo_app
    from src.inference import Recognizer
    rec = Recognizer(ckpt)
    strokes = [[[10, 50, 0.0, 0.5], [20, 20, 0.05, 0.5], [30, 50, 0.1, 0.5]], [[40, 30, 0.3, 0.5], [55, 30, 0.35, 0.5]]]
    r = rec.recognize(strokes, "greedy")
    assert isinstance(r["text"], str) and r["frames"] > 0 and r["ms"]["total"] > 0
    client = TestClient(demo_app.create_app(rec, "last.pt"))
    assert client.get("/").status_code == 200
    info = client.get("/api/info").json()
    assert "greedy" in info["modes"]
    j = client.post("/api/recognize", json={"strokes": strokes, "mode": "greedy", "compare": True}).json()
    assert "text" in j and set(j["all"]) == set(info["modes"])
    assert client.post("/api/recognize", json={"strokes": strokes, "mode": "nope"}).status_code == 400
    assert client.post("/api/recognize", json={"strokes": [], "mode": "greedy"}).json()["text"] == ""


def test_gradio_builds(ckpt):
    sys.path.insert(0, str(ROOT / "demo"))
    import app as demo_app
    from src.inference import Recognizer
    demo_app.build_gradio(Recognizer(ckpt))
