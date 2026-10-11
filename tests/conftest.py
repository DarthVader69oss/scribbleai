"""Shared fixtures: a tiny stand-in UJI file + a temp config (never touches real data)."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fake_uji(tmp_path_factory):
    out = tmp_path_factory.mktemp("uji") / "ujipenchars2.txt"
    subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_uji.py"), str(out),
                    "--writers", "12", "--chars", "30"], check=True, capture_output=True)
    return out


def _write_inhouse(folder):
    """Two fake writers with timestamps, like collector exports."""
    import json, math, random
    rng = random.Random(0)
    for w in ("W01", "W02"):
        rows = []
        for k, text in enumerate(["abc", "bad cab", "face", "Bead"]):
            strokes, t = [], 0.0
            for j, _ in enumerate(text.replace(" ", "")):
                pts = []
                for i in range(12):
                    a = i / 11 * 2 * math.pi
                    pts.append([40 * j + 15 * math.cos(a) + rng.random(), 50 + 15 * math.sin(a), round(t, 3), 0.5])
                    t += 0.01
                strokes.append(pts)
                t += 0.2
            rows.append({"writer": w, "hand": "right", "device": "tablet-stylus", "text": text,
                         "canvas": [800, 260], "strokes": strokes})
        (folder / f"scribbleai_{w}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory, fake_uji):
    """A config pointing at temp dirs, with a tiny model and tiny synthetic sets."""
    import yaml
    d = tmp_path_factory.mktemp("proj")
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    cfg["data"].update({"uji_file": str(fake_uji), "processed_dir": str(d / "processed"),
                        "inhouse_dir": str(d / "inhouse"), "vocab_file": str(d / "processed" / "vocab.json"),
                        "stats_file": str(d / "processed" / "feature_stats.json"),
                        "corpus_file": str(d / "no_corpus.txt"), "uji_val_writers": [1, 2]})
    (d / "inhouse").mkdir()
    _write_inhouse(d / "inhouse")
    cfg["data"]["inhouse_test_writers"] = ["W02"]
    cfg["model"].update({"hidden_size": 16, "num_layers": 1})
    cfg["synthetic"].update({"max_words": 2, "max_chars": 10})
    cfg["training"].update({"lines_per_epoch": 16, "batch_size": 8, "num_epochs": 1, "num_workers": 0,
                            "checkpoint_dir": str(d / "ckpt")})
    cfg["decoding"].update({"char_lm_file": str(d / "processed" / "char_lm.json"),
                            "word_arpa_file": str(d / "processed" / "w.arpa"), "char_lm_order": 3})
    p = d / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def built(tiny_cfg):
    from src.preprocessing import build_dataset
    build_dataset.main(["--config", str(tiny_cfg), "--quick"])
    return tiny_cfg
