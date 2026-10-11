import random
import numpy as np
import pytest
from src.preprocessing.parse_uji import UJISample
from src.preprocessing.synth_lines import (char_layout, CharBank, TextSampler, LineBuilder, ctc_feasible)
from src.preprocessing.pipeline import preprocess
from src.utils import load_config


def fake_samples(writers=(1, 2), labels="abcgl.A"):
    out = []
    for w in writers:
        for k, c in enumerate(labels):
            for rep in (1, 2):
                h = 30 + 10 * (c in "lA") + 5 * w
                stroke = [(0, 0), (10 + k, h), (20, 0)]
                out.append(UJISample(c, f"trn_UJI_W{w:02d}-{rep:02d}", "trn", "UJI", w, rep, [stroke, [(5, 5)]]))
    return out


CFG = load_config("configs/base.yaml")


def test_layout_categories():
    assert char_layout("a")[:2] == (-1.0, 0.0)
    assert char_layout("l")[0] < -1.5 and char_layout("g")[1] > 0.5
    assert char_layout("A")[0] < -1.5 and char_layout("7") == char_layout("A")
    assert char_layout("á")[0] < char_layout("a")[0]          # accent sits higher
    assert char_layout(".")[0] > -0.3                          # full stop near the baseline


def test_bank_is_per_writer():
    b = CharBank(fake_samples())
    assert b.writers == [1, 2] and set(b.labels) == set("abcgl.A")
    assert len(b.bank[1]["a"]) == 2


def test_builder_uses_one_writer_and_space_gaps():
    b = CharBank(fake_samples())
    lb = LineBuilder(b, CFG["synthetic"])
    rng = random.Random(0)
    strokes = lb.build("ab ca", 1, rng)
    assert len(strokes) == 4 * 2                  # 4 glyphs x 2 strokes
    left = [s[:, 0].min() for s in strokes[::2]]
    assert left == sorted(left)


def test_relative_size_kept():
    cfg = dict(CFG["synthetic"], slant_deg=[0, 0], rotate_deg=[0, 0], scale=[1, 1],
               char_scale_jitter=0.0, baseline_wobble=0.0, point_jitter=0.0)
    lb = LineBuilder(CharBank(fake_samples()), cfg)
    s = lb.build("aA", 1, random.Random(0))
    ha = np.ptp(np.concatenate(s[:2])[:, 1]); hA = np.ptp(np.concatenate(s[2:])[:, 1])
    assert hA > 1.4 * ha


def test_unknown_char_raises():
    lb = LineBuilder(CharBank(fake_samples()), CFG["synthetic"])
    with pytest.raises(KeyError):
        lb.build("z", 1, random.Random(0))


def test_text_sampler_only_allowed_chars():
    ts = TextSampler(list("abcdefghijklmnopqrstuvwxyz") + [" "], None, 1, 4, 30, 0.1)
    rng = random.Random(1)
    for _ in range(200):
        t = ts.sample(rng)
        assert t and set(t) <= set("abcdefghijklmnopqrstuvwxyz ") and len(t) <= 30
        assert "  " not in t and t == t.strip()


def test_text_sampler_corpus(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(" ".join(["hello world foo bar"] * 400), encoding="utf-8")
    ts = TextSampler(list("abcdefghijklmnopqrstuvwxyz "), p, 2, 2, 40, 0.0)
    assert ts.corpus_words and set(ts.sample(random.Random(0)).split()) <= {"hello", "world", "foo", "bar"}


def test_ctc_feasible():
    assert ctc_feasible(3, [1, 2, 3]) and not ctc_feasible(3, [1, 1, 2])
    assert ctc_feasible(4, [1, 1, 2])


def test_synthetic_line_survives_pipeline():
    lb = LineBuilder(CharBank(fake_samples()), CFG["synthetic"])
    st = lb.build("gl.Ab ca", 2, random.Random(3))
    F, ink = preprocess(st, CFG["preprocessing"])
    assert F.shape[1] == 8 and np.isfinite(F).all() and F[:, 2].sum() == len(ink) - 1
