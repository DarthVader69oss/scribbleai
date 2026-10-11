"""
tests/test_parse_uji.py  --  run with:  pytest tests/test_parse_uji.py -v

Real-data tests need data/raw/ujipenchars2/ujipenchars2.txt (skipped if absent).
Structural-failure tests use small temp files and always run.
"""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.parse_uji import parse_uji_file, make_splits

REAL = Path(__file__).parent.parent / "data" / "raw" / "ujipenchars2" / "ujipenchars2.txt"
needs_data = pytest.mark.skipif(not REAL.exists(), reason="UJIpenchars2 not downloaded")


@pytest.fixture(scope="module")
def samples():
    return parse_uji_file(REAL)


# ---------- real data ----------
@needs_data
def test_sample_count(samples):
    assert len(samples) == 11640          # 60 writers x 97 chars x 2 reps


@needs_data
def test_class_count(samples):
    assert len({s.label for s in samples}) == 97


@needs_data
def test_utf8_labels_survive(samples):
    labels = {s.label for s in samples}
    assert {"ñ", "Ñ", "€", "¿", "¡", "ü"} <= labels


@needs_data
def test_every_sample_has_valid_points(samples):
    for s in samples:
        assert s.num_strokes >= 1
        for st in s.strokes:
            assert len(st) >= 1
            for x, y in st:
                assert not (math.isnan(x) or math.isnan(y))


@needs_data
def test_each_writer_char_has_two_repetitions(samples):
    from collections import Counter
    c = Counter((s.writer, s.split, s.label) for s in samples)
    assert set(c.values()) == {2}


@needs_data
def test_splits_sizes_and_disjoint_writers(samples):
    train, val, test = make_splits(samples)
    assert len(train) + len(val) + len(test) == len(samples)
    w = lambda xs: {s.writer for s in xs}
    assert not (w(train) & w(val)) and not (w(train) & w(test)) and not (w(val) & w(test))
    assert len(w(test)) == 20


@needs_data
def test_make_splits_rejects_unknown_val_writers(samples):
    with pytest.raises(ValueError, match="matched no training writers"):
        make_splits(samples, val_writers=(999,))


# ---------- structural failures ----------
def _w(tmp_path, body):
    p = tmp_path / "x.txt"
    p.write_text(body, encoding="utf-8")
    return p


def test_valid_minimal_file(tmp_path):
    p = _w(tmp_path, "// c\nWORD a trn_UJI_W01-01\n  NUMSTROKES 1\n  POINTS 2 # 1 2 3 4\n")
    s = parse_uji_file(p)
    assert len(s) == 1 and s[0].strokes == [[(1.0, 2.0), (3.0, 4.0)]]
    assert (s[0].split, s[0].writer, s[0].repetition) == ("trn", 1, 1)


def test_point_count_mismatch_raises(tmp_path):
    p = _w(tmp_path, "WORD a trn_UJI_W01-01\n  NUMSTROKES 1\n  POINTS 3 # 1 2 3 4\n")
    with pytest.raises(ValueError, match="POINTS says 3"):
        parse_uji_file(p)


def test_stroke_count_mismatch_raises(tmp_path):
    p = _w(tmp_path, "WORD a trn_UJI_W01-01\n  NUMSTROKES 2\n  POINTS 1 # 1 2\n")
    with pytest.raises(ValueError, match="NUMSTROKES=2"):
        parse_uji_file(p)


def test_bad_sample_id_raises(tmp_path):
    p = _w(tmp_path, "WORD a weird_id\n  NUMSTROKES 1\n  POINTS 1 # 1 2\n")
    with pytest.raises(ValueError, match="Unrecognised sample id"):
        parse_uji_file(p)


def test_unknown_line_raises(tmp_path):
    p = _w(tmp_path, "WORD a trn_UJI_W01-01\n  GARBAGE\n")
    with pytest.raises(ValueError, match="unrecognised line"):
        parse_uji_file(p)


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="No samples"):
        parse_uji_file(_w(tmp_path, "// only comments\n"))
