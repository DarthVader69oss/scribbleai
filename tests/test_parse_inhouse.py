import json, os, sys
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing.parse_inhouse import parse_inhouse_file, load_dir, writer_split


def sample(writer="W01", text="the cat", strokes=None):
    strokes = strokes or [[[10, 20, 0.0, 0.5], [12, 22, 0.01, 0.6]], [[30, 20, 0.4, 0.5]]]
    return {"writer": writer, "hand": "right", "device": "mouse", "text": text,
            "canvas": [800, 260], "strokes": strokes}


def write(tmp_path, rows, name="a.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_valid_roundtrip(tmp_path):
    s = parse_inhouse_file(write(tmp_path, [sample()]))
    assert len(s) == 1 and s[0].text == "the cat" and s[0].num_points == 3
    assert abs(s[0].duration - 0.4) < 1e-9


def test_backwards_time_rejected(tmp_path):
    bad = sample(strokes=[[[1, 1, 0.5, 0.5], [2, 2, 0.1, 0.5]]])
    with pytest.raises(ValueError, match="backwards"):
        parse_inhouse_file(write(tmp_path, [bad]))


def test_tiny_timestamp_jitter_is_clamped(tmp_path):
    jitter = sample(strokes=[[[1, 1, 0.500, 0.5], [2, 2, 0.499, 0.5], [3, 3, 0.600, 0.5]]])
    s = parse_inhouse_file(write(tmp_path, [jitter]))
    ts = [p[2] for p in s[0].strokes[0]]
    assert ts == sorted(ts) and ts[1] == 0.5


def test_empty_text_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty text"):
        parse_inhouse_file(write(tmp_path, [sample(text="  ")]))


def test_missing_field_rejected(tmp_path):
    d = sample(); del d["strokes"]
    with pytest.raises(ValueError, match="malformed"):
        parse_inhouse_file(write(tmp_path, [d]))


def test_empty_stroke_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty stroke"):
        parse_inhouse_file(write(tmp_path, [sample(strokes=[[]])]))


def test_load_dir_and_writer_split(tmp_path):
    write(tmp_path, [sample("W01"), sample("W01")], "w1.jsonl")
    write(tmp_path, [sample("W02")], "w2.jsonl")
    allv = load_dir(tmp_path)
    train, test = writer_split(allv, ["W02"])
    assert len(train) == 2 and len(test) == 1
    assert {s.writer for s in train}.isdisjoint({s.writer for s in test})


def test_unknown_test_writer_rejected(tmp_path):
    write(tmp_path, [sample("W01")])
    with pytest.raises(ValueError, match="Unknown writers"):
        writer_split(load_dir(tmp_path), ["W99"])
