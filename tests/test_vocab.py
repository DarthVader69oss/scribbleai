import pytest
from src.preprocessing.vocab import Vocab


def test_from_labels_adds_space_blank_last():
    v = Vocab.from_labels(["b", "a", "a", "Ñ"])
    assert v.chars == ["a", "b", "Ñ", " "] and v.blank == 4 and v.size == 5


def test_roundtrip_and_unknown():
    v = Vocab.from_labels("abc")
    assert v.decode(v.encode("ab c")) == "ab c"
    with pytest.raises(KeyError):
        v.encode("abz")
    assert v.encode("abz", on_unknown="skip") == v.encode("ab")


def test_normalize_text():
    v = Vocab.from_labels("abc'")
    assert v.normalize_text("  a’b   c ") == "a'b c"


def test_decode_skips_blank():
    v = Vocab.from_labels("ab")
    assert v.decode([0, v.blank, 1]) == "ab"


def test_save_load(tmp_path):
    v = Vocab.from_labels("xyzé")
    v.save(tmp_path / "v.json")
    assert Vocab.load(tmp_path / "v.json").chars == v.chars
