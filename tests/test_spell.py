from src.decoding.spell import SpellCorrector


def test_corrects_oov_keeps_case_and_punct():
    sc = SpellCorrector()
    assert sc("Recieve this wrld, becuase QUIKC") == "Receive this world, because QUICK"
    assert sc("teh") in ("the", "ten", "tea", "tech", "eh")


def test_leaves_known_numbers_and_short():
    sc = SpellCorrector()
    assert sc("the cat 2026 a") == "the cat 2026 a"


def test_extra_words():
    assert SpellCorrector(["scribbleai"])("scribbleai") == "scribbleai"
