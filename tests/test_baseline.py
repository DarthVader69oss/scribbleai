import numpy as np
from src.models.baseline import char_vector, N_POINTS, top_confusions


def test_vector_shape_and_scale_invariance():
    s = [[(0, 0), (10, 20), (20, 0)], [(5, 10), (15, 10)]]
    v = char_vector(s)
    assert v.shape == (N_POINTS * 3 + 2,)
    big = [[(x * 7 + 100, y * 7 - 50) for x, y in st] for st in s]
    assert np.allclose(v, char_vector(big), atol=1e-5)


def test_pen_up_flag_present():
    v = char_vector([[(0, 0), (10, 0)], [(0, 10), (10, 10)]])
    assert v[2 * N_POINTS:3 * N_POINTS].sum() > 0


def test_single_point_char():
    assert np.isfinite(char_vector([[(3, 3)]])).all()


def test_top_confusions():
    assert top_confusions(list("aab"), list("abb"))[0] == {"true": "a", "pred": "b", "count": 1}
