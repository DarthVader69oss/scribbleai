import numpy as np
from src.preprocessing.clean import to_ink, remove_duplicates, remove_hooks, ink_bbox


def test_to_ink_drops_time_pressure_and_empty_strokes():
    ink = to_ink([[(0, 0, 0.1, 0.5), (1, 1, 0.2, 0.5)], []])
    assert len(ink) == 1 and ink[0].shape == (2, 2)


def test_to_ink_flip_y():
    assert to_ink([[(1, 2)]], flip_y=True)[0][0, 1] == -2


def test_to_ink_rejects_empty():
    import pytest
    with pytest.raises(ValueError):
        to_ink([[]])


def test_exact_duplicates_removed():
    s = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [2, 0]], float)
    out = remove_duplicates([s], 0.0)[0]
    assert out.tolist() == [[0, 0], [1, 0], [2, 0]]


def test_near_duplicates_removed_but_endpoints_kept():
    s = np.array([[0, 0], [0.001, 0], [0.5, 0], [0.5005, 0]], float)
    out = remove_duplicates([s], 0.01)[0]
    assert out[0].tolist() == [0, 0] and out[-1].tolist() == [0.5005, 0] and len(out) == 2


def test_single_point_stroke_survives():
    out = remove_duplicates([np.array([[3.0, 4.0]])], 0.1)
    assert out[0].tolist() == [[3.0, 4.0]]


def test_hook_trimmed_at_start():
    # tiny backwards curl, then a long straight stroke to the right
    hook = [[0.3, 0.3], [0.15, 0.15], [0.0, 0.0]]
    line = [[x, 0.0] for x in np.linspace(0.1, 10, 40)]
    s = np.array(hook + line)
    out = remove_hooks([s], 0.10, 90, 12)[0]
    assert len(out) < len(s) and out[0][0] >= 0.0 and out[0][1] == 0.0


def test_short_strokes_untouched():
    s = np.array([[0, 0], [1, 1], [0, 2]], float)
    assert np.array_equal(remove_hooks([s], 0.1, 90, 12)[0], s)


def test_straight_stroke_untouched():
    s = np.array([[x, 0.0] for x in range(30)], float)
    assert np.array_equal(remove_hooks([s])[0], s)


def test_bbox():
    assert ink_bbox([np.array([[1, 2], [3, -1]], float)]) == (1, -1, 3, 2)
