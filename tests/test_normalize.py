import numpy as np
from src.preprocessing.normalize import (baseline_angle, rotate, slant_angle, shear,
                                         normalize_height, normalize_line, core_height)
from src.preprocessing.clean import ink_points


def zigzag_line(slope=0.0, slant=0.0, n=12):
    """a row of vertical bars (like 'lllll'), optionally tilted and slanted. y down."""
    ink = []
    for i in range(n):
        x = i * 1.0
        top, bot = np.array([x - slant, -1.0]), np.array([x, 0.0])
        s = np.linspace(top, bot, 10)
        s[:, 1] += slope * s[:, 0]
        ink.append(s)
    return ink


def test_baseline_angle_detects_tilt():
    ang = baseline_angle(zigzag_line(slope=np.tan(np.radians(8))))
    assert abs(ang - 8) < 1.0


def test_rotation_removes_tilt():
    ink = zigzag_line(slope=np.tan(np.radians(8)))
    flat = rotate(ink, -baseline_angle(ink))
    assert abs(baseline_angle(flat)) < 0.5


def test_baseline_ignores_single_characters():
    assert baseline_angle([np.array([[0, 0], [0.3, 1]], float)]) == 0.0


def test_slant_detected_and_removed():
    ink = zigzag_line(slant=0.4)          # tops shifted LEFT -> left-leaning
    a = slant_angle(ink)
    assert a > 15
    assert abs(slant_angle(shear(ink, a))) < 2


def test_slant_zero_for_single_char():
    assert slant_angle([np.array([[0, -1], [0.5, 0]], float)]) == 0.0


def test_height_normalised_to_one():
    ink = [s * 37.0 for s in zigzag_line()]
    out = normalize_height(ink)
    assert abs(core_height(out) - 1.0) < 1e-6
    assert ink_points(out)[:, 0].min() == 0


def test_normalize_line_scale_invariant():
    ink = zigzag_line(slope=0.05, slant=0.2)
    a = ink_points(normalize_line(ink))
    b = ink_points(normalize_line([s * 250 + 1000 for s in ink]))
    assert np.allclose(a, b, atol=1e-6)


def test_case_preserved_within_line():
    # a tall glyph next to short glyphs keeps being taller after per-line normalisation
    small = [np.array([[i, 0.0], [i, -1.0]]) for i in range(0, 8, 2)]
    tall = [np.array([[9, 0.0], [9, -2.0]])]
    out = normalize_line(small + tall)
    h = [np.ptp(s[:, 1]) for s in out]
    assert h[-1] > 1.5 * h[0]

def test_slant_angle_flat_stroke_is_zero():
    import numpy as np
    from src.preprocessing.normalize import slant_angle
    ink = [np.stack([np.linspace(0, 1, 20), np.zeros(20)], axis=1)]   # perfectly flat
    assert slant_angle(ink) == 0.0
    assert slant_angle([np.array([[0.5, 0.5]])]) == 0.0               # a single dot
