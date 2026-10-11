import numpy as np
from src.preprocessing.features import encode, FeatureNormalizer, FEATURE_NAMES


def ink():
    return [np.array([[0, 0], [1, 0], [2, 0], [2, 1]], float), np.array([[4, 0], [5, 0]], float)]


def test_shape_and_names():
    F = encode(ink())
    assert F.shape == (6, 8) and len(FEATURE_NAMES) == 8 and F.dtype == np.float32


def test_pen_up_and_stroke_end():
    F = encode(ink())
    assert F[:, 2].tolist() == [0, 0, 0, 0, 1, 0]
    assert F[:, 7].tolist() == [0, 0, 0, 1, 0, 1]


def test_offsets():
    F = encode(ink())
    assert F[1, 0] == 1 and F[3, 1] == 1 and F[4, 0] == 2 and F[4, 1] == -1


def test_direction_is_unit_and_no_wraparound():
    F = encode(ink())
    assert np.allclose(F[:, 3] ** 2 + F[:, 4] ** 2, 1)
    assert np.allclose(F[1, 3:5], [0, 1])     # moving right: sin 0, cos 1


def test_curvature_signed_at_turn():
    F = encode(ink())
    assert abs(F[2, 5]) == 0.5 and F[1, 5] == 0 and F[0, 5] == 0


def test_single_point():
    F = encode([np.array([[0.0, 0.0]])])
    assert F.shape == (1, 8) and np.isfinite(F).all()


def test_normalizer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    fs = [rng.normal(3, 2, size=(50, 8)).astype(np.float32) for _ in range(5)]
    n = FeatureNormalizer.fit(fs)
    X = np.concatenate([n(f) for f in fs])
    assert abs(X[:, 0].mean()) < 1e-4 and abs(X[:, 0].std() - 1) < 1e-3
    assert np.allclose(X[:, 2], np.concatenate(fs)[:, 2])          # binary/sincos dims untouched
    n.save(tmp_path / "s.json")
    n2 = FeatureNormalizer.load(tmp_path / "s.json")
    assert np.allclose(n2(fs[0]), n(fs[0]))
